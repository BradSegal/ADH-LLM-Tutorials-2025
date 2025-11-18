"""
PhysioNet 2019 Sepsis Challenge Data Pipeline.

This module implements the full data ingestion pipeline that powers the
sequence modeling notebooks. It downloads the raw data (favoring the public
S3 mirror), parses every patient file, imputes missing values, scales
features, and enforces a strict schema contract before handing results back
to notebooks.
"""

from __future__ import annotations

import getpass
import logging
import os
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import requests
from botocore import UNSIGNED
from botocore.client import BaseClient
from botocore.client import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from core.config import PhysioNetConfig, load_data_config

logger = logging.getLogger(__name__)

# Resolve config path relative to repository root (not CWD)
# This ensures notebooks work regardless of working directory
_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parent.parent  # core/data -> core -> root
CONFIG_PATH = _REPO_ROOT / "configs" / "data.yaml"
DEFAULT_S3_WORKERS = int(os.getenv("PHYSIONET_S3_WORKERS", "8"))

# Define the 34 physiological feature columns for the dataset
FEATURE_COLUMNS = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
]

IDENTIFIER_COLUMNS = ["patient_id"]
STATIC_COLUMNS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
TARGET_COLUMNS = ["SepsisLabel"]
TIMING_COLUMNS = ["ICULOS"]

REQUIRED_COLUMNS = set(
    FEATURE_COLUMNS
    + IDENTIFIER_COLUMNS
    + STATIC_COLUMNS
    + TARGET_COLUMNS
    + TIMING_COLUMNS
)
NUMERIC_COLUMNS = set(
    FEATURE_COLUMNS + STATIC_COLUMNS + TARGET_COLUMNS + TIMING_COLUMNS
)


class SepsisDataFrame(BaseModel):
    """Schema contract for the PhysioNet Sepsis DataFrame."""

    data: pd.DataFrame = Field(..., description="Validated Sepsis dataset.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("data")
    def _validate_dataframe(cls, value: pd.DataFrame) -> pd.DataFrame:
        missing_cols = REQUIRED_COLUMNS - set(value.columns)
        if missing_cols:
            raise ValueError(
                "Sepsis dataset is missing required columns: "
                f"{', '.join(sorted(missing_cols))}"
            )

        if value.isna().sum().sum() > 0:
            raise ValueError("Sepsis dataset contains NaN values after preprocessing.")

        for column in NUMERIC_COLUMNS:
            if not is_numeric_dtype(value[column]):
                raise TypeError(f"Column '{column}' must be numeric.")

        if not set(value["SepsisLabel"].unique()).issubset({0, 1}):
            raise ValueError("SepsisLabel column must be binary (0 or 1).")

        return value


class StandardScalerStats(BaseModel):
    """Portable representation of a fitted StandardScaler and feature medians."""

    feature_names: list[str]
    means: list[float]
    scales: list[float]
    feature_medians: list[float]

    def to_scaler(self) -> StandardScaler:
        scaler = StandardScaler()
        scaler.mean_ = np.array(self.means, dtype=float)
        scaler.scale_ = np.array(self.scales, dtype=float)
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = len(self.feature_names)
        scaler.feature_names_in_ = np.array(self.feature_names, dtype=object)
        return scaler

    def to_preprocessor(self) -> tuple[StandardScaler, dict[str, float]]:
        scaler = self.to_scaler()
        medians = dict(zip(self.feature_names, self.feature_medians, strict=True))
        return scaler, medians

    @classmethod
    def from_scaler(
        cls,
        scaler: StandardScaler,
        feature_names: list[str],
        feature_medians: list[float],
    ) -> StandardScalerStats:
        return cls(
            feature_names=feature_names,
            means=scaler.mean_.tolist(),
            scales=scaler.scale_.tolist(),
            feature_medians=feature_medians,
        )


def _save_scaler_stats(
    scaler: StandardScaler,
    path: Path,
    feature_medians: list[float],
) -> None:
    stats = StandardScalerStats.from_scaler(
        scaler,
        FEATURE_COLUMNS,
        feature_medians=feature_medians,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stats.model_dump_json(indent=2))


def _load_scaler_stats(path: Path) -> StandardScaler:
    """
    Deprecated: retained for backward compatibility.

    Use :func:`load_feature_preprocessor` to obtain both scaler and medians.
    """
    scaler, _ = load_feature_preprocessor(path)
    return scaler


def load_feature_preprocessor(path: Path) -> tuple[StandardScaler, dict[str, float]]:
    """
    Load the scaler and feature medians used during training-time preprocessing.

    Parameters
    ----------
    path : Path
        Path to the serialized scaler stats JSON file.

    Returns
    -------
    tuple[StandardScaler, dict[str, float]]
        The fitted StandardScaler and a mapping of feature medians used to
        impute any remaining missing values at inference time.
    """
    stats = StandardScalerStats.model_validate_json(path.read_text())
    return stats.to_preprocessor()


def _load_config(config_path: Path | None = None) -> PhysioNetConfig:
    """
    Load and validate the PhysioNet configuration block.

    Parameters
    ----------
    config_path:
        Optional override for the configuration path, used primarily in tests.

    Returns
    -------
    PhysioNetConfig
        Typed configuration object.
    """
    data_config = load_data_config(config_path or CONFIG_PATH)
    physionet_config = data_config.physionet_2019
    physionet_config.ensure_directories()
    return physionet_config


def _create_unsigned_s3_client() -> BaseClient:
    """Create an unsigned boto3 S3 client for public downloads."""
    return boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))


def _download_s3_object(
    client: BaseClient,
    bucket: str,
    key: str,
    size: int,
    destination: Path,
    *,
    show_progress: bool = True,
) -> None:
    """
    Download a single S3 object with resumable support.

    Parameters
    ----------
    client:
        boto3 client used for the transfer.
    bucket:
        S3 bucket name.
    key:
        Object key to download.
    size:
        Size of the remote object in bytes.
    destination:
        Target filesystem path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(f"{destination.suffix}.part")

    if destination.exists() and destination.stat().st_size == size:
        return

    if destination.exists():
        destination.unlink()

    if tmp_path.exists() and tmp_path.stat().st_size == size:
        tmp_path.replace(destination)
        return

    downloaded_bytes = 0
    if tmp_path.exists():
        current_size = tmp_path.stat().st_size
        if current_size > size:
            tmp_path.unlink()
        else:
            downloaded_bytes = current_size

    range_kwargs = {"Range": f"bytes={downloaded_bytes}-"} if downloaded_bytes else {}

    try:
        response = client.get_object(Bucket=bucket, Key=key, **range_kwargs)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to download {key} from S3: {exc}") from exc

    body = response["Body"]
    chunk_size = 1024 * 1024
    mode = "ab" if downloaded_bytes else "wb"

    with (
        tqdm(
            total=size,
            initial=downloaded_bytes,
            unit="iB",
            unit_scale=True,
            desc=f"S3:{Path(key).name}",
            leave=False,
            disable=not show_progress,
        ) as progress,
        open(tmp_path, mode) as file_obj,
    ):
        while True:
            chunk = body.read(chunk_size)
            if not chunk:
                break
            file_obj.write(chunk)
            progress.update(len(chunk))

    body.close()
    tmp_path.replace(destination)


def _download_via_public_s3(
    config: PhysioNetConfig,
    *,
    client: BaseClient | None = None,
    max_workers: int | None = None,
) -> bool:
    """
    Attempt to download data via the anonymous public S3 mirror.

    Parameters
    ----------
    config:
        Validated configuration with bucket/prefix info.
    client:
        Optional boto3 client injection for testing.

    Returns
    -------
    bool
        True when at least one file was downloaded successfully.
    """
    s3_client = client or _create_unsigned_s3_client()
    prefix = config.s3_prefix.rstrip("/") + "/"
    target_root = config.raw_dir / "training"
    paginator = s3_client.get_paginator("list_objects_v2")

    try:
        pages = paginator.paginate(Bucket=config.s3_bucket, Prefix=prefix)
    except (BotoCoreError, ClientError) as exc:
        logger.warning("Failed to enumerate S3 objects: %s", exc)
        return False

    max_workers = max_workers or max(1, DEFAULT_S3_WORKERS)
    download_futures: list[Future[None]] = []
    print("\n" + "=" * 70)
    print("Downloading PhysioNet Data (Public S3 Mirror)")
    print("=" * 70)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for page in pages:
            contents = page.get("Contents", [])
            for obj in contents:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                relative_path = key[len(prefix) :]
                destination = target_root / relative_path
                if destination.exists() and destination.stat().st_size == obj["Size"]:
                    continue

                download_futures.append(
                    executor.submit(
                        _download_s3_object,
                        s3_client,
                        config.s3_bucket,
                        key,
                        obj["Size"],
                        destination,
                        show_progress=False,
                    )
                )

        if not download_futures:
            print("Public S3 mirror did not yield new files.\n")
            return False

        with tqdm(
            total=len(download_futures),
            unit="file",
            desc="S3 files",
            leave=False,
        ) as pool_progress:
            for future in as_completed(download_futures):
                try:
                    future.result()
                except RuntimeError as exc:  # noqa: BLE001
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(f"S3 download failed: {exc}") from exc
                pool_progress.update(1)

    print("✓ Data downloaded successfully via public S3.\n")
    return True


def _prompt_for_credentials() -> tuple[str, str]:
    """
    Retrieve PhysioNet credentials from the environment or prompt securely.

    Environment variables:
    - PHYSIONET_USERNAME
    - PHYSIONET_PASSWORD
    """
    username = (
        os.getenv("PHYSIONET_USERNAME")
        or input("Enter your PhysioNet username: ").strip()
    )
    password = os.getenv("PHYSIONET_PASSWORD") or getpass.getpass(
        "Enter your PhysioNet password: "
    )

    if not username or not password:
        raise ValueError("Username and password cannot be empty.")
    return username, password


def _download_via_credentials(config: PhysioNetConfig) -> None:
    """
    Download the dataset using credentialed HTTPS endpoint.

    Parameters
    ----------
    config:
        Validated PhysioNet configuration object.

    Raises
    ------
    ValueError, requests.exceptions.HTTPError, zipfile.BadZipFile, AssertionError
    """
    print("\n" + "=" * 70)
    print("PhysioNet Data Download (Credentials Required)")
    print("=" * 70)
    print(
        "Attempting credentialed download via PhysioNet API. "
        "You can create an account at https://physionet.org/register/ if needed.\n"
    )

    username, password = _prompt_for_credentials()
    zip_path = config.raw_dir / "training.zip"
    existing_size = zip_path.stat().st_size if zip_path.exists() else 0

    def _issue_request(range_header: str | None) -> requests.Response:
        headers = {"Range": range_header} if range_header else None
        response = requests.get(
            str(config.download_url),
            auth=(username, password),
            stream=True,
            timeout=300,
            headers=headers,
        )
        response.raise_for_status()
        return response

    range_header = f"bytes={existing_size}-" if existing_size else None
    try:
        response = _issue_request(range_header)
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 401:
            raise requests.exceptions.HTTPError(
                "Authentication failed. Please verify your PhysioNet credentials."
            ) from exc
        raise

    if existing_size and response.status_code == 200:
        response.close()
        zip_path.unlink(missing_ok=True)
        existing_size = 0
        response = _issue_request(None)

    if response.status_code not in {200, 206}:
        raise requests.exceptions.HTTPError(
            f"Unexpected status code {response.status_code} during download."
        )

    print(f"\nDownloading data from {config.download_url}...")
    remaining_size = int(response.headers.get("content-length", 0))
    if "content-range" in response.headers:
        total_size = int(response.headers["content-range"].split("/")[-1])
    else:
        total_size = existing_size + remaining_size

    mode = "ab" if existing_size else "wb"

    with (
        open(zip_path, mode) as zip_file,
        tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc="Downloading",
            initial=existing_size,
        ) as progress,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            written = zip_file.write(chunk)
            progress.update(written)

    response.close()

    print("\nExtracting files...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(config.raw_dir)
    except zipfile.BadZipFile as exc:
        raise zipfile.BadZipFile(
            "Downloaded file is not a valid ZIP archive. "
            "Please try again or download manually from PhysioNet."
        ) from exc
    finally:
        if zip_path.exists():
            zip_path.unlink()

    training_dir = config.raw_dir / "training"
    if not any(training_dir.glob("**/*.psv")):
        raise AssertionError("Extraction failed: no .psv files were found.")

    print("✓ Data downloaded and extracted successfully.\n")


def _download_data_if_needed(
    config: PhysioNetConfig, *, s3_client: BaseClient | None = None
) -> None:
    """
    Download and extract PhysioNet 2019 data if not already present.

    Parameters
    ----------
    config:
        Validated PhysioNet configuration.
    s3_client:
        Optional boto client injection for deterministic tests.
    """
    training_dir = config.raw_dir / "training"
    if any(training_dir.glob("**/*.psv")):
        logger.info("Raw data already exists. Skipping download.")
        print("✓ Raw data already downloaded.")
        return

    if _download_via_public_s3(config, client=s3_client):
        return

    _download_via_credentials(config)


def _parse_and_preprocess_data(config: PhysioNetConfig) -> pd.DataFrame:
    """
    Parse all .psv files and perform preprocessing.

    Parameters
    ----------
    config:
        Validated PhysioNet configuration with scaler path info.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with no missing values and scaled features.
    """
    raw_dir = config.raw_dir
    print("\n" + "=" * 70)
    print("Data Preprocessing Pipeline")
    print("=" * 70)

    print("\n[1/4] Parsing patient files...")
    psv_files = list((raw_dir / "training").glob("**/*.psv"))
    if not psv_files:
        raise FileNotFoundError(
            f"No .psv files found in {raw_dir / 'training'}. "
            "Please ensure the data was downloaded and extracted correctly."
        )

    all_patient_dfs: list[pd.DataFrame] = []
    for psv_file in tqdm(psv_files, desc="Parsing files"):
        patient_id = psv_file.stem.replace("p", "")
        try:
            df = pd.read_csv(psv_file, sep="|")
            df["patient_id"] = patient_id
            all_patient_dfs.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to parse %s: %s", psv_file, exc)
            raise ValueError(f"Error parsing {psv_file}: {exc}") from exc

    full_df = pd.concat(all_patient_dfs, ignore_index=True)
    print(f"  ✓ Parsed {len(all_patient_dfs)} patient files")
    print(f"  ✓ Total records: {len(full_df):,}")

    print("\n[2/4] Imputing missing values...")
    initial_nan_count = full_df.isna().sum().sum()
    print(f"  • Initial NaN count: {initial_nan_count:,}")

    for column in full_df.columns:
        if column == "patient_id":
            continue
        full_df[column] = full_df.groupby("patient_id")[column].transform(
            lambda series: series.ffill().bfill()
        )

    after_ffill_bfill_nan_count = full_df.isna().sum().sum()
    print(f"  • After ffill/bfill: {after_ffill_bfill_nan_count:,} NaNs remaining")

    if after_ffill_bfill_nan_count > 0:
        print("\n[3/4] Applying global median imputation for remaining NaNs...")
        for column in NUMERIC_COLUMNS:
            if column in full_df.columns and full_df[column].isna().any():
                median_value = full_df[column].median()
                if pd.isna(median_value):
                    logger.warning(
                        "Column '%s' median is NaN (all values missing). "
                        "Falling back to zero.",
                        column,
                    )
                    median_value = 0.0
                missing_count = full_df[column].isna().sum()
                full_df[column] = full_df[column].fillna(median_value)
                logger.info(
                    "Imputed %s NaNs in '%s' with median %.2f",
                    missing_count,
                    column,
                    median_value,
                )

        final_nan_count = full_df.isna().sum().sum()
        print(f"  ✓ Final NaN count: {final_nan_count}")
        if final_nan_count > 0:
            remaining = full_df.columns[full_df.isna().any()].tolist()
            raise ValueError(
                "Preprocessing failed: NaNs remain after all imputation steps: "
                + ", ".join(remaining)
            )
    else:
        print("\n[3/4] No global median imputation needed (all NaNs resolved)")

    print("\n[4/4] Scaling features...")
    missing_features = set(FEATURE_COLUMNS) - set(full_df.columns)
    if missing_features:
        raise ValueError(
            "Expected feature columns are missing: "
            f"{', '.join(sorted(missing_features))}"
        )

    feature_medians_list = full_df[FEATURE_COLUMNS].median().tolist()
    scaler = StandardScaler()
    full_df[FEATURE_COLUMNS] = scaler.fit_transform(full_df[FEATURE_COLUMNS])

    _save_scaler_stats(scaler, config.scaler_file, feature_medians_list)
    print(f"  ✓ Scaler saved to {config.scaler_file}")

    scaled_means = full_df[FEATURE_COLUMNS].mean()
    scaled_stds = full_df[FEATURE_COLUMNS].std()
    if not (abs(scaled_means.mean()) < 0.1 and abs(scaled_stds.mean() - 1.0) < 0.1):
        logger.warning(
            "Scaling may be off. Mean of means %.4f, mean of stds %.4f",
            scaled_means.mean(),
            scaled_stds.mean(),
        )

    print("  ✓ Features scaled successfully")
    print(f"\n{'=' * 70}")
    print("Preprocessing Complete")
    print(f"{'=' * 70}\n")

    validated_df = SepsisDataFrame(data=full_df).data
    return validated_df


def get_sepsis_data(config_path: Path | None = None) -> pd.DataFrame:
    """
    Main orchestrator for the PhysioNet 2019 Sepsis data pipeline.

    Parameters
    ----------
    config_path:
        Optional path override primarily for tests.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame ready for downstream modeling.
    """
    config = _load_config(config_path)
    processed_file = config.processed_file

    if processed_file.exists():
        print("=" * 70)
        print("Loading cached preprocessed data...")
        print("=" * 70)
        cached_df = pd.read_parquet(processed_file)
        validated = SepsisDataFrame(data=cached_df).data
        print(f"✓ Loaded {len(validated):,} records from cache")
        print(f"✓ Shape: {validated.shape}")
        print(f"✓ No missing values: {validated.isna().sum().sum() == 0}")
        print("=" * 70 + "\n")
        return validated

    print("\n" + "=" * 70)
    print("Cached data not found. Running full pipeline...")
    print("=" * 70 + "\n")

    _download_data_if_needed(config)
    df = _parse_and_preprocess_data(config)
    df.to_parquet(processed_file, index=False)

    print(f"✓ Data cached at {processed_file}")
    print(f"✓ Final shape: {df.shape}")
    print(f"✓ No missing values: {df.isna().sum().sum() == 0}\n")

    return df
