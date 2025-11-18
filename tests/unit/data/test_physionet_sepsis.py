"""
Unit tests for the PhysioNet 2019 Sepsis data pipeline.

All tests mock out heavyweight behaviors (network, filesystem sync) to ensure
the suite stays deterministic and fast.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from botocore.client import BaseClient
from pydantic import HttpUrl, ValidationError

from core.config import PhysioNetConfig
from core.data.physionet_sepsis import (
    FEATURE_COLUMNS,
    _download_data_if_needed,
    _download_s3_object,
    _download_via_credentials,
    _download_via_public_s3,
    _load_config,
    _parse_and_preprocess_data,
    _prompt_for_credentials,
    get_sepsis_data,
    load_feature_preprocessor,
)


def _make_config(tmp_path: Path) -> PhysioNetConfig:
    config = PhysioNetConfig(
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        processed_file=tmp_path / "processed" / "sepsis.parquet",
        scaler_file=tmp_path / "processed" / "scaler.json",
        download_url=cast(HttpUrl, "https://example.com/data.zip"),
    )
    config.ensure_directories()
    return config


def _build_dataframe(rows: int = 2) -> pd.DataFrame:
    frame_data: dict[str, list[float]] = {
        column: [float(idx + 1 + i) for i in range(rows)]
        for idx, column in enumerate(FEATURE_COLUMNS)
    }
    frame_data.update(
        {
            "Age": [float(60 + i) for i in range(rows)],
            "Gender": [float(i % 2) for i in range(rows)],
            "Unit1": [0.0 for _ in range(rows)],
            "Unit2": [1.0 for _ in range(rows)],
            "HospAdmTime": [float(-10.0 - i) for i in range(rows)],
            "ICULOS": [float(i + 1) for i in range(rows)],
            "SepsisLabel": [float(i % 2) for i in range(rows)],
        }
    )
    df = pd.DataFrame(frame_data)
    df["patient_id"] = [f"{i+1}" for i in range(rows)]
    return df


class TestConfigLoading:
    """Test configuration file loading logic."""

    def test_load_config_success(self, tmp_path: Path) -> None:
        config_path = tmp_path / "data.yaml"
        config_path.write_text(
            """
physionet_2019:
  raw_dir: "data/raw/physionet_2019"
  processed_dir: "data/processed"
  processed_file: "data/processed/sepsis_preprocessed.parquet"
  scaler_file: "data/processed/feature_scaler.json"
  download_url: "https://example.com/data.zip"
"""
        )

        config = _load_config(config_path)

        assert isinstance(config, PhysioNetConfig)
        assert config.raw_dir == Path("data/raw/physionet_2019")

    def test_load_config_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _load_config(tmp_path / "missing.yaml")

    def test_load_config_missing_section(self, tmp_path: Path) -> None:
        config_path = tmp_path / "data.yaml"
        config_path.write_text("other_section:\n  key: value\n")

        with pytest.raises(ValidationError):
            _load_config(config_path)


class TestPromptForCredentials:
    """Credential helper tests."""

    def test_prompt_uses_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PHYSIONET_USERNAME", "env_user")
        monkeypatch.setenv("PHYSIONET_PASSWORD", "env_pass")

        username, password = _prompt_for_credentials()
        assert username == "env_user"
        assert password == "env_pass"

    @patch("builtins.input", return_value="")
    @patch("core.data.physionet_sepsis.getpass.getpass", return_value="")
    def test_prompt_rejects_empty(
        self, mock_getpass: MagicMock, mock_input: MagicMock
    ) -> None:
        with pytest.raises(ValueError):
            _prompt_for_credentials()


class TestDataDownloader:
    """Tests for orchestrating the download logic."""

    @patch("core.data.physionet_sepsis._download_via_public_s3")
    @patch("core.data.physionet_sepsis._download_via_credentials")
    def test_download_skips_when_data_exists(
        self,
        mock_credentials: MagicMock,
        mock_public: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        training_file = config.raw_dir / "training" / "p0001.psv"
        training_file.parent.mkdir(parents=True, exist_ok=True)
        training_file.write_text("fake data")

        _download_data_if_needed(config)

        mock_public.assert_not_called()
        mock_credentials.assert_not_called()

    @patch("core.data.physionet_sepsis._download_via_public_s3", return_value=True)
    @patch("core.data.physionet_sepsis._download_via_credentials")
    def test_download_uses_public_s3(
        self,
        mock_credentials: MagicMock,
        mock_public: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)

        _download_data_if_needed(config)

        mock_public.assert_called_once_with(config, client=None)
        mock_credentials.assert_not_called()

    @patch("core.data.physionet_sepsis._download_via_public_s3", return_value=False)
    @patch("core.data.physionet_sepsis._download_via_credentials")
    def test_download_falls_back_to_credentials(
        self,
        mock_credentials: MagicMock,
        mock_public: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)

        _download_data_if_needed(config)

        mock_public.assert_called_once()
        mock_credentials.assert_called_once_with(config)


class TestCredentialDownload:
    """Tests for the credentialed download fallback."""

    @patch("core.data.physionet_sepsis.zipfile.ZipFile")
    @patch("core.data.physionet_sepsis.requests.get")
    @patch(
        "core.data.physionet_sepsis._prompt_for_credentials",
        return_value=("user", "pass"),
    )
    def test_download_via_credentials_success(
        self,
        mock_credentials: MagicMock,
        mock_requests_get: MagicMock,
        mock_zipfile: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)

        response = MagicMock()
        response.status_code = 200
        response.headers = {"content-length": "10"}
        response.iter_content.return_value = [b"12345"]
        mock_requests_get.return_value = response

        def _extract_all(path: Path) -> None:
            (path / "training" / "p0001.psv").parent.mkdir(parents=True, exist_ok=True)
            (path / "training" / "p0001.psv").write_text("content")

        mock_zipfile.return_value.__enter__.return_value.extractall.side_effect = (
            _extract_all
        )

        _download_via_credentials(config)

        assert mock_requests_get.call_count == 1
        _, kwargs = mock_requests_get.call_args
        assert kwargs["headers"] is None
        assert (config.raw_dir / "training" / "p0001.psv").exists()

    @patch("core.data.physionet_sepsis.requests.get")
    @patch(
        "core.data.physionet_sepsis._prompt_for_credentials",
        return_value=("user", "pass"),
    )
    def test_download_via_credentials_auth_failure(
        self,
        mock_credentials: MagicMock,
        mock_requests_get: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)

        response = MagicMock()
        response.status_code = 401
        response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=response
        )
        mock_requests_get.return_value = response

        with pytest.raises(requests.exceptions.HTTPError):
            _download_via_credentials(config)

    @patch("core.data.physionet_sepsis.zipfile.ZipFile")
    @patch("core.data.physionet_sepsis.requests.get")
    @patch(
        "core.data.physionet_sepsis._prompt_for_credentials",
        return_value=("user", "pass"),
    )
    def test_download_via_credentials_resumes_partial(
        self,
        mock_credentials: MagicMock,
        mock_requests_get: MagicMock,
        mock_zipfile: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        zip_path = config.raw_dir / "training.zip"
        zip_path.write_bytes(b"12345")

        response = MagicMock()
        response.status_code = 206
        response.headers = {
            "content-length": "5",
            "content-range": "bytes 5-9/10",
        }
        response.iter_content.return_value = [b"67890"]
        mock_requests_get.return_value = response

        def _extract(path: Path) -> None:
            training_dir = path / "training"
            training_dir.mkdir(parents=True, exist_ok=True)
            (training_dir / "p0001.psv").write_text("content")

        mock_zipfile.return_value.__enter__.return_value.extractall.side_effect = (
            _extract
        )

        _download_via_credentials(config)

        _, kwargs = mock_requests_get.call_args
        assert kwargs["headers"] == {"Range": "bytes=5-"}

    @patch("core.data.physionet_sepsis.zipfile.ZipFile")
    @patch("core.data.physionet_sepsis.requests.get")
    @patch(
        "core.data.physionet_sepsis._prompt_for_credentials",
        return_value=("user", "pass"),
    )
    def test_download_via_credentials_restarts_when_range_unsupported(
        self,
        mock_credentials: MagicMock,
        mock_requests_get: MagicMock,
        mock_zipfile: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        zip_path = config.raw_dir / "training.zip"
        zip_path.write_bytes(b"123")

        first_response = MagicMock()
        first_response.status_code = 200
        first_response.headers = {"content-length": "10"}
        first_response.iter_content.return_value = [b"data"]

        second_response = MagicMock()
        second_response.status_code = 200
        second_response.headers = {"content-length": "10"}
        second_response.iter_content.return_value = [b"data2"]

        mock_requests_get.side_effect = [first_response, second_response]

        def _extract(path: Path) -> None:
            training_dir = path / "training"
            training_dir.mkdir(parents=True, exist_ok=True)
            (training_dir / "p0001.psv").write_text("content")

        mock_zipfile.return_value.__enter__.return_value.extractall.side_effect = (
            _extract
        )

        _download_via_credentials(config)

        assert mock_requests_get.call_count == 2
        first_call_kwargs = mock_requests_get.call_args_list[0][1]
        second_call_kwargs = mock_requests_get.call_args_list[1][1]
        assert first_call_kwargs["headers"] == {"Range": "bytes=3-"}
        assert second_call_kwargs["headers"] is None


class DummyBody:
    """Minimal streaming body for S3 download tests."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def read(self, _: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def close(self) -> None:  # pragma: no cover - simple stub
        self._chunks.clear()


class DummyProgress:
    """Simplified tqdm stand-in for deterministic tests."""

    def __enter__(self) -> DummyProgress:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def update(self, _: int) -> None:
        return None


class TestS3ObjectDownload:
    """Tests for the resumable S3 download helper."""

    @patch("core.data.physionet_sepsis.tqdm", return_value=DummyProgress())
    def test_s3_download_resumes_from_partial(
        self, mock_tqdm: MagicMock, tmp_path: Path
    ) -> None:
        destination = tmp_path / "training" / "file.psv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(".psv.part")
        partial.write_bytes(b"12345")

        client = MagicMock()
        client.get_object.return_value = {"Body": DummyBody([b"67890"])}

        _download_s3_object(client, "bucket", "key", 10, destination)

        client.get_object.assert_called_once_with(
            Bucket="bucket", Key="key", Range="bytes=5-"
        )
        assert destination.read_bytes() == b"1234567890"

    @patch("core.data.physionet_sepsis.tqdm", return_value=DummyProgress())
    def test_s3_download_skips_when_complete(
        self, mock_tqdm: MagicMock, tmp_path: Path
    ) -> None:
        destination = tmp_path / "training" / "file.psv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"content")

        client = MagicMock()

        _download_s3_object(client, "bucket", "key", len("content"), destination)

        client.get_object.assert_not_called()


class DummyPaginator:
    """Paginator stub that returns a static set of keys."""

    def __init__(self, contents: list[dict[str, str | int]]) -> None:
        self._contents = contents

    def paginate(self, **kwargs: object) -> list[dict[str, list[dict[str, str | int]]]]:
        return [{"Contents": self._contents}]


class DummyS3Client:
    """S3 client stub exposing a paginator interface."""

    def __init__(self, contents: list[dict[str, str | int]]) -> None:
        self._contents = contents

    def get_paginator(self, name: str) -> DummyPaginator:
        assert name == "list_objects_v2"
        return DummyPaginator(self._contents)


class TestPublicS3Download:
    """Ensure the S3 orchestration schedules downloads efficiently."""

    @patch("core.data.physionet_sepsis._download_s3_object")
    def test_parallel_downloads_submit_missing_files(
        self, mock_download: MagicMock, tmp_path: Path
    ) -> None:
        config = _make_config(tmp_path)
        key_prefix = config.s3_prefix.rstrip("/") + "/"
        contents: list[dict[str, str | int]] = [
            {"Key": f"{key_prefix}training_setA/p0001.psv", "Size": 10},
            {"Key": f"{key_prefix}training_setA/p0002.psv", "Size": 12},
        ]
        client = DummyS3Client(contents)

        result = _download_via_public_s3(
            config,
            client=cast(BaseClient, client),
            max_workers=2,
        )

        assert result is True
        assert mock_download.call_count == 2


class TestImputationLogic:
    """Test missing value imputation logic."""

    def test_ffill_bfill_imputation(self) -> None:
        data = {
            "patient_id": ["p1", "p1", "p1", "p2", "p2"],
            "HR": [None, 80.0, None, 70.0, None],
            "Temp": [36.5, None, 37.0, None, 36.8],
        }
        df = pd.DataFrame(data)

        imputed = (
            df.groupby("patient_id", group_keys=False)[["HR", "Temp"]]
            .apply(lambda group: group.ffill().bfill())
            .reset_index(drop=True)
        )

        assert imputed.loc[0, "HR"] == 80.0
        assert imputed.loc[2, "Temp"] == 37.0

    def test_median_imputation_fallback(self) -> None:
        data = {
            "patient_id": ["p1", "p2", "p3"],
            "HR": [None, 90.0, 70.0],
            "Temp": [36.5, 37.0, None],
        }
        df = pd.DataFrame(data)

        imputed = (
            df.groupby("patient_id", group_keys=False)[["HR", "Temp"]]
            .apply(lambda group: group.ffill().bfill())
            .reset_index(drop=True)
        )

        assert pd.isna(imputed.loc[0, "HR"])

        median_hr = imputed["HR"].median()
        imputed["HR"] = imputed["HR"].fillna(median_hr)

        assert imputed.loc[0, "HR"] == pytest.approx(80.0)


class TestPreprocessingPipeline:
    """Test the full preprocessing pipeline."""

    def test_parse_and_preprocess_data(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        training_dir = config.raw_dir / "training" / "training_setA"
        training_dir.mkdir(parents=True, exist_ok=True)

        content = (
            "HR|O2Sat|Temp|SBP|MAP|DBP|Resp|EtCO2|BaseExcess|HCO3|FiO2|pH|"
            "PaCO2|SaO2|AST|BUN|Alkalinephos|Calcium|Chloride|Creatinine|"
            "Bilirubin_direct|Glucose|Lactate|Magnesium|Phosphate|Potassium|"
            "Bilirubin_total|TroponinI|Hct|Hgb|PTT|WBC|Fibrinogen|Platelets|"
            "Age|Gender|Unit1|Unit2|HospAdmTime|ICULOS|SepsisLabel\n"
            "80|95|36.5|120|80|70|16|35|0|24|0.21|7.4|40|95|20|15|60|8.5|105|"
            "1.0|0.2|100|1.5|2.0|3.5|4.0|0.5|0.1|40|13|30|10|300|200|65|0|1|0|-50|1|0\n"
            "85|96|36.8|125|82|72|18|36|1|25|0.21|7.4|40|96|21|16|61|8.6|106|"
            "1.1|0.2|105|1.6|2.1|3.6|4.1|0.6|0.1|41|14|31|11|310|210|65|0|1|0|-49|2|0\n"
        )
        (training_dir / "p0001.psv").write_text(content)

        df = _parse_and_preprocess_data(config)

        assert df.isna().sum().sum() == 0
        assert df["patient_id"].nunique() == 1
        scaler_path = config.scaler_file
        assert scaler_path.exists()
        scaler, medians = load_feature_preprocessor(scaler_path)
        assert scaler.n_features_in_ == len(FEATURE_COLUMNS)
        assert set(medians) == set(FEATURE_COLUMNS)


class TestGetSepsisData:
    """Test the main public API function."""

    def test_get_sepsis_data_from_cache(self, tmp_path: Path) -> None:
        processed_file = tmp_path / "processed" / "sepsis.parquet"
        processed_file.parent.mkdir(parents=True, exist_ok=True)
        df = _build_dataframe()
        df.to_parquet(processed_file, index=False)

        config_path = tmp_path / "data.yaml"
        config_path.write_text(
            f"""
physionet_2019:
  raw_dir: "{tmp_path / "raw"}"
  processed_dir: "{tmp_path / "processed"}"
  processed_file: "{processed_file}"
  scaler_file: "{tmp_path / "processed" / "scaler.json"}"
  download_url: "https://example.com/data.zip"
"""
        )

        result = get_sepsis_data(config_path=config_path)
        assert result.equals(df)

    @patch("core.data.physionet_sepsis._parse_and_preprocess_data")
    @patch("core.data.physionet_sepsis._download_data_if_needed")
    def test_get_sepsis_data_pipeline(
        self,
        mock_download: MagicMock,
        mock_preprocess: MagicMock,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "data.yaml"
        processed_file = tmp_path / "processed" / "sepsis.parquet"
        config_path.write_text(
            f"""
physionet_2019:
  raw_dir: "{tmp_path / "raw"}"
  processed_dir: "{tmp_path / "processed"}"
  processed_file: "{processed_file}"
  scaler_file: "{tmp_path / "processed" / "scaler.json"}"
  download_url: "https://example.com/data.zip"
"""
        )

        df = _build_dataframe()
        mock_preprocess.return_value = df

        result = get_sepsis_data(config_path=config_path)

        mock_download.assert_called_once()
        mock_preprocess.assert_called_once()
        assert result.equals(df)
        assert processed_file.exists()
