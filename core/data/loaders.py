"""
Dataset and DataLoader utilities for the PhysioNet sepsis data.

This module centralizes all torch-facing data preparation so that notebooks
remain thin orchestrators focused on pedagogy. It caches per-patient tensors
up-front, provides a deterministic train/validation split, and exposes a
collate function that pads sequences to the maximum length in each batch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from core.config import TrainConfig
from core.data.physionet_sepsis import FEATURE_COLUMNS, get_sepsis_data

logger = logging.getLogger(__name__)

PatientSample: TypeAlias = tuple[Tensor, Tensor, Tensor]


class SepsisDataset(Dataset[PatientSample]):
    """
    Cached, patient-level dataset for the PhysioNet sepsis time series.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed sepsis dataframe returned by ``get_sepsis_data``.
    feature_columns : Sequence[str] | None
        Ordered list of feature columns to extract. Defaults to the canonical
        ``FEATURE_COLUMNS`` defined in ``core.data.physionet_sepsis``.

    Notes
    -----
    All heavy preprocessing (sorting, tensor conversion) happens once during
    initialization to keep per-epoch iteration fast inside notebooks and tests.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str] | None = None,
    ) -> None:
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self._sequences: list[PatientSample] = []
        self._patient_ids: list[int] = []

        grouped = data.groupby("patient_id", sort=False)
        for patient_id, patient_df in grouped:
            sorted_df = patient_df.sort_values("ICULOS")
            features = torch.tensor(
                sorted_df[self.feature_columns].to_numpy(copy=True),
                dtype=torch.float32,
            )
            mask = torch.ones(len(sorted_df), dtype=torch.bool)
            label_value = float(sorted_df["SepsisLabel"].iloc[-1])
            label = torch.tensor(label_value, dtype=torch.float32)
            self._sequences.append((features, mask, label))
            # PhysioNet patient identifiers are integers.
            self._patient_ids.append(int(patient_id))

        logger.info(
            "Constructed SepsisDataset with %d patients and %d feature columns",
            len(self._sequences),
            len(self.feature_columns),
        )

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, index: int) -> PatientSample:
        return self._sequences[index]

    @property
    def patient_ids(self) -> list[int]:
        """Return the cached ordering of patient identifiers."""
        return list(self._patient_ids)


def sepsis_collate_fn(batch: Sequence[PatientSample]) -> PatientSample:
    """
    Collate function that pads variable-length sequences in a batch.

    Parameters
    ----------
    batch : Sequence[PatientSample]
        Batch of (features, mask, label) tuples.

    Returns
    -------
    PatientSample
        Tuple containing padded feature tensor [batch, seq, feat], padded mask
        [batch, seq], and label tensor [batch].
    """
    features_list, masks_list, labels_list = zip(*batch, strict=True)
    max_seq_len = max(features.size(0) for features in features_list)
    feature_dim = features_list[0].size(1)

    padded_features = []
    padded_masks = []
    for features, mask in zip(features_list, masks_list, strict=True):
        pad_len = max_seq_len - features.size(0)
        if pad_len > 0:
            feature_padding = torch.zeros(pad_len, feature_dim, dtype=features.dtype)
            mask_padding = torch.zeros(pad_len, dtype=torch.bool)
            features = torch.cat((features, feature_padding), dim=0)
            mask = torch.cat((mask, mask_padding), dim=0)
        padded_features.append(features)
        padded_masks.append(mask)

    labels = torch.stack(labels_list).to(torch.float32)
    return torch.stack(padded_features), torch.stack(padded_masks), labels


def _split_dataset(
    dataset: SepsisDataset,
    train_val_split: float,
    *,
    seed: int | None,
) -> tuple[Subset[SepsisDataset], Subset[SepsisDataset]]:
    """Deterministically split the dataset into train/validation subsets."""
    if not 0.0 < train_val_split < 1.0:
        raise ValueError("train_val_split must be in the interval (0, 1).")

    total = len(dataset)
    if total < 2:
        raise ValueError(
            "SepsisDataset must contain at least two patient sequences "
            "to perform a train/validation split."
        )

    ideal_train = total * train_val_split
    train_size = int(ideal_train)
    val_size = total - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(
            "train_val_split produced an empty split. Adjust the ratio or "
            "use a dataset with more patient sequences."
        )

    logger.info(
        "Splitting %d patients -> train=%d (%.1f%%) | val=%d (%.1f%%) "
        "(ideal train=%0.2f)",
        total,
        train_size,
        (train_size / total) * 100,
        val_size,
        (val_size / total) * 100,
        ideal_train,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_subset, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )
    return (
        cast(Subset[SepsisDataset], train_subset),
        cast(Subset[SepsisDataset], val_subset),
    )


def split_sepsis_dataset(
    dataset: SepsisDataset,
    train_val_split: float,
    *,
    seed: int | None,
) -> tuple[Subset[SepsisDataset], Subset[SepsisDataset]]:
    """
    Public helper that exposes the deterministic dataset split logic.

    This keeps notebooks and tests DRY by reusing the same validation contract
    implemented by :func:`create_dataloaders`.
    """
    return _split_dataset(dataset, train_val_split, seed=seed)


def create_dataloaders(
    train_config: TrainConfig,
    *,
    df: pd.DataFrame | None = None,
) -> tuple[DataLoader[PatientSample], DataLoader[PatientSample]]:
    """
    Construct deterministic train/validation DataLoaders for sepsis modeling.

    Parameters
    ----------
    train_config : TrainConfig
        Validated training configuration containing batch size, device, and
        split settings (train_val_split, split_seed).
    df : pd.DataFrame | None
        Optional pre-loaded dataframe. If omitted, ``get_sepsis_data`` is used.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Train and validation DataLoaders ready for consumption by Trainer.
    """
    data = df if df is not None else get_sepsis_data()
    dataset = SepsisDataset(data)
    train_subset, val_subset = _split_dataset(
        dataset,
        train_val_split=train_config.train_val_split,
        seed=train_config.split_seed,
    )

    logger.info(
        "Building DataLoaders with batch_size=%d | train=%d | val=%d | seed=%s",
        train_config.batch_size,
        len(train_subset),
        len(val_subset),
        str(train_config.split_seed),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=train_config.batch_size,
        shuffle=True,
        collate_fn=sepsis_collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=train_config.batch_size,
        shuffle=False,
        collate_fn=sepsis_collate_fn,
    )
    return (
        cast(DataLoader[PatientSample], train_loader),
        cast(DataLoader[PatientSample], val_loader),
    )


# Backwards compatibility: historical name used by earlier tickets/notebooks.
build_sepsis_dataloaders = create_dataloaders


def build_patient_dataframe_for_subset(
    subset: Subset[Any],
    full_df: pd.DataFrame,
    *,
    columns: Sequence[str],
    patient_id_column: str = "patient_id",
    time_column: str = "ICULOS",
) -> pd.DataFrame:
    """
    Create a patient-level DataFrame aligned with a dataset subset.

    Parameters
    ----------
    subset : Subset[Any]
        Dataset subset whose ordering must line up with prediction arrays. The
        underlying dataset must be :class:`SepsisDataset`.
    full_df : pd.DataFrame
        Full PhysioNet dataframe containing demographic/static columns.
    columns : Sequence[str]
        Columns to extract for each patient (e.g., ['Age', 'Gender']).
    patient_id_column : str, default="patient_id"
        Name of the identifier column in ``full_df``.
    time_column : str, default="ICULOS"
        Name of the time-ordering column used to select the latest row.

    Returns
    -------
    pd.DataFrame
        DataFrame with exactly ``len(subset)`` rows, ordered identically to
        the subset indices. Includes ``patient_id`` plus the requested columns.

    Raises
    ------
    TypeError
        If ``subset.dataset`` is not a :class:`SepsisDataset`.
    ValueError
        If ``columns`` is empty or subset has no indices.
    KeyError
        If required columns are missing from ``full_df``.
    """
    if not isinstance(subset.dataset, SepsisDataset):
        raise TypeError(
            "build_patient_dataframe_for_subset requires a Subset[SepsisDataset]."
        )

    if not subset.indices:
        raise ValueError("subset must contain at least one patient index.")

    if not columns:
        raise ValueError("columns must contain at least one column name.")

    required_columns = {patient_id_column, time_column, *columns}
    missing = required_columns - set(full_df.columns)
    if missing:
        raise KeyError(
            "full_df is missing required columns: " f"{', '.join(sorted(missing))}."
        )

    # Aggregate to one row per patient using the latest timestep
    sorted_df = full_df.sort_values([patient_id_column, time_column])
    patient_level = (
        sorted_df.groupby(patient_id_column, sort=False).last().reset_index()
    )

    selected_columns = [patient_id_column, *columns]
    available = set(patient_level.columns)
    missing_selected = set(selected_columns) - available
    if missing_selected:
        raise KeyError(
            "Patient-level dataframe is missing columns: "
            f"{', '.join(sorted(missing_selected))}."
        )

    patient_level = patient_level[selected_columns]
    # Convert patient_id to int to match SepsisDataset.patient_ids type
    # (SepsisDataset stores patient_ids as integers, see line 68)
    patient_level[patient_id_column] = patient_level[patient_id_column].astype(int)
    patient_level = patient_level.set_index(patient_id_column)

    dataset = subset.dataset
    ordered_patient_ids = [dataset.patient_ids[idx] for idx in subset.indices]

    try:
        ordered_rows = patient_level.loc[ordered_patient_ids]
    except KeyError as exc:  # pragma: no cover - safety net
        raise KeyError(
            "One or more patient_ids from the subset were not found "
            "in the source dataframe."
        ) from exc

    return ordered_rows.reset_index().rename(columns={"index": patient_id_column})
