"""
Utilities for preparing inference-ready tensors from raw patient data.

All heavy preprocessing logic (imputation, scaling, mask construction) lives
here so notebooks can remain thin orchestrators focused on pedagogy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor

from core.data.physionet_sepsis import FEATURE_COLUMNS


def prepare_patient_batch(
    *,
    patient_df: pd.DataFrame,
    scaler: StandardScaler,
    feature_medians: Mapping[str, float],
    feature_columns: Sequence[str] | None = None,
    time_column: str = "ICULOS",
) -> tuple[Tensor, Tensor]:
    """
    Convert a raw patient dataframe into tensors suitable for inference.

    Parameters
    ----------
    patient_df : pd.DataFrame
        Raw ICU time-series data for a single patient. Must contain all
        required feature columns.
    scaler : StandardScaler
        Scaler fitted on the training data. Used to normalize features.
    feature_medians : Mapping[str, float]
        Mapping of feature name -> median value computed during training.
        Used to fill any columns that remain missing after ffill/bfill.
    feature_columns : Sequence[str] | None, optional
        Ordered list of feature columns. Defaults to the canonical
        :data:`FEATURE_COLUMNS`.
    time_column : str, default="ICULOS"
        Column indicating chronological order for the time series. The
        dataframe is sorted by this column to mirror the training pipeline.

    Returns
    -------
    tuple[Tensor, Tensor]
        A tuple of (features, mask) tensors ready for model inference.

    Raises
    ------
    KeyError
        If required feature columns are missing from ``patient_df``.
    ValueError
        If ``feature_medians`` is missing values for required columns.
    """
    columns = list(feature_columns or FEATURE_COLUMNS)
    missing = set(columns) - set(patient_df.columns)
    if missing:
        raise KeyError(
            "Patient dataframe is missing required feature columns: "
            + ", ".join(sorted(missing))
        )
    if time_column not in patient_df.columns:
        raise KeyError(
            "Patient dataframe is missing the time column required for sorting: "
            f"{time_column}"
        )

    # Ensure chronological ordering matches the deterministic training split.
    sorted_df = patient_df.sort_values(time_column)
    features_df = sorted_df[columns].copy()

    # Mirror the training-time preprocessing pipeline.
    features_df = features_df.ffill().bfill()

    for column in columns:
        if features_df[column].isna().any():
            if column not in feature_medians:
                raise ValueError(
                    f"No median available for column '{column}' while imputing."
                )
            features_df[column] = features_df[column].fillna(feature_medians[column])

    scaled = scaler.transform(features_df)
    features_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    mask = torch.ones(1, features_tensor.shape[1], dtype=torch.bool)
    return features_tensor, mask
