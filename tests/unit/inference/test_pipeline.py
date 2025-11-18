"""Tests for inference preprocessing helpers."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from core.inference import prepare_patient_batch


def _build_scaler(feature_columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    data = pd.DataFrame(
        [
            {
                column: float(idx + position)
                for position, column in enumerate(feature_columns, start=1)
            }
            for idx in range(3)
        ]
    )
    scaler.fit(data)
    assert list(scaler.feature_names_in_) == feature_columns
    return scaler


def test_prepare_patient_batch_imputes_with_medians() -> None:
    feature_columns = ["HR", "O2Sat"]
    patient_df = pd.DataFrame(
        {
            "HR": [float("nan"), 80.0, 82.0],
            "O2Sat": [95.0, float("nan"), 96.0],
            "ICULOS": [0, 1, 2],
        }
    )
    scaler = _build_scaler(feature_columns)
    feature_medians = {"HR": 81.0, "O2Sat": 95.5}

    features, mask = prepare_patient_batch(
        patient_df=patient_df,
        scaler=scaler,
        feature_medians=feature_medians,
        feature_columns=feature_columns,
    )

    assert features.shape == (1, 3, 2)
    assert mask.shape == (1, 3)
    assert mask.all()


def test_prepare_patient_batch_missing_column_raises() -> None:
    feature_columns = ["HR", "O2Sat"]
    patient_df = pd.DataFrame({"HR": [80.0], "ICULOS": [0]})
    scaler = _build_scaler(feature_columns)
    feature_medians = {"HR": 81.0, "O2Sat": 95.5}

    with pytest.raises(KeyError):
        prepare_patient_batch(
            patient_df=patient_df,
            scaler=scaler,
            feature_medians=feature_medians,
            feature_columns=feature_columns,
        )


def test_prepare_patient_batch_sorts_by_time_column() -> None:
    feature_columns = ["HR", "O2Sat"]
    scaler = _build_scaler(feature_columns)
    feature_medians = {"HR": 80.0, "O2Sat": 95.0}
    patient_df = pd.DataFrame(
        {
            "ICULOS": [2, 0, 1],
            "HR": [90.0, 70.0, 80.0],
            "O2Sat": [97.0, 93.0, 95.0],
        }
    )

    features, _ = prepare_patient_batch(
        patient_df=patient_df,
        scaler=scaler,
        feature_medians=feature_medians,
        feature_columns=feature_columns,
        time_column="ICULOS",
    )

    ordered_hr = features.squeeze(0)[:, 0].tolist()
    assert ordered_hr == sorted(ordered_hr)
