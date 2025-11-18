"""
Unit tests for dataset and DataLoader helpers.

These tests ensure that all heavy lifting happens inside the core library,
leaving notebooks with a thin orchestration layer.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from core.config.models import TrainConfig
from core.data.loaders import (
    SepsisDataset,
    build_patient_dataframe_for_subset,
    build_sepsis_dataloaders,
    create_dataloaders,
    sepsis_collate_fn,
)
from core.data.physionet_sepsis import FEATURE_COLUMNS


def _build_mock_dataframe(
    num_patients: int = 4,
    timesteps: int = 5,
) -> pd.DataFrame:
    rows = []
    for patient_id in range(num_patients):
        for t in range(timesteps):
            row = {feature: float(patient_id + t) for feature in FEATURE_COLUMNS}
            row["patient_id"] = patient_id
            row["SepsisLabel"] = int(patient_id % 2)
            row["ICULOS"] = t
            row["Age"] = 30 + patient_id
            row["Gender"] = patient_id % 2
            rows.append(row)
    return pd.DataFrame(rows)


class TestSepsisDataset:
    """Tests covering the cached dataset implementation."""

    def test_dataset_caches_per_patient_sequences(self) -> None:
        df = _build_mock_dataframe(num_patients=3, timesteps=4)
        dataset = SepsisDataset(df)

        assert len(dataset) == 3
        features, mask, label = dataset[0]

        assert features.shape == (4, len(FEATURE_COLUMNS))
        assert features.dtype == torch.float32
        assert mask.shape == (4,)
        assert mask.dtype == torch.bool
        assert label.shape == ()
        assert label.item() in (0.0, 1.0)

        # Subsequent access returns the same cached tensors (identity equality).
        features_again, _, _ = dataset[0]
        assert torch.equal(features, features_again)


class TestSepsisCollateFn:
    """Tests for the padding-aware collate function."""

    def test_collate_fn_pads_to_max_sequence(self) -> None:
        features_short = torch.ones(2, len(FEATURE_COLUMNS))
        mask_short = torch.ones(2, dtype=torch.bool)
        label_short = torch.tensor(0.0)

        features_long = torch.ones(5, len(FEATURE_COLUMNS)) * 2
        mask_long = torch.ones(5, dtype=torch.bool)
        label_long = torch.tensor(1.0)

        batch = [
            (features_short, mask_short, label_short),
            (features_long, mask_long, label_long),
        ]
        padded_features, padded_masks, labels = sepsis_collate_fn(batch)

        assert padded_features.shape == (2, 5, len(FEATURE_COLUMNS))
        assert padded_masks.shape == (2, 5)
        assert padded_masks.dtype == torch.bool
        assert labels.shape == (2,)
        # Padding rows should be zeros.
        assert torch.allclose(
            padded_features[0, 2:], torch.zeros(3, len(FEATURE_COLUMNS))
        )
        assert torch.equal(padded_masks[0, 2:], torch.zeros(3, dtype=torch.bool))


class TestCreateSepsisDataLoaders:
    """Integration-esque tests for the loader helper."""

    def test_builder_returns_deterministic_splits(self) -> None:
        df = _build_mock_dataframe(num_patients=6, timesteps=3)
        config = TrainConfig(
            learning_rate=0.001,
            epochs=1,
            batch_size=2,
            train_val_split=0.5,
            split_seed=123,
        )

        train_loader_a, val_loader_a = create_dataloaders(config, df=df)
        train_loader_b, val_loader_b = create_dataloaders(config, df=df)

        assert isinstance(train_loader_a.dataset, Subset)
        assert isinstance(val_loader_a.dataset, Subset)

        train_subset_a = cast(Subset[SepsisDataset], train_loader_a.dataset)
        train_subset_b = cast(Subset[SepsisDataset], train_loader_b.dataset)
        val_subset_a = cast(Subset[SepsisDataset], val_loader_a.dataset)
        val_subset_b = cast(Subset[SepsisDataset], val_loader_b.dataset)

        assert train_subset_a.indices == train_subset_b.indices
        assert val_subset_a.indices == val_subset_b.indices

    def test_loader_shapes_match_trainer_contract(self) -> None:
        df = _build_mock_dataframe(num_patients=5, timesteps=4)
        config = TrainConfig(
            learning_rate=0.001,
            epochs=1,
            batch_size=2,
            train_val_split=0.6,
            split_seed=7,
        )

        train_loader, val_loader = create_dataloaders(config, df=df)

        batch = next(iter(train_loader))
        features, mask, labels = batch

        assert features.ndim == 3
        assert mask.ndim == 2
        assert labels.ndim == 1
        assert mask.dtype == torch.bool
        assert features.shape[0] == labels.shape[0] == config.batch_size

        val_features, val_mask, val_labels = next(iter(val_loader))
        assert val_features.ndim == 3
        assert val_mask.dtype == torch.bool
        assert val_labels.ndim == 1

    def test_split_counts_match_requested_ratio(self) -> None:
        df = _build_mock_dataframe(num_patients=10, timesteps=3)
        config = TrainConfig(
            learning_rate=0.001,
            epochs=1,
            batch_size=4,
            train_val_split=0.7,
            split_seed=99,
        )

        train_loader, val_loader = create_dataloaders(config, df=df)
        total_patients = df["patient_id"].nunique()
        expected_train = int(total_patients * config.train_val_split)
        expected_val = total_patients - expected_train

        train_subset = cast(Subset[SepsisDataset], train_loader.dataset)
        val_subset = cast(Subset[SepsisDataset], val_loader.dataset)

        assert len(train_subset) == expected_train
        assert len(val_subset) == expected_val

    def test_build_alias_matches_new_api(self) -> None:
        df = _build_mock_dataframe(num_patients=5, timesteps=2)
        config = TrainConfig(
            learning_rate=0.001,
            epochs=1,
            batch_size=2,
            train_val_split=0.6,
            split_seed=1,
        )

        train_alias, val_alias = build_sepsis_dataloaders(config, df=df)
        train_new, val_new = create_dataloaders(config, df=df)

        alias_subset = cast(Subset[SepsisDataset], train_alias.dataset)
        new_subset = cast(Subset[SepsisDataset], train_new.dataset)
        assert alias_subset.indices == new_subset.indices

        alias_val_subset = cast(Subset[SepsisDataset], val_alias.dataset)
        new_val_subset = cast(Subset[SepsisDataset], val_new.dataset)
        assert alias_val_subset.indices == new_val_subset.indices


class TestPatientDataFrameBuilder:
    """Tests for build_patient_dataframe_for_subset helper."""

    def test_dataframe_respects_subset_order(self) -> None:
        df = _build_mock_dataframe(num_patients=5, timesteps=3)
        dataset = SepsisDataset(df)
        subset = Subset(dataset, indices=[4, 1, 3])

        patient_df = build_patient_dataframe_for_subset(
            subset,
            df,
            columns=["Age", "SepsisLabel"],
        )

        assert list(patient_df["patient_id"]) == [4, 1, 3]
        assert list(patient_df["Age"]) == [34, 31, 33]
        assert patient_df.shape[0] == len(subset)

    def test_missing_column_raises_key_error(self) -> None:
        df = _build_mock_dataframe(num_patients=2, timesteps=2)
        dataset = SepsisDataset(df)
        subset = Subset(dataset, indices=[0])

        with pytest.raises(KeyError):
            build_patient_dataframe_for_subset(subset, df, columns=["NonExistent"])

    def test_non_sepsis_subset_rejected(self) -> None:
        tensor_dataset = TensorDataset(torch.arange(5))
        subset = Subset(tensor_dataset, indices=[0, 1])
        df = _build_mock_dataframe(num_patients=2, timesteps=2)

        with pytest.raises(TypeError):
            build_patient_dataframe_for_subset(subset, df, columns=["Age"])

    def test_handles_string_patient_ids(self) -> None:
        """Regression test for string patient_id type conversion.

        This test verifies that build_patient_dataframe_for_subset correctly
        handles DataFrames with string patient_ids (as returned by PhysioNet
        data loading), even though SepsisDataset internally converts them to
        integers. This was the root cause of a KeyError in notebook 6.
        """
        # Create a DataFrame with string patient IDs (mimics PhysioNet data)
        df = _build_mock_dataframe(num_patients=5, timesteps=3)
        df["patient_id"] = df["patient_id"].astype(str)

        # SepsisDataset will convert these string IDs to integers internally
        dataset = SepsisDataset(df)
        subset = Subset(dataset, indices=[2, 0, 4])

        # This should work despite the type mismatch - the function
        # should convert patient_id to int before lookup
        patient_df = build_patient_dataframe_for_subset(
            subset,
            df,
            columns=["Age", "Gender"],
        )

        # Verify correct ordering and values
        assert list(patient_df["patient_id"]) == [2, 0, 4]
        assert list(patient_df["Age"]) == [32, 30, 34]
        assert patient_df.shape[0] == len(subset)
