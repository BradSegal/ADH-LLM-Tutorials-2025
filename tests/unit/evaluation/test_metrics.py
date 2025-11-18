"""
Unit tests for the evaluation.metrics module.

These tests validate the prediction generation and metric calculation functions
using mock models and synthetic data.
"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from core.evaluation.metrics import (
    PredictionResults,
    calculate_calibration_metrics,
    calculate_classification_metrics,
    calculate_subgroup_metrics,
    get_predictions,
)
from core.models.base import BaseSequenceModel


class MockModel(BaseSequenceModel):
    """Mock model for testing that returns predetermined logits."""

    def __init__(self, mock_logits: torch.Tensor) -> None:
        super().__init__()
        self.mock_logits = mock_logits
        self.call_count = 0

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return predetermined logits for testing."""
        batch_size = x.shape[0]
        # Return the appropriate slice of mock logits
        start_idx = self.call_count
        end_idx = start_idx + batch_size
        self.call_count += batch_size
        return self.mock_logits[start_idx:end_idx].unsqueeze(1)


@pytest.fixture
def sample_data():
    """Create sample labels and predictions for testing metrics."""
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    predictions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9], dtype=np.float32)
    return labels, predictions


@pytest.fixture
def mock_dataloader():
    """Create a mock DataLoader for testing get_predictions."""
    # Create synthetic data: 12 samples with 5 timesteps and 10 features
    features = torch.randn(12, 5, 10)
    masks = torch.ones(12, 5, dtype=torch.bool)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype=torch.float32)

    dataset = TensorDataset(features, masks, labels)
    loader = DataLoader(dataset, batch_size=4)

    return loader, labels


class TestCalculateClassificationMetrics:
    """Tests for calculate_classification_metrics function."""

    def test_valid_metrics_calculation(self, sample_data):
        """Test that metrics are calculated correctly for valid inputs."""
        labels, predictions = sample_data
        metrics = calculate_classification_metrics(labels, predictions)

        # Check that both metrics are present
        assert "auroc" in metrics
        assert "auprc" in metrics

        # Check that values are in valid range [0, 1]
        assert 0.0 <= metrics["auroc"] <= 1.0
        assert 0.0 <= metrics["auprc"] <= 1.0

        # For this specific data, AUROC should be 1.0 (perfect separation)
        assert metrics["auroc"] == 1.0

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        labels = np.array([0, 0, 1, 1], dtype=np.float32)
        predictions = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        metrics = calculate_classification_metrics(labels, predictions)

        assert metrics["auroc"] == 1.0

    def test_random_predictions(self):
        """Test metrics with random-like predictions."""
        np.random.seed(42)
        labels = np.random.randint(0, 2, 100).astype(np.float32)
        predictions = np.random.rand(100).astype(np.float32)

        metrics = calculate_classification_metrics(labels, predictions)

        # Random predictions should give AUROC around 0.5
        # We use a loose bound since it's random
        assert 0.3 <= metrics["auroc"] <= 0.7

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        labels = np.array([0, 1, 0], dtype=np.float32)
        predictions = np.array([0.1, 0.9], dtype=np.float32)

        with pytest.raises(ValueError, match="same length"):
            calculate_classification_metrics(labels, predictions)

    def test_empty_arrays_raise_error(self):
        """Test that empty arrays raise ValueError."""
        labels = np.array([], dtype=np.float32)
        predictions = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="must not be empty"):
            calculate_classification_metrics(labels, predictions)

    def test_single_class_raises_error(self):
        """Test that having only one class raises ValueError."""
        labels = np.array([1, 1, 1, 1], dtype=np.float32)
        predictions = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)

        with pytest.raises(ValueError, match="both classes"):
            calculate_classification_metrics(labels, predictions)


class TestGetPredictions:
    """Tests for get_predictions function."""

    def test_get_predictions_output_shapes(self, mock_dataloader):
        """Test that get_predictions returns correct output shapes."""
        loader, true_labels = mock_dataloader

        # Create mock logits (12 samples)
        mock_logits = torch.tensor(
            [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, -1.0, 1.0, -0.5, 0.5],
            dtype=torch.float32,
        )
        mock_model = MockModel(mock_logits)

        labels, predictions, logits = get_predictions(mock_model, loader, device="cpu")

        # Check shapes
        assert labels.shape == (12,)
        assert predictions.shape == (12,)
        assert logits.shape == (12,)

        # Check that labels match the true labels
        np.testing.assert_array_equal(labels, true_labels.numpy())

        # Check that predictions are in [0, 1] (sigmoid output)
        assert np.all(predictions >= 0.0)
        assert np.all(predictions <= 1.0)

    def test_get_predictions_sigmoid_applied(self):
        """Test that sigmoid is correctly applied to logits."""
        # Create simple data
        features = torch.randn(4, 3, 5)
        masks = torch.ones(4, 3, dtype=torch.bool)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.float32)

        dataset = TensorDataset(features, masks, labels)
        loader = DataLoader(dataset, batch_size=2)

        # Known logits
        mock_logits = torch.tensor([0.0, 1.0, -1.0, 2.0], dtype=torch.float32)
        mock_model = MockModel(mock_logits)

        _, predictions, logits = get_predictions(
            mock_model, loader, device="cpu"  # type: ignore[arg-type]
        )

        # Verify sigmoid relationship: pred = 1 / (1 + exp(-logit))
        expected_preds = 1.0 / (1.0 + np.exp(-logits))
        np.testing.assert_array_almost_equal(predictions, expected_preds, decimal=5)

    def test_get_predictions_invalid_model(self, mock_dataloader):
        """Test that non-BaseSequenceModel raises TypeError."""
        loader, _ = mock_dataloader

        # Create a mock that is not a BaseSequenceModel
        invalid_model = MagicMock()

        with pytest.raises(TypeError, match="BaseSequenceModel"):
            get_predictions(invalid_model, loader)

    def test_get_predictions_device_handling(self, mock_dataloader):
        """Test that device parameter works correctly."""
        loader, _ = mock_dataloader

        mock_logits = torch.randn(12)
        mock_model = MockModel(mock_logits)

        # Test with device as string
        labels, preds, logits = get_predictions(mock_model, loader, device="cpu")
        assert labels.shape == (12,)

        # Test with device as torch.device
        labels, preds, logits = get_predictions(
            mock_model, loader, device=torch.device("cpu")
        )
        assert labels.shape == (12,)

    def test_get_predictions_tensor_return(self, mock_dataloader):
        """get_predictions can optionally return tensors via dataclass."""
        loader, _ = mock_dataloader
        mock_logits = torch.randn(12)
        mock_model = MockModel(mock_logits)

        result = get_predictions(mock_model, loader, device="cpu", return_tensors=True)

        assert isinstance(result, PredictionResults)
        assert isinstance(result.labels, torch.Tensor)
        assert result.labels.shape == torch.Size([12])

    def test_get_predictions_model_eval_mode(self, mock_dataloader):
        """Test that model is set to eval mode."""
        loader, _ = mock_dataloader

        mock_logits = torch.randn(12)
        mock_model = MockModel(mock_logits)

        # Set to training mode initially
        mock_model.train()
        assert mock_model.training is True

        # Call get_predictions
        get_predictions(mock_model, loader, device="cpu")

        # Model should be in eval mode
        assert mock_model.training is False


class TestCalculateCalibrationMetrics:
    """Tests for calculate_calibration_metrics function."""

    def test_valid_calibration_calculation(self):
        """Test that calibration metrics are calculated correctly."""
        # Create well-calibrated predictions
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
        predictions = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9], dtype=np.float32
        )

        metrics = calculate_calibration_metrics(labels, predictions, n_bins=5)

        # Check that all required keys are present
        assert "brier_score" in metrics
        assert "prob_true" in metrics
        assert "prob_pred" in metrics

        # Check that brier_score is in valid range [0, 1]
        assert 0.0 <= metrics["brier_score"] <= 1.0

        # Check that calibration curve arrays are numpy arrays
        assert isinstance(metrics["prob_true"], np.ndarray)
        assert isinstance(metrics["prob_pred"], np.ndarray)

    def test_perfect_calibration(self):
        """Test with perfectly calibrated predictions."""
        # Create a large dataset where predictions match true frequencies
        np.random.seed(42)
        n_samples = 1000

        # Create predictions uniformly distributed in [0, 1]
        predictions = np.random.rand(n_samples).astype(np.float32)

        # Generate labels based on predictions (perfectly calibrated)
        labels = (np.random.rand(n_samples) < predictions).astype(np.float32)

        metrics = calculate_calibration_metrics(labels, predictions, n_bins=10)

        # For perfectly calibrated model, Brier score should be relatively low
        # We use a loose bound because of randomness
        assert metrics["brier_score"] < 0.3

    def test_poor_calibration(self):
        """Test with poorly calibrated predictions."""
        # Model always predicts 0.5 regardless of true label
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
        predictions = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32
        )

        metrics = calculate_calibration_metrics(labels, predictions, n_bins=5)

        # Check that brier_score is calculated (should be 0.25 for this case)
        assert abs(metrics["brier_score"] - 0.25) < 0.01

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        labels = np.array([0, 1, 0], dtype=np.float32)
        predictions = np.array([0.1, 0.9], dtype=np.float32)

        with pytest.raises(ValueError, match="same length"):
            calculate_calibration_metrics(labels, predictions)

    def test_empty_arrays_raise_error(self):
        """Test that empty arrays raise ValueError."""
        labels = np.array([], dtype=np.float32)
        predictions = np.array([], dtype=np.float32)

        with pytest.raises(ValueError, match="must not be empty"):
            calculate_calibration_metrics(labels, predictions)

    def test_invalid_n_bins_raises_error(self):
        """Test that n_bins < 2 raises ValueError."""
        labels = np.array([0, 1], dtype=np.float32)
        predictions = np.array([0.1, 0.9], dtype=np.float32)

        with pytest.raises(ValueError, match="n_bins must be at least 2"):
            calculate_calibration_metrics(labels, predictions, n_bins=1)

    def test_calibrated_classifier_improves_brier_score(self):
        """Calibrated classifier should reduce (or match) the Brier score."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            random_state=42,
            flip_y=0.05,
        )
        y = y.astype(np.float32)

        base_model = LogisticRegression(max_iter=1000)
        base_model.fit(X, y)
        raw_probs = base_model.predict_proba(X)[:, 1].astype(np.float32)

        calibrated_model = CalibratedClassifierCV(
            estimator=LogisticRegression(max_iter=1000),
            cv=5,
            method="isotonic",
        )
        calibrated_model.fit(X, y)
        calibrated_probs = calibrated_model.predict_proba(X)[:, 1].astype(np.float32)

        raw_metrics = calculate_calibration_metrics(y, raw_probs, n_bins=10)
        cal_metrics = calculate_calibration_metrics(y, calibrated_probs, n_bins=10)

        assert cal_metrics["brier_score"] <= raw_metrics["brier_score"]


class TestCalculateSubgroupMetrics:
    """Tests for calculate_subgroup_metrics function."""

    def test_valid_subgroup_calculation(self):
        """Test that subgroup metrics are calculated correctly."""
        import pandas as pd

        # Create test data with two age groups
        df = pd.DataFrame(
            {
                "age_group": [
                    "young",
                    "young",
                    "young",
                    "young",
                    "old",
                    "old",
                    "old",
                    "old",
                ]
            }
        )
        labels = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float32)
        predictions = np.array(
            [0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.75, 0.85], dtype=np.float32
        )

        metrics = calculate_subgroup_metrics(df, labels, predictions, "age_group")

        # Check that both subgroups are present
        assert "young" in metrics
        assert "old" in metrics

        # Check that AUROC values are in valid range [0, 1]
        assert 0.0 <= metrics["young"] <= 1.0
        assert 0.0 <= metrics["old"] <= 1.0

        # For this data, both should have perfect separation (AUROC = 1.0)
        assert metrics["young"] == 1.0
        assert metrics["old"] == 1.0

    def test_disparate_subgroup_performance(self):
        """Test with different performance across subgroups."""
        import pandas as pd

        df = pd.DataFrame({"group": ["A", "A", "A", "A", "B", "B", "B", "B"]})
        # Group A: good separation
        # Group B: poor separation
        labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32)
        predictions = np.array(
            [0.1, 0.2, 0.8, 0.9, 0.4, 0.5, 0.6, 0.7], dtype=np.float32
        )

        metrics = calculate_subgroup_metrics(df, labels, predictions, "group")

        # Group A should have higher AUROC than Group B
        assert metrics["A"] > metrics["B"]

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        import pandas as pd

        df = pd.DataFrame({"group": ["A", "A", "B"]})
        labels = np.array([0, 1], dtype=np.float32)
        predictions = np.array([0.1, 0.9], dtype=np.float32)

        with pytest.raises(ValueError, match="same length"):
            calculate_subgroup_metrics(df, labels, predictions, "group")

    def test_missing_column_raises_error(self):
        """Test that missing subgroup column raises KeyError."""
        import pandas as pd

        df = pd.DataFrame({"age": [20, 30, 40, 50]})
        labels = np.array([0, 1, 0, 1], dtype=np.float32)
        predictions = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)

        with pytest.raises(KeyError, match="not found"):
            calculate_subgroup_metrics(df, labels, predictions, "gender")

    def test_subgroup_with_single_class_is_skipped(self):
        """Test that subgroups with only one class are skipped with warning."""
        import pandas as pd

        df = pd.DataFrame({"group": ["A", "A", "B", "B"]})
        # Group A: only class 0
        # Group B: both classes
        labels = np.array([0, 0, 0, 1], dtype=np.float32)
        predictions = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)

        metrics = calculate_subgroup_metrics(df, labels, predictions, "group")

        # Only Group B should be in the results
        assert "A" not in metrics
        assert "B" in metrics

    def test_all_invalid_subgroups_raises_error(self):
        """Test that having no valid subgroups raises ValueError."""
        import pandas as pd

        df = pd.DataFrame({"group": ["A", "A", "B", "B"]})
        # Both groups have only one class
        labels = np.array([0, 0, 1, 1], dtype=np.float32)
        predictions = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)

        with pytest.raises(ValueError, match="No valid subgroups"):
            calculate_subgroup_metrics(df, labels, predictions, "group")

    def test_categorical_subgroups_preserve_order(self):
        """Categorical subgroup columns should preserve declared ordering."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "group": pd.Categorical(
                    ["B", "A", "C", "B", "A", "C"],
                    categories=["B", "A", "C"],
                    ordered=True,
                )
            }
        )
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
        predictions = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6], dtype=np.float32)

        metrics = calculate_subgroup_metrics(df, labels, predictions, "group")

        assert list(metrics.keys()) == ["B", "A", "C"]

    def test_nan_subgroup_values_are_dropped_with_warning(self, caplog):
        """Rows with NaN subgroup values should be dropped with a warning."""
        import pandas as pd

        df = pd.DataFrame({"group": ["A", None, "B", "A", "B"]})
        labels = np.array([0, 1, 0, 1, 1], dtype=np.float32)
        predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.7], dtype=np.float32)

        with caplog.at_level(logging.WARNING):
            metrics = calculate_subgroup_metrics(df, labels, predictions, "group")

        assert "Dropped 1 rows" in caplog.text
        assert set(metrics.keys()) == {"A", "B"}

    def test_all_nan_subgroup_values_raise_error(self):
        """If all subgroup entries are NaN, raise ValueError."""
        import pandas as pd

        df = pd.DataFrame({"group": [np.nan, np.nan]})
        labels = np.array([0, 1], dtype=np.float32)
        predictions = np.array([0.4, 0.6], dtype=np.float32)

        with pytest.raises(ValueError, match="All rows have NaN"):
            calculate_subgroup_metrics(df, labels, predictions, "group")
