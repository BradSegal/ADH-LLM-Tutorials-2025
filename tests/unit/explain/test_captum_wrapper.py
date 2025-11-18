"""
Unit tests for the explain.captum_wrapper module.

These tests validate the feature attribution and plotting functions using
mock models and synthetic data.
"""

from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest
import torch

from core.explain.captum_wrapper import explain_instance, plot_feature_attributions
from core.models.base import BaseSequenceModel

# Use non-interactive backend for testing
matplotlib.use("Agg")


class SimpleMockModel(BaseSequenceModel):
    """Simple mock model for testing explanations."""

    def __init__(self, input_dim: int = 5) -> None:
        super().__init__()
        # Add a simple linear layer to make it a proper nn.Module
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return simple output based on input."""
        # Just return a simple function of the input
        # x shape: (batch, seq_len, features)
        # Take mean across sequence dimension
        output: torch.Tensor = self.linear(x.mean(dim=1))
        return output


class MultiClassMockModel(BaseSequenceModel):
    """Mock model that emits multi-class logits for validation tests."""

    def __init__(self, input_dim: int = 5, num_classes: int = 3) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.linear(x.mean(dim=1))
        return output


class PrepareMaskMockModel(SimpleMockModel):
    """Mock model to verify prepare_mask hook usage."""

    def __init__(self, input_dim: int = 5) -> None:
        super().__init__(input_dim=input_dim)
        self.prepare_calls = 0

    def prepare_mask(self, mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        self.prepare_calls += 1
        return super().prepare_mask(mask, batch_size)


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    return SimpleMockModel()


@pytest.fixture
def sample_instance():
    """Create sample instance data for testing."""
    instance = torch.randn(10, 5)  # 10 timesteps, 5 features
    mask = torch.ones(10, dtype=torch.bool)
    return instance, mask


@pytest.fixture
def sample_attributions():
    """Create sample attribution data for testing plotting."""
    # 10 timesteps, 5 features
    attributions = np.random.randn(10, 5).astype(np.float32)
    feature_names = [f"feature_{i}" for i in range(5)]
    return attributions, feature_names


class TestExplainInstance:
    """Tests for explain_instance function."""

    def test_explain_instance_output_shape(self, mock_model, sample_instance):
        """Test that explain_instance returns correct output shape."""
        instance, mask = sample_instance

        attributions = explain_instance(
            mock_model, instance, mask, target_label=0, n_steps=10
        )

        # Should return same shape as input instance
        assert attributions.shape == instance.shape
        assert isinstance(attributions, np.ndarray)

    def test_explain_instance_defaults_target_label(self, mock_model, sample_instance):
        """target_label should default to 0 for single-logit models."""
        instance, mask = sample_instance

        attributions = explain_instance(mock_model, instance, mask, n_steps=5)

        assert attributions.shape == instance.shape

    def test_explain_instance_invalid_model(self, sample_instance):
        """Test that non-BaseSequenceModel raises TypeError."""
        instance, mask = sample_instance
        invalid_model = MagicMock()

        with pytest.raises(TypeError, match="BaseSequenceModel"):
            explain_instance(invalid_model, instance, mask)

    def test_explain_instance_wrong_instance_dims(self, mock_model):
        """Test that instance with wrong dimensions raises ValueError."""
        # 3D instance (should be 2D)
        instance = torch.randn(2, 10, 5)
        mask = torch.ones(10, dtype=torch.bool)

        with pytest.raises(ValueError, match="must be 2D"):
            explain_instance(mock_model, instance, mask)

    def test_explain_instance_wrong_mask_dims(self, mock_model):
        """Test that mask with wrong dimensions raises ValueError."""
        instance = torch.randn(10, 5)
        # 2D mask (should be 1D)
        mask = torch.ones(10, 5, dtype=torch.bool)

        with pytest.raises(ValueError, match="must be 1D"):
            explain_instance(mock_model, instance, mask)

    def test_explain_instance_mismatched_lengths(self, mock_model):
        """Test that mismatched instance and mask lengths raise ValueError."""
        instance = torch.randn(10, 5)
        mask = torch.ones(8, dtype=torch.bool)  # Different length

        with pytest.raises(ValueError, match="must match"):
            explain_instance(mock_model, instance, mask)

    def test_explain_instance_unsupported_method(self, mock_model, sample_instance):
        """Test that unsupported attribution method raises ValueError."""
        instance, mask = sample_instance

        with pytest.raises(ValueError, match="Unsupported attribution method"):
            explain_instance(
                mock_model, instance, mask, method="gradient_shap"  # type: ignore[arg-type]
            )

    def test_explain_instance_model_in_eval_mode(self, mock_model, sample_instance):
        """Test that model is set to eval mode during attribution."""
        instance, mask = sample_instance

        # Set model to training mode
        mock_model.train()
        assert mock_model.training is True

        # Run explanation
        explain_instance(mock_model, instance, mask, target_label=0, n_steps=10)

        # Model should still be in eval mode
        assert mock_model.training is False

    def test_explain_instance_device_handling(self, mock_model, sample_instance):
        """Test that explain_instance handles device placement correctly."""
        instance, mask = sample_instance

        # Model is on CPU by default
        attributions = explain_instance(
            mock_model, instance, mask, target_label=0, n_steps=10
        )

        # Should return numpy array regardless of device
        assert isinstance(attributions, np.ndarray)

    def test_explain_instance_different_n_steps(self, mock_model, sample_instance):
        """Test that explanation runs successfully with different n_steps."""
        instance, mask = sample_instance

        # Try different number of steps (both should work)
        attr_few_steps = explain_instance(
            mock_model, instance, mask, target_label=0, n_steps=5
        )
        attr_many_steps = explain_instance(
            mock_model, instance, mask, target_label=0, n_steps=50
        )

        # Both should have the same shape as input
        assert attr_few_steps.shape == instance.shape
        assert attr_many_steps.shape == instance.shape

    def test_explain_instance_uses_prepare_mask(self, sample_instance):
        instance, mask = sample_instance
        model = PrepareMaskMockModel()

        explain_instance(model, instance, mask, n_steps=3)

        assert model.prepare_calls > 0

    def test_explain_instance_requires_target_for_multi_output(self, sample_instance):
        """Models with multi-class logits must provide explicit target label."""
        instance, mask = sample_instance
        multi_model = MultiClassMockModel()

        with pytest.raises(ValueError, match="target_label must be provided"):
            explain_instance(multi_model, instance, mask, n_steps=5)

        # Providing a valid label should work
        attributions = explain_instance(
            multi_model, instance, mask, target_label=2, n_steps=5
        )
        assert attributions.shape == instance.shape

    def test_explain_instance_rejects_invalid_target_index(self, sample_instance):
        """Target label must be within valid bounds for model outputs."""
        instance, mask = sample_instance
        multi_model = MultiClassMockModel()

        with pytest.raises(ValueError, match="valid output range"):
            explain_instance(multi_model, instance, mask, target_label=5, n_steps=5)


class TestPlotFeatureAttributions:
    """Tests for plot_feature_attributions function."""

    def test_plot_feature_attributions_creates_figure(self, sample_attributions):
        """Test that plotting creates a valid matplotlib Figure."""
        attributions, feature_names = sample_attributions

        fig = plot_feature_attributions(attributions, feature_names, top_k=3)

        assert fig is not None
        assert len(fig.axes) == 1  # Should have one axis

    def test_plot_feature_attributions_top_k(self, sample_attributions):
        """Test that top_k parameter limits the number of features shown."""
        attributions, feature_names = sample_attributions

        # Plot only top 3 features
        fig = plot_feature_attributions(attributions, feature_names, top_k=3)

        ax = fig.axes[0]
        # Check that only 3 bars are plotted
        assert len(ax.patches) == 3

    def test_plot_wrong_attribution_dims(self):
        """Test that wrong attribution dimensions raise ValueError."""
        # 1D attributions (should be 2D)
        attributions = np.random.randn(10).astype(np.float32)
        feature_names = ["f1", "f2", "f3"]

        with pytest.raises(ValueError, match="must be 2D"):
            plot_feature_attributions(attributions, feature_names)

    def test_plot_mismatched_feature_names(self, sample_attributions):
        """Test that mismatched feature names raise ValueError."""
        attributions, _ = sample_attributions

        # Wrong number of feature names
        wrong_names = ["f1", "f2", "f3"]  # Only 3 names for 5 features

        with pytest.raises(ValueError, match="must match"):
            plot_feature_attributions(attributions, wrong_names)

    def test_plot_different_aggregate_methods(self, sample_attributions):
        """Test that different aggregation methods work correctly."""
        attributions, feature_names = sample_attributions

        # Test all supported aggregation methods
        from typing import Literal, cast

        for method_str in ["mean", "max", "sum"]:
            method = cast(Literal["mean", "max", "sum"], method_str)
            fig = plot_feature_attributions(
                attributions, feature_names, top_k=3, aggregate_method=method
            )
            assert fig is not None
            assert len(fig.axes) == 1

    def test_plot_unsupported_aggregate_method(self, sample_attributions):
        """Test that unsupported aggregation method raises ValueError."""
        attributions, feature_names = sample_attributions

        with pytest.raises(ValueError, match="Unsupported aggregate_method"):
            plot_feature_attributions(
                attributions, feature_names, aggregate_method="median"  # type: ignore[arg-type]
            )

    def test_plot_custom_figsize(self, sample_attributions):
        """Test that custom figsize is applied correctly."""
        attributions, feature_names = sample_attributions

        custom_size = (12, 10)
        fig = plot_feature_attributions(
            attributions, feature_names, top_k=3, figsize=custom_size
        )

        # Check figure size (within small tolerance for rounding)
        assert abs(fig.get_figwidth() - custom_size[0]) < 0.1
        assert abs(fig.get_figheight() - custom_size[1]) < 0.1

    def test_plot_aggregation_correctness(self):
        """Test that aggregation methods produce expected results."""
        # Create known attributions
        attributions = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        feature_names = ["f1", "f2", "f3"]

        # Mean aggregation
        fig_mean = plot_feature_attributions(
            attributions, feature_names, top_k=3, aggregate_method="mean"
        )

        # The aggregation should produce: mean(abs([1,4])),
        # mean(abs([2,5])), mean(abs([3,6]))
        # = [2.5, 3.5, 4.5]
        # Top feature should be f3

        ax = fig_mean.axes[0]
        y_labels = [label.get_text() for label in ax.get_yticklabels()]

        # Top label (inverted y-axis) should be f3
        assert "f3" in y_labels[0]
