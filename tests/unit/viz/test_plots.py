"""
Unit tests for the viz.plots module.

These tests validate the ROC curve plotting functionality using
synthetic data to ensure correct visualization behavior.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from core.evaluation.metrics import CalibrationMetrics
from core.viz.plots import (
    plot_calibration_curve,
    plot_roc_curves,
    plot_subgroup_performance,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    """Ensure matplotlib figures are closed between tests."""
    yield
    plt.close("all")


@pytest.fixture
def sample_binary_results():
    """Create sample binary classification results for testing."""
    # Create perfect predictions for model 1
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    perfect_preds = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9], dtype=np.float32)

    # Create slightly worse predictions for model 2
    good_preds = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.85], dtype=np.float32)

    # Create random predictions for model 3
    np.random.seed(42)
    random_preds = np.random.rand(6).astype(np.float32)

    results = {
        "Perfect Model": (labels, perfect_preds),
        "Good Model": (labels, good_preds),
        "Random Model": (labels, random_preds),
    }

    return results


@pytest.fixture
def single_model_results():
    """Create results for a single model."""
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    predictions = np.array([0.1, 0.3, 0.8, 0.9], dtype=np.float32)

    return {"GRU": (labels, predictions)}


class TestPlotROCCurves:
    """Tests for plot_roc_curves function."""

    def test_plot_creates_valid_figure(self, sample_binary_results):
        """Test that plot_roc_curves creates a valid matplotlib Figure."""
        fig = plot_roc_curves(sample_binary_results)

        assert fig is not None
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_plot_includes_all_models(self, sample_binary_results):
        """Test that all models are plotted."""
        fig = plot_roc_curves(sample_binary_results)
        ax = fig.axes[0]

        # Should have 3 model curves + 1 diagonal reference line = 4 lines
        assert len(ax.lines) == 4

    def test_plot_has_correct_labels(self, sample_binary_results):
        """Test that the plot has correct axis labels and title."""
        fig = plot_roc_curves(sample_binary_results)
        ax = fig.axes[0]

        assert ax.get_xlabel() == "False Positive Rate"
        assert ax.get_ylabel() == "True Positive Rate"
        assert "ROC Curves" in ax.get_title()

    def test_plot_includes_legend(self, sample_binary_results):
        """Test that the plot includes a legend with model names and AUC."""
        fig = plot_roc_curves(sample_binary_results)
        ax = fig.axes[0]

        legend = ax.get_legend()
        assert legend is not None

        # Get legend text
        legend_texts = [text.get_text() for text in legend.get_texts()]

        # Check that model names appear in legend
        assert any("Perfect Model" in text for text in legend_texts)
        assert any("Good Model" in text for text in legend_texts)
        assert any("Random Model" in text for text in legend_texts)

        # Check that AUC values appear
        assert any("AUC" in text for text in legend_texts)

    def test_plot_single_model(self, single_model_results):
        """Test that plotting works with a single model."""
        fig = plot_roc_curves(single_model_results)

        assert fig is not None
        assert isinstance(fig, Figure)

        ax = fig.axes[0]
        # Should have 1 model curve + 1 diagonal reference = 2 lines
        assert len(ax.lines) == 2

    def test_plot_empty_dict_raises_error(self):
        """Test that empty results dictionary raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_roc_curves({})

    def test_plot_mismatched_lengths_raises_error(self):
        """Test that mismatched label/prediction lengths raise ValueError."""
        results = {
            "Model1": (
                np.array([0, 1, 0], dtype=np.float32),
                np.array([0.1, 0.9], dtype=np.float32),  # Wrong length
            )
        }

        with pytest.raises(ValueError, match="same length"):
            plot_roc_curves(results)

    def test_plot_empty_arrays_raise_error(self):
        """Test that empty arrays raise ValueError."""
        results = {
            "Model1": (np.array([], dtype=np.float32), np.array([], dtype=np.float32))
        }

        with pytest.raises(ValueError, match="must not be empty"):
            plot_roc_curves(results)

    def test_plot_custom_figsize(self, single_model_results):
        """Test that custom figsize is applied correctly."""
        custom_size = (12, 10)
        fig = plot_roc_curves(single_model_results, figsize=custom_size)

        # Check figure size (within small tolerance for rounding)
        assert abs(fig.get_figwidth() - custom_size[0]) < 0.1
        assert abs(fig.get_figheight() - custom_size[1]) < 0.1

    def test_plot_axis_limits(self, sample_binary_results):
        """Test that axis limits are set correctly to [0, 1]."""
        fig = plot_roc_curves(sample_binary_results)
        ax = fig.axes[0]

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # X-axis should be [0, 1]
        assert xlim[0] == 0.0
        assert xlim[1] == 1.0

        # Y-axis should be [0, 1.05] for some padding
        assert ylim[0] == 0.0
        assert ylim[1] == 1.05

    def test_plot_includes_diagonal_reference(self, sample_binary_results):
        """Test that the diagonal reference line (random classifier) is plotted."""
        fig = plot_roc_curves(sample_binary_results)
        ax = fig.axes[0]

        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        # Check for random classifier in legend
        assert any("Random Classifier" in text for text in legend_texts)

    def test_plot_with_perfect_classifier(self):
        """Test plotting with a perfect classifier (AUC = 1.0)."""
        labels = np.array([0, 0, 1, 1], dtype=np.float32)
        perfect_preds = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        results = {"Perfect": (labels, perfect_preds)}

        fig = plot_roc_curves(results)
        assert fig is not None

        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        # Perfect classifier should have AUC = 1.000
        assert any("1.000" in text for text in legend_texts)

    def test_plot_with_multiple_models_different_performance(self):
        """Test that models with different performance levels are all plotted."""
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)

        # Very good model
        good_preds = np.array(
            [0.1, 0.2, 0.15, 0.25, 0.75, 0.85, 0.9, 0.8], dtype=np.float32
        )

        # Mediocre model
        mediocre_preds = np.array(
            [0.3, 0.4, 0.35, 0.45, 0.55, 0.65, 0.6, 0.7], dtype=np.float32
        )

        # Bad model (worse than random)
        bad_preds = np.array(
            [0.9, 0.8, 0.85, 0.75, 0.25, 0.15, 0.1, 0.2], dtype=np.float32
        )

        results = {
            "Good": (labels, good_preds),
            "Mediocre": (labels, mediocre_preds),
            "Bad": (labels, bad_preds),
        }

        fig = plot_roc_curves(results)
        ax = fig.axes[0]

        # Should have 3 model lines + 1 diagonal = 4 lines
        assert len(ax.lines) == 4

    def test_plot_grid_is_enabled(self, sample_binary_results):
        """Test that grid is enabled for better readability."""
        fig = plot_roc_curves(sample_binary_results)
        ax = fig.axes[0]

        # Check that grid is enabled
        assert ax.get_xgridlines()[0].get_visible() or ax.xaxis.get_gridlines()


class TestPlotCalibrationCurve:
    """Tests for plot_calibration_curve function."""

    @pytest.fixture
    def sample_calibration_data(self) -> CalibrationMetrics:
        """Create sample calibration data for testing."""
        return CalibrationMetrics(
            brier_score=0.123,
            prob_true=np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32),
            prob_pred=np.array([0.15, 0.35, 0.45, 0.75, 0.85], dtype=np.float32),
        )

    def test_plot_creates_valid_figure(self, sample_calibration_data):
        """Test that plot_calibration_curve creates a valid matplotlib Figure."""
        fig = plot_calibration_curve(sample_calibration_data)

        assert fig is not None
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_plot_has_correct_labels(self, sample_calibration_data):
        """Test that the plot has correct axis labels and title."""
        fig = plot_calibration_curve(sample_calibration_data, model_name="LSTM")
        ax = fig.axes[0]

        assert "Mean Predicted Probability" in ax.get_xlabel()
        assert "Fraction of Positives" in ax.get_ylabel()
        assert "LSTM" in ax.get_title()

    def test_plot_includes_diagonal_reference(self, sample_calibration_data):
        """Test that perfect calibration diagonal line is plotted."""
        fig = plot_calibration_curve(sample_calibration_data)
        ax = fig.axes[0]

        # Should have calibration curve + diagonal = 2 lines
        assert len(ax.lines) == 2

        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]
        assert any("Perfect Calibration" in text for text in legend_texts)

    def test_plot_shows_brier_score(self, sample_calibration_data):
        """Test that Brier score is displayed in the legend."""
        fig = plot_calibration_curve(sample_calibration_data)
        ax = fig.axes[0]

        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]

        # Brier score should appear in legend
        assert any("Brier" in text for text in legend_texts)
        assert any("0.123" in text for text in legend_texts)

    def test_plot_missing_key_raises_error(self):
        """Test that missing required keys raise KeyError."""
        incomplete_data = {
            "brier_score": 0.1,
            "prob_true": np.array([0.1, 0.2], dtype=np.float32),
            # Missing 'prob_pred'
        }

        with pytest.raises(KeyError, match="prob_pred"):
            plot_calibration_curve(incomplete_data)  # type: ignore[arg-type]

    def test_plot_mismatched_array_lengths_raises_error(self):
        """Test that mismatched array lengths raise ValueError."""
        bad_data = CalibrationMetrics(
            brier_score=0.1,
            prob_true=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            prob_pred=np.array([0.1, 0.2], dtype=np.float32),  # Wrong length
        )

        with pytest.raises(ValueError, match="same length"):
            plot_calibration_curve(bad_data)

    def test_plot_custom_model_name(self, sample_calibration_data):
        """Test that custom model name appears in plot."""
        model_name = "Transformer-XL"
        fig = plot_calibration_curve(sample_calibration_data, model_name=model_name)
        ax = fig.axes[0]

        assert model_name in ax.get_title()

        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]
        assert any(model_name in text for text in legend_texts)

    def test_plot_custom_figsize(self, sample_calibration_data):
        """Test that custom figsize is applied correctly."""
        custom_size = (10, 10)
        fig = plot_calibration_curve(sample_calibration_data, figsize=custom_size)

        assert abs(fig.get_figwidth() - custom_size[0]) < 0.1
        assert abs(fig.get_figheight() - custom_size[1]) < 0.1

    def test_plot_axis_limits(self, sample_calibration_data):
        """Test that axis limits are set to [0, 1]."""
        fig = plot_calibration_curve(sample_calibration_data)
        ax = fig.axes[0]

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        assert xlim[0] == 0.0
        assert xlim[1] == 1.0
        assert ylim[0] == 0.0
        assert ylim[1] == 1.0

    def test_plot_includes_interpretation_guide(self, sample_calibration_data):
        """Test that interpretation guide text box is present."""
        fig = plot_calibration_curve(sample_calibration_data)
        ax = fig.axes[0]

        # Check that there is at least one text annotation
        texts = ax.texts
        assert len(texts) > 0

        # Check that interpretation text contains expected keywords
        text_content = " ".join([t.get_text() for t in texts])
        assert "underconfident" in text_content or "overconfident" in text_content


class TestPlotSubgroupPerformance:
    """Tests for plot_subgroup_performance function."""

    @pytest.fixture
    def sample_subgroup_metrics(self):
        """Create sample subgroup metrics for testing."""
        return {
            "0-40": 0.85,
            "41-60": 0.78,
            "61-80": 0.72,
            "81+": 0.68,
        }

    def test_plot_creates_valid_figure(self, sample_subgroup_metrics):
        """Test that plot_subgroup_performance creates a valid matplotlib Figure."""
        fig = plot_subgroup_performance(sample_subgroup_metrics)

        assert fig is not None
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 1

    def test_plot_has_correct_labels(self, sample_subgroup_metrics):
        """Test that the plot has correct axis labels and title."""
        fig = plot_subgroup_performance(sample_subgroup_metrics)
        ax = fig.axes[0]

        assert "Subgroup" in ax.get_xlabel()
        assert "AUROC" in ax.get_ylabel()
        assert "Fairness" in ax.get_title()

    def test_plot_includes_all_subgroups(self, sample_subgroup_metrics):
        """Test that all subgroups are plotted."""
        fig = plot_subgroup_performance(sample_subgroup_metrics)
        ax = fig.axes[0]

        # Should have one bar per subgroup
        bars = [
            patch
            for patch in ax.patches
            if hasattr(patch, "get_height") and patch.get_height() > 0
        ]
        assert len(bars) == 4

    def test_plot_includes_mean_line(self, sample_subgroup_metrics):
        """Test that mean performance line is plotted."""
        fig = plot_subgroup_performance(sample_subgroup_metrics)
        ax = fig.axes[0]

        # Check for horizontal lines (mean line)
        xdata = ax.lines[0].get_xdata()
        hlines = [line for line in ax.lines if len(line.get_xdata()) == len(xdata)]  # type: ignore[arg-type]
        assert len(hlines) > 0

        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [text.get_text() for text in legend.get_texts()]
        assert any("Mean" in text for text in legend_texts)

    def test_plot_empty_dict_raises_error(self):
        """Test that empty metrics dictionary raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            plot_subgroup_performance({})

    def test_plot_custom_metric_name(self, sample_subgroup_metrics):
        """Test that custom metric name appears in plot."""
        metric_name = "F1-Score"
        fig = plot_subgroup_performance(
            sample_subgroup_metrics, metric_name=metric_name
        )
        ax = fig.axes[0]

        assert metric_name in ax.get_ylabel()
        assert metric_name in ax.get_title()

    def test_plot_custom_figsize(self, sample_subgroup_metrics):
        """Test that custom figsize is applied correctly."""
        custom_size = (12, 8)
        fig = plot_subgroup_performance(sample_subgroup_metrics, figsize=custom_size)

        assert abs(fig.get_figwidth() - custom_size[0]) < 0.1
        assert abs(fig.get_figheight() - custom_size[1]) < 0.1

    def test_plot_auroc_axis_limits(self, sample_subgroup_metrics):
        """Test that AUROC plots use [0, 1] y-axis limits."""
        fig = plot_subgroup_performance(sample_subgroup_metrics, metric_name="AUROC")
        ax = fig.axes[0]

        ylim = ax.get_ylim()
        assert ylim[0] == 0.0
        assert ylim[1] == 1.0

    def test_plot_bar_labels_present(self, sample_subgroup_metrics):
        """Test that value labels are displayed on top of bars."""
        fig = plot_subgroup_performance(sample_subgroup_metrics)
        ax = fig.axes[0]

        # Check that there are text labels on the plot
        texts = ax.texts
        assert len(texts) > 0

        # Check that at least one label contains a metric value
        text_content = [t.get_text() for t in texts]
        assert any("0." in text for text in text_content)

    def test_plot_single_subgroup(self):
        """Test that plotting works with a single subgroup."""
        single_subgroup = {"All": 0.82}
        fig = plot_subgroup_performance(single_subgroup)

        assert fig is not None
        assert isinstance(fig, Figure)

    def test_plot_with_different_performance_levels(self):
        """Test plotting with widely varying performance levels."""
        varying_metrics = {
            "High": 0.95,
            "Medium": 0.75,
            "Low": 0.55,
        }

        fig = plot_subgroup_performance(varying_metrics)
        ax = fig.axes[0]

        # Should have 3 bars
        bars = [
            patch
            for patch in ax.patches
            if hasattr(patch, "get_height") and patch.get_height() > 0
        ]
        assert len(bars) == 3
