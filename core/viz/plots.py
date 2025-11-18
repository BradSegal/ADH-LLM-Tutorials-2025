"""
Plotting functions for model evaluation and comparison.

This module provides standardized plotting utilities for visualizing model
performance, including ROC curves and other evaluation metrics.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import roc_curve

from core.evaluation.metrics import CalibrationMetrics

logger = logging.getLogger(__name__)


def plot_roc_curves(
    results_dict: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]],
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    """
    Plot ROC curves for multiple models on a single figure.

    Creates a publication-quality plot comparing the ROC (Receiver Operating
    Characteristic) curves for multiple models. Each curve shows the trade-off
    between true positive rate and false positive rate at different classification
    thresholds.

    Parameters
    ----------
    results_dict : dict[str, tuple[NDArray, NDArray]]
        Dictionary mapping model names to (labels, predictions) tuples.
        Each tuple should contain:
        - labels: Ground truth binary labels, shape (N,)
        - predictions: Predicted probabilities, shape (N,)
    figsize : tuple[int, int], default=(10, 8)
        Figure size as (width, height) in inches.

    Returns
    -------
    Figure
        Matplotlib Figure object containing the ROC curve plot.

    Raises
    ------
    ValueError
        If results_dict is empty, or if any model's labels/predictions are
        invalid or mismatched in length.

    Examples
    --------
    >>> from core.viz.plots import plot_roc_curves
    >>> import numpy as np
    >>>
    >>> results = {
    ...     'GRU': (np.array([0, 0, 1, 1]), np.array([0.1, 0.3, 0.8, 0.9])),
    ...     'LSTM': (np.array([0, 0, 1, 1]), np.array([0.2, 0.4, 0.7, 0.95])),
    ... }
    >>> fig = plot_roc_curves(results)
    >>> fig.savefig('roc_comparison.png')
    """
    # Validate input
    if not results_dict:
        raise ValueError("results_dict must not be empty.")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve for each model
    for model_name, (labels, predictions) in results_dict.items():
        # Validate data
        if len(labels) != len(predictions):
            raise ValueError(
                f"For model '{model_name}': labels and predictions must have "
                f"the same length. Got {len(labels)} labels and "
                f"{len(predictions)} predictions."
            )

        if len(labels) == 0:
            raise ValueError(
                f"For model '{model_name}': labels and predictions must not be empty."
            )

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(labels, predictions)

        # Calculate AUC for the legend
        from sklearn.metrics import auc

        roc_auc = auc(fpr, tpr)

        # Plot the curve
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {roc_auc:.3f})",
        )

    # Plot diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier (AUC = 0.500)")

    # Formatting
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))

    plt.tight_layout()

    logger.info(f"Generated ROC curve comparison for {len(results_dict)} models")

    return fig


def plot_calibration_curve(
    calibration_data: CalibrationMetrics,
    model_name: str = "Model",
    figsize: tuple[int, int] = (8, 8),
) -> Figure:
    """
    Plot a calibration curve (reliability diagram) for a probabilistic model.

    Creates a reliability diagram showing the relationship between predicted
    probabilities and observed event frequencies. A well-calibrated model's
    points should lie close to the diagonal line (perfect calibration).

    Parameters
    ----------
    calibration_data : CalibrationMetrics
        TypedDict containing calibration metrics from
        :func:`~core.evaluation.metrics.calculate_calibration_metrics`.
        Must contain keys:
        - 'prob_true': True probability in each bin (mean label value)
        - 'prob_pred': Mean predicted probability in each bin
        - 'brier_score': Brier score (shown in plot title)
    model_name : str, default="Model"
        Name of the model to display in the plot title and legend.
    figsize : tuple[int, int], default=(8, 8)
        Figure size as (width, height) in inches.

    Returns
    -------
    Figure
        Matplotlib Figure object containing the calibration plot.

    Raises
    ------
    KeyError
        If calibration_data is missing required keys.
    ValueError
        If prob_true and prob_pred arrays have different lengths.

    Examples
    --------
    >>> from core.evaluation.metrics import calculate_calibration_metrics
    >>> from core.viz.plots import plot_calibration_curve
    >>> import numpy as np
    >>>
    >>> labels = np.array([0, 0, 1, 1, 1])
    >>> predictions = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    >>> cal_data = calculate_calibration_metrics(labels, predictions)
    >>> fig = plot_calibration_curve(cal_data, model_name='LSTM')
    >>> fig.savefig('calibration_plot.png')
    """
    # Validate required keys
    required_keys = ["prob_true", "prob_pred", "brier_score"]
    for key in required_keys:
        if key not in calibration_data:
            raise KeyError(
                f"calibration_data must contain key '{key}'. "
                f"Available keys: {list(calibration_data.keys())}"
            )

    prob_true = calibration_data["prob_true"]
    prob_pred = calibration_data["prob_pred"]
    brier_score = calibration_data["brier_score"]

    # Type guard for mypy
    if not isinstance(prob_true, np.ndarray) or not isinstance(prob_pred, np.ndarray):
        raise TypeError("prob_true and prob_pred must be numpy arrays")

    # Validate array shapes
    if len(prob_true) != len(prob_pred):
        raise ValueError(
            f"prob_true and prob_pred must have the same length. "
            f"Got {len(prob_true)} and {len(prob_pred)}."
        )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the calibration curve
    ax.plot(
        prob_pred,
        prob_true,
        marker="o",
        linewidth=2,
        markersize=8,
        label=f"{model_name} (Brier={brier_score:.3f})",
    )

    # Plot perfect calibration line
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=1.5,
        label="Perfect Calibration",
    )

    # Formatting
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives (True Probability)", fontsize=12)
    ax.set_title(f"Calibration Curve - {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))

    # Add text box with interpretation guide
    textstr = (
        "Interpretation:\n"
        "• Points above diagonal: Model underconfident\n"
        "• Points below diagonal: Model overconfident\n"
        "• Lower Brier score is better"
    )
    props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.3}
    ax.text(
        0.55,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=props,
    )

    plt.tight_layout()

    logger.info(f"Generated calibration curve for {model_name}")

    return fig


def plot_subgroup_performance(
    subgroup_metrics: dict[str, float],
    metric_name: str = "AUROC",
    figsize: tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot performance metrics across different subgroups.

    Creates a bar chart showing model performance (typically AUROC) for each
    demographic or clinical subgroup. This visualization makes it easy to
    identify performance disparities that could lead to unfair outcomes.

    Parameters
    ----------
    subgroup_metrics : dict[str, float]
        Dictionary mapping subgroup names to metric values.
        Output from :func:`~core.evaluation.metrics.calculate_subgroup_metrics`.
        For example: {'0-40': 0.85, '41-60': 0.78, '61-80': 0.72}
    metric_name : str, default="AUROC"
        Name of the metric being plotted (for axis labels and title).
    figsize : tuple[int, int], default=(10, 6)
        Figure size as (width, height) in inches.

    Returns
    -------
    Figure
        Matplotlib Figure object containing the subgroup performance plot.

    Raises
    ------
    ValueError
        If subgroup_metrics is empty.

    Examples
    --------
    >>> from core.evaluation.metrics import calculate_subgroup_metrics
    >>> from core.viz.plots import plot_subgroup_performance
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({'age_group': ['young', 'young', 'old', 'old']})
    >>> labels = np.array([0, 1, 0, 1], dtype=np.float32)
    >>> predictions = np.array([0.2, 0.8, 0.3, 0.9], dtype=np.float32)
    >>> metrics = calculate_subgroup_metrics(df, labels, predictions, 'age_group')
    >>> fig = plot_subgroup_performance(metrics)
    >>> fig.savefig('subgroup_fairness.png')
    """
    # Validate input
    if not subgroup_metrics:
        raise ValueError("subgroup_metrics must not be empty.")

    # Extract subgroup names and values
    subgroups = list(subgroup_metrics.keys())
    values = list(subgroup_metrics.values())

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar chart
    bars = ax.bar(subgroups, values, color="steelblue", alpha=0.7, edgecolor="black")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Calculate and plot mean line
    mean_value = float(np.mean(values))
    ax.axhline(
        mean_value,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean {metric_name} = {mean_value:.3f}",
    )

    # Formatting
    ax.set_xlabel("Subgroup", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(
        f"{metric_name} by Subgroup - Fairness Analysis", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Set y-axis limits for better visualization
    # For AUROC, use [0, 1], otherwise use data-driven limits
    if metric_name.upper() == "AUROC":
        ax.set_ylim((0.0, 1.0))
    else:
        # Add some padding to the y-axis
        y_min = max(0, min(values) - 0.1)
        y_max = min(1, max(values) + 0.1)
        ax.set_ylim((y_min, y_max))

    plt.tight_layout()

    logger.info(f"Generated subgroup performance plot for {len(subgroups)} subgroups")

    return fig
