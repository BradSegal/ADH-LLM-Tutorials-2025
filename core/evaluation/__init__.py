"""
Model evaluation utilities.

This module provides tools for evaluating trained sequence models, including
generating predictions and calculating classification metrics.
"""

from core.evaluation.metrics import (
    CalibrationMetrics,
    PredictionResults,
    calculate_calibration_metrics,
    calculate_classification_metrics,
    calculate_subgroup_metrics,
    get_predictions,
)

__all__ = [
    "get_predictions",
    "calculate_classification_metrics",
    "calculate_calibration_metrics",
    "calculate_subgroup_metrics",
    "PredictionResults",
    "CalibrationMetrics",
]
