"""
Model explainability utilities using Captum.

This module provides tools for understanding model predictions through
feature attribution methods like Integrated Gradients.
"""

from core.explain.captum_wrapper import explain_instance, plot_feature_attributions

__all__ = ["explain_instance", "plot_feature_attributions"]
