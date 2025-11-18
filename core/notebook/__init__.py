"""Utilities shared across instructional notebooks.

This package centralizes runtime helpers so every notebook can bootstrap the
environment the same way, ensuring we respect the DRY mandate from AGENTS.md.
"""

from core.notebook.display import (
    display_hyperparameter_table,
    plot_training_dashboard,
    print_pre_training_summary,
)
from core.notebook.runtime import ensure_project_root, get_feature_names

__all__ = [
    "ensure_project_root",
    "get_feature_names",
    "display_hyperparameter_table",
    "plot_training_dashboard",
    "print_pre_training_summary",
]
