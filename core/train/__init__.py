"""
Training utilities and harnesses.

This subpackage contains the unified Trainer class that handles the complete
training and evaluation loop for all sequence models. By centralizing training
logic here, we avoid code duplication across notebooks and enforce the DRY
principle.
"""

from core.train.trainer import Trainer

__all__ = ["Trainer"]
