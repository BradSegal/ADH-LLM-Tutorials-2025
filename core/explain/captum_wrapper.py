"""
Wrapper for Captum explainability library.

This module provides a simplified interface to the Captum library for
computing feature attributions using methods like Integrated Gradients.
"""

import logging
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch import Tensor

from core.models.base import BaseSequenceModel

logger = logging.getLogger(__name__)


def explain_instance(
    model: BaseSequenceModel,
    instance_tensor: Tensor,
    mask: Tensor,
    target_label: int | None = None,
    method: Literal["integrated_gradients"] = "integrated_gradients",
    n_steps: int = 50,
) -> NDArray[np.float32]:
    """
    Compute feature attributions for a single prediction.

    Uses Captum's attribution methods to determine which features were most
    important for the model's prediction on a specific instance. This helps
    answer "Why did the model predict this outcome?"

    Parameters
    ----------
    model : BaseSequenceModel
        The trained sequence model to explain. Must inherit from BaseSequenceModel.
    instance_tensor : Tensor
        Input features for a single instance, shape (seq_len, n_features).
        Should NOT include batch dimension.
    mask : Tensor
        Attention mask for the instance, shape (seq_len,). Boolean tensor
        indicating which timesteps are valid (True) vs. padding (False).
    target_label : int | None, default=None
        The target class to explain. If None and the model outputs a single
        logit per instance, the target defaults to 0. For multi-class models,
        you must provide an explicit class index in ``[0, n_classes)``.
    method : Literal["integrated_gradients"], default="integrated_gradients"
        Attribution method to use. Currently only "integrated_gradients" is supported.
    n_steps : int, default=50
        Number of steps for the Integrated Gradients approximation. More steps
        give more accurate attributions but take longer to compute.

    Returns
    -------
    NDArray[np.float32]
        Feature attributions, shape (seq_len, n_features). Positive values
        indicate features that pushed the prediction toward the target class,
        negative values pushed away from it.

    Raises
    ------
    TypeError
        If model does not inherit from BaseSequenceModel.
    ValueError
        If instance_tensor has wrong number of dimensions or if method is unsupported.

    Examples
    --------
    >>> from core.models.lstm import LSTMModel
    >>> from core.explain.captum_wrapper import explain_instance
    >>> import torch
    >>>
    >>> # Assume model is trained and instance is a patient's time series
    >>> instance = torch.randn(24, 40)  # 24 timesteps, 40 features
    >>> mask = torch.ones(24, dtype=torch.bool)
    >>> attributions = explain_instance(model, instance, mask, target_label=1)
    >>> print(f"Attribution shape: {attributions.shape}")
    """
    # Validate model contract
    if not isinstance(model, BaseSequenceModel):
        raise TypeError(
            f"Model must inherit from BaseSequenceModel, got {type(model).__name__}."
        )

    # Validate instance dimensions
    if instance_tensor.ndim != 2:
        raise ValueError(
            f"instance_tensor must be 2D (seq_len, n_features), "
            f"got shape {instance_tensor.shape}"
        )

    # Validate mask dimensions
    if mask.ndim != 1:
        raise ValueError(f"mask must be 1D (seq_len,), got shape {mask.shape}")

    if mask.shape[0] != instance_tensor.shape[0]:
        raise ValueError(
            f"mask length ({mask.shape[0]}) must match instance_tensor "
            f"sequence length ({instance_tensor.shape[0]})"
        )

    # Set model to evaluation mode and align inputs to the model device.
    model.eval()
    runtime_model = model
    runtime_device = next(model.parameters()).device
    instance_tensor = instance_tensor.to(runtime_device)
    mask = mask.to(runtime_device)

    # Add batch dimension (Captum expects batched inputs)
    instance_batched = instance_tensor.unsqueeze(0)  # (1, seq_len, n_features)
    mask_batched = mask.unsqueeze(0)  # (1, seq_len)

    # Create wrapper function for the model that handles mask
    def forward_func(inputs: Tensor) -> Tensor:
        """Wrapper to pass both inputs and mask to the model."""
        current_mask = mask_batched
        if current_mask.size(0) != inputs.size(0):
            current_mask = runtime_model.prepare_mask(current_mask, inputs.size(0))
        output: Tensor = runtime_model(inputs, current_mask)
        return output

    # Run a quick forward pass to validate target selection
    with torch.no_grad():
        sample_output = forward_func(instance_batched)

    if sample_output.ndim == 1:
        output_dim = 1
    elif sample_output.ndim >= 2:
        output_dim = sample_output.size(1)
    else:
        raise ValueError(
            "Model output must be at least 1D for explainability computations."
        )

    if output_dim == 1:
        if target_label not in (None, 0):
            raise ValueError(
                "Single-output models do not support explicit target "
                f"index {target_label}."
            )
        target_index: int | None = None
    else:
        if target_label is None:
            raise ValueError("target_label must be provided for multi-output models.")
        if target_label < 0 or target_label >= output_dim:
            raise ValueError(
                "target_label must be within the valid output range. "
                f"Got {target_label} for output dimension {output_dim}."
            )
        target_index = target_label

    # Initialize attribution method
    if method == "integrated_gradients":
        attributor = IntegratedGradients(forward_func)
    else:
        raise ValueError(
            f"Unsupported attribution method: {method}. "
            "Supported methods: 'integrated_gradients'"
        )

    # Compute attributions
    logger.info(
        f"Computing {method} attributions with {n_steps} steps "
        f"for target {target_index if target_index is not None else 'logit'}..."
    )

    # Create baseline (all zeros)
    baseline = torch.zeros_like(instance_batched)

    # Compute attributions
    attributions = attributor.attribute(
        instance_batched,
        baselines=baseline,
        target=target_index,
        n_steps=n_steps,
    )

    # Remove batch dimension and convert to numpy
    attributions_np: NDArray[np.float32] = (
        attributions.squeeze(0).detach().cpu().numpy()
    )

    logger.info(f"Computed attributions with shape {attributions_np.shape}")

    return attributions_np


def plot_feature_attributions(
    attributions: NDArray[np.float32],
    feature_names: list[str],
    top_k: int = 10,
    aggregate_method: Literal["mean", "max", "sum"] = "mean",
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    """
    Plot the most important features based on attribution values.

    Creates a bar chart showing which features contributed most to a prediction.
    Since attributions are computed per-timestep, we aggregate across time to
    get a single importance score per feature.

    Parameters
    ----------
    attributions : NDArray[np.float32]
        Feature attributions from explain_instance, shape (seq_len, n_features).
    feature_names : list[str]
        Names of the features, length must match n_features.
    top_k : int, default=10
        Number of top features to display in the plot.
    aggregate_method : Literal["mean", "max", "sum"], default="mean"
        How to aggregate attributions across timesteps:
        - "mean": Average attribution value across time
        - "max": Maximum absolute attribution across time
        - "sum": Sum of attributions across time
    figsize : tuple[int, int], default=(10, 8)
        Figure size as (width, height) in inches.

    Returns
    -------
    Figure
        Matplotlib Figure object containing the attribution bar chart.

    Raises
    ------
    ValueError
        If number of feature_names doesn't match attributions shape, or if
        aggregate_method is unsupported.

    Examples
    --------
    >>> from core.explain.captum_wrapper import plot_feature_attributions
    >>> import numpy as np
    >>>
    >>> attributions = np.random.randn(24, 40)  # 24 timesteps, 40 features
    >>> feature_names = [f"feature_{i}" for i in range(40)]
    >>> fig = plot_feature_attributions(attributions, feature_names, top_k=15)
    >>> fig.savefig('feature_importance.png')
    """
    # Validate inputs
    if attributions.ndim != 2:
        raise ValueError(
            f"attributions must be 2D (seq_len, n_features), "
            f"got shape {attributions.shape}"
        )

    n_features = attributions.shape[1]
    if len(feature_names) != n_features:
        raise ValueError(
            f"Number of feature_names ({len(feature_names)}) must match "
            f"number of features in attributions ({n_features})"
        )

    # Aggregate attributions across timesteps
    if aggregate_method == "mean":
        aggregated = np.mean(np.abs(attributions), axis=0)
    elif aggregate_method == "max":
        aggregated = np.max(np.abs(attributions), axis=0)
    elif aggregate_method == "sum":
        aggregated = np.sum(np.abs(attributions), axis=0)
    else:
        raise ValueError(
            f"Unsupported aggregate_method: {aggregate_method}. "
            "Supported methods: 'mean', 'max', 'sum'"
        )

    # Get top-k features
    top_indices = np.argsort(aggregated)[-top_k:][::-1]  # Descending order
    top_features = [feature_names[i] for i in top_indices]
    top_values = aggregated[top_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create bar chart
    import matplotlib as mpl

    viridis_cmap = mpl.colormaps.get_cmap("viridis")
    colors = viridis_cmap(np.linspace(0.3, 0.9, len(top_features)))
    ax.barh(range(len(top_features)), top_values, color=colors)

    # Formatting
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features)
    ax.set_xlabel("Attribution Magnitude (Absolute Value)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(
        f"Top {top_k} Most Important Features\n"
        f"({aggregate_method.capitalize()} absolute attribution across time)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Invert y-axis so most important is on top
    ax.invert_yaxis()

    plt.tight_layout()

    logger.info(f"Generated feature attribution plot for top {top_k} features")

    return fig
