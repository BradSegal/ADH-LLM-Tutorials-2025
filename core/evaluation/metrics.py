"""
Evaluation metrics for sequence models.

This module provides functions to generate predictions from trained models
and calculate classification performance metrics such as AUROC and AUPRC.
"""

import logging
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict, overload

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.models.base import BaseSequenceModel

logger = logging.getLogger(__name__)
Batch: TypeAlias = tuple[Tensor, Tensor, Tensor]


class CalibrationMetrics(TypedDict):
    """Typed dictionary for calibration metrics."""

    brier_score: float
    prob_true: NDArray[np.float32]
    prob_pred: NDArray[np.float32]


@dataclass(slots=True)
class PredictionResults:
    """Structured container returned by :func:`get_predictions`."""

    labels: Tensor
    probabilities: Tensor
    logits: Tensor


@overload
def get_predictions(
    model: BaseSequenceModel,
    loader: DataLoader[Batch],
    device: torch.device | str = "cpu",
    *,
    return_tensors: Literal[False] = False,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: ...


@overload
def get_predictions(
    model: BaseSequenceModel,
    loader: DataLoader[Batch],
    device: torch.device | str = "cpu",
    *,
    return_tensors: Literal[True],
) -> PredictionResults: ...


def get_predictions(
    model: BaseSequenceModel,
    loader: DataLoader[Batch],
    device: torch.device | str = "cpu",
    *,
    return_tensors: bool = False,
) -> (
    tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
    | PredictionResults
):
    """
    Generate predictions from a trained model on a dataset.

    Runs the model in evaluation mode (no gradient computation) and collects
    all labels, predicted probabilities, and raw logits across all batches.

    Parameters
    ----------
    model : BaseSequenceModel
        The trained sequence model to evaluate. Must inherit from
        BaseSequenceModel and implement the forward(x, mask) method.
    loader : DataLoader
        DataLoader providing batches of (features, mask, labels).
    device : torch.device | str, default="cpu"
        Device to run inference on. Can be "cpu", "cuda", or a torch.device.
    return_tensors : bool, default=False
        When True, return a :class:`PredictionResults` object with torch tensors
        (on CPU). When False, return numpy arrays for compatibility with
        existing notebook code.

    Returns
    -------
    tuple[NDArray[np.float32], ...] | PredictionResults
        Either a tuple of numpy arrays or, when ``return_tensors`` is True,
        a :class:`PredictionResults` object containing torch tensors on CPU.

    Raises
    ------
    TypeError
        If model does not inherit from BaseSequenceModel.
    ValueError
        If loader contains no batches.

    Examples
    --------
    >>> from core.models.lstm import LSTMModel
    >>> from core.evaluation.metrics import get_predictions
    >>>
    >>> # Assume model and test_loader are defined
    >>> labels, preds, logits = get_predictions(model, test_loader, device="cuda")
    >>> print(f"Generated {len(labels)} predictions")
    """
    # Validate model contract
    if not isinstance(model, BaseSequenceModel):
        raise TypeError(
            f"Model must inherit from BaseSequenceModel, got {type(model).__name__}."
        )

    # Convert device string to torch.device if needed
    if isinstance(device, str):
        device = torch.device(device)

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()

    all_labels: list[Tensor] = []
    all_logits: list[Tensor] = []

    logger.info("Generating predictions...")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            features, mask, labels = batch

            # Move batch to device
            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(features, mask)

            # Collect results (move to CPU and convert to numpy)
            all_labels.append(labels.detach().cpu())
            all_logits.append(logits.detach().cpu())

    # Concatenate all batches
    concatenated_labels_t = torch.cat(all_labels).to(torch.float32)
    concatenated_logits_t = torch.cat(all_logits).to(torch.float32).squeeze()
    concatenated_labels_t = concatenated_labels_t.squeeze()
    concatenated_predictions_t = torch.sigmoid(concatenated_logits_t)

    logger.info("Generated %d predictions", concatenated_labels_t.shape[0])

    if return_tensors:
        return PredictionResults(
            labels=concatenated_labels_t,
            probabilities=concatenated_predictions_t,
            logits=concatenated_logits_t,
        )

    return (
        concatenated_labels_t.numpy(),
        concatenated_predictions_t.numpy(),
        concatenated_logits_t.numpy(),
    )


def calculate_classification_metrics(
    labels: NDArray[np.float32], predictions: NDArray[np.float32]
) -> dict[str, float]:
    """
    Calculate binary classification metrics.

    Computes AUROC (Area Under ROC Curve) and AUPRC (Area Under Precision-Recall
    Curve) for binary classification tasks. These metrics are particularly useful
    for imbalanced datasets like sepsis prediction.

    Parameters
    ----------
    labels : NDArray[np.float32]
        Ground truth binary labels, shape (N,). Values should be 0 or 1.
    predictions : NDArray[np.float32]
        Predicted probabilities, shape (N,). Values should be in range [0, 1].

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - 'auroc': Area under the ROC curve
        - 'auprc': Area under the precision-recall curve

    Raises
    ------
    ValueError
        If labels and predictions have different lengths, or if labels contain
        only one class (metrics require both positive and negative examples).

    Examples
    --------
    >>> from core.evaluation.metrics import calculate_classification_metrics
    >>> import numpy as np
    >>>
    >>> labels = np.array([0, 0, 1, 1])
    >>> predictions = np.array([0.1, 0.3, 0.8, 0.9])
    >>> metrics = calculate_classification_metrics(labels, predictions)
    >>> print(f"AUROC: {metrics['auroc']:.3f}")
    >>> print(f"AUPRC: {metrics['auprc']:.3f}")
    """
    # Validate inputs
    if len(labels) != len(predictions):
        raise ValueError(
            f"labels and predictions must have the same length. "
            f"Got {len(labels)} labels and {len(predictions)} predictions."
        )

    if len(labels) == 0:
        raise ValueError("labels and predictions must not be empty.")

    # Check that we have both classes present
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError(
            f"Labels must contain both classes (0 and 1) to calculate metrics. "
            f"Found only class(es): {unique_labels}"
        )

    # Calculate metrics
    auroc = roc_auc_score(labels, predictions)
    auprc = average_precision_score(labels, predictions)

    logger.info(f"Calculated metrics: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

    return {"auroc": auroc, "auprc": auprc}


def calculate_calibration_metrics(
    labels: NDArray[np.float32], predictions: NDArray[np.float32], n_bins: int = 10
) -> CalibrationMetrics:
    """
    Calculate calibration metrics for probabilistic predictions.

    Computes the Brier score and calibration curve data (reliability diagram)
    to assess whether predicted probabilities match true event frequencies.
    A well-calibrated model's predicted probability of 0.8 should correspond
    to an 80% actual event rate.

    Parameters
    ----------
    labels : NDArray[np.float32]
        Ground truth binary labels, shape (N,). Values should be 0 or 1.
    predictions : NDArray[np.float32]
        Predicted probabilities, shape (N,). Values should be in range [0, 1].
    n_bins : int, default=10
        Number of bins to use for the calibration curve. More bins provide
        finer granularity but require more data for stable estimates.

    Returns
    -------
    CalibrationMetrics
        TypedDict containing:
        - 'brier_score': Brier score (lower is better, range [0, 1])
        - 'prob_true': True probability in each bin (mean label value)
        - 'prob_pred': Mean predicted probability in each bin

    Raises
    ------
    ValueError
        If labels and predictions have different lengths, are empty,
        or if n_bins is less than 2.

    Examples
    --------
    >>> from core.evaluation.metrics import calculate_calibration_metrics
    >>> import numpy as np
    >>>
    >>> labels = np.array([0, 0, 1, 1, 1])
    >>> predictions = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
    >>> metrics = calculate_calibration_metrics(labels, predictions)
    >>> print(f"Brier Score: {metrics['brier_score']:.3f}")
    """
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss

    # Validate inputs
    if len(labels) != len(predictions):
        raise ValueError(
            f"labels and predictions must have the same length. "
            f"Got {len(labels)} labels and {len(predictions)} predictions."
        )

    if len(labels) == 0:
        raise ValueError("labels and predictions must not be empty.")

    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}.")

    # Calculate Brier score (mean squared error of probabilities)
    brier_score = brier_score_loss(labels, predictions)

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(
        labels, predictions, n_bins=n_bins, strategy="uniform"
    )

    logger.info(f"Calculated calibration metrics: Brier score={brier_score:.4f}")

    return {
        "brier_score": float(brier_score),
        "prob_true": prob_true,
        "prob_pred": prob_pred,
    }


def calculate_subgroup_metrics(
    df: pd.DataFrame,
    labels: NDArray[np.float32],
    predictions: NDArray[np.float32],
    subgroup_col: str,
) -> dict[str, float]:
    """
    Calculate performance metrics for each subgroup in the dataset.

    Splits the data by a demographic or clinical subgroup column and computes
    AUROC for each subgroup independently. This is essential for detecting
    performance disparities that could lead to unfair outcomes in clinical
    deployment.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the subgroup column. Must have the same number
        of rows as labels and predictions.
    labels : NDArray[np.float32]
        Ground truth binary labels, shape (N,). Values should be 0 or 1.
    predictions : NDArray[np.float32]
        Predicted probabilities, shape (N,). Values should be in range [0, 1].
    subgroup_col : str
        Name of the column in df to use for grouping (e.g., 'age_group', 'gender').

    Returns
    -------
    dict[str, float]
        Dictionary mapping subgroup values to their AUROC scores.
        For example: {'0-40': 0.85, '41-60': 0.78, '61-80': 0.72}

    Raises
    ------
    ValueError
        If df, labels, and predictions have different lengths, if subgroup_col
        is not in df, if all subgroup values are NaN, if any subgroup has fewer
        than 2 samples, or if any subgroup contains only one class.
    KeyError
        If subgroup_col does not exist in the DataFrame.

    Examples
    --------
    >>> from core.evaluation.metrics import calculate_subgroup_metrics
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> df = pd.DataFrame({'age_group': ['young', 'young', 'old', 'old']})
    >>> labels = np.array([0, 1, 0, 1], dtype=np.float32)
    >>> predictions = np.array([0.2, 0.8, 0.3, 0.9], dtype=np.float32)
    >>> metrics = calculate_subgroup_metrics(df, labels, predictions, 'age_group')
    >>> print(metrics)
    """
    # Validate inputs
    if len(df) != len(labels) or len(df) != len(predictions):
        raise ValueError(
            f"df, labels, and predictions must have the same length. "
            f"Got {len(df)} rows in df, {len(labels)} labels, and "
            f"{len(predictions)} predictions."
        )

    if subgroup_col not in df.columns:
        raise KeyError(
            f"Column '{subgroup_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Create a temporary DataFrame for grouping
    temp_df = pd.DataFrame(
        {
            "subgroup": df[subgroup_col],
            "label": labels,
            "prediction": predictions,
        }
    )

    before_drop = len(temp_df)
    temp_df = temp_df.dropna(subset=["subgroup"])
    dropped = before_drop - len(temp_df)

    if dropped > 0:
        logger.warning(
            "Dropped %d rows with NaN subgroup values for column '%s'.",
            dropped,
            subgroup_col,
        )

    if temp_df.empty:
        raise ValueError(
            f"All rows have NaN subgroup values for column '{subgroup_col}'."
        )

    # Calculate AUROC for each subgroup
    subgroup_metrics: dict[str, float] = {}

    for subgroup_value, group_df in temp_df.groupby(
        "subgroup", sort=False, observed=False
    ):
        group_labels = group_df["label"].values
        group_predictions = group_df["prediction"].values

        # Validate subgroup has enough samples
        if len(group_labels) < 2:
            logger.warning(
                f"Subgroup '{subgroup_value}' has only {len(group_labels)} sample(s). "
                f"Skipping AUROC calculation (requires at least 2 samples)."
            )
            continue

        # Check that we have both classes in this subgroup
        unique_labels = np.unique(group_labels)
        if len(unique_labels) < 2:
            logger.warning(
                f"Subgroup '{subgroup_value}' contains only class {unique_labels[0]}. "
                f"Skipping AUROC calculation (requires both classes)."
            )
            continue

        # Calculate AUROC for this subgroup
        auroc = roc_auc_score(group_labels, group_predictions)
        subgroup_metrics[str(subgroup_value)] = float(auroc)

        logger.info(f"Subgroup '{subgroup_value}': AUROC={auroc:.4f}")

    if not subgroup_metrics:
        raise ValueError(
            f"No valid subgroups found for column '{subgroup_col}'. "
            f"All subgroups either had too few samples or only one class."
        )

    return subgroup_metrics
