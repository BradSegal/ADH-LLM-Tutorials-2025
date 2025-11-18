"""
Unified, model-agnostic training and evaluation harness.

This module implements the core Trainer class that encapsulates the complete
training loop. By abstracting away the complexity of PyTorch training, this
enables notebooks to focus on pedagogy while ensuring a consistent, correct
training process across all models.
"""

import logging
from collections.abc import Collection
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config.models import TrainConfig
from core.models.base import BaseSequenceModel

logger = logging.getLogger(__name__)
Batch: TypeAlias = tuple[Tensor, Tensor, Tensor]


class Trainer:
    """
    A unified, model-agnostic training and evaluation harness.

    This class implements the complete PyTorch training loop, including device
    management, epoch iteration, loss calculation, backpropagation, and metrics
    collection. It works with any model that inherits from BaseSequenceModel,
    enabling the DRY principle by implementing training logic once.

    The Trainer follows the "fail fast, fail loudly" principle by validating
    all components during initialization and using Pydantic-validated config.

    Parameters
    ----------
    model : BaseSequenceModel
        The sequence model to train. Must inherit from BaseSequenceModel and
        implement the forward(x, mask) method.
    train_loader : DataLoader
        DataLoader for training data. Each batch should yield a tuple of
        (features, mask, labels).
    val_loader : DataLoader
        DataLoader for validation data. Same format as train_loader.
    config : TrainConfig
        Validated training configuration containing learning_rate, epochs,
        batch_size, and device preferences.
    save_path : Path | str | None, optional
        Path where the best model checkpoint will be saved. If None, model
        checkpoints will not be saved. Default is None.

    Attributes
    ----------
    model : BaseSequenceModel
        The model being trained.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    config : TrainConfig
        Validated training configuration.
    device : torch.device
        The device (CPU or CUDA) where training will occur.
    optimizer : torch.optim.Optimizer
        The optimizer used for training (Adam by default).
    loss_fn : nn.Module
        The loss function (BCEWithLogitsLoss for binary classification).
    save_path : Path | None
        Path where the best model checkpoint will be saved, or None if saving
        is disabled.

    Examples
    --------
    >>> from core.config.models import TrainConfig
    >>> from core.models.base import BaseSequenceModel
    >>> from core.train.trainer import Trainer
    >>>
    >>> # Assume model, train_loader, val_loader are already defined
    >>> config = TrainConfig(learning_rate=0.001, epochs=10, batch_size=32)
    >>> trainer = Trainer(model, train_loader, val_loader, config)
    >>> history, best_path = trainer.fit()
    >>> print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    >>> print(f"Best model saved to: {best_path}")
    """

    def __init__(
        self,
        model: BaseSequenceModel,
        train_loader: DataLoader[Batch],
        val_loader: DataLoader[Batch],
        config: TrainConfig,
        save_path: Path | str | None = None,
    ) -> None:
        """
        Initialize the Trainer with model, data, and configuration.

        Performs validation of inputs and sets up the device, optimizer, and
        loss function.

        Raises
        ------
        TypeError
            If model does not inherit from BaseSequenceModel.
        """
        # Validate that model follows the contract
        if not isinstance(model, BaseSequenceModel):
            raise TypeError(
                f"Model must inherit from BaseSequenceModel, "
                f"got {type(model).__name__}. "
                "This ensures the model has the required forward(x, mask) interface."
            )

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self._metric_set = set(config.val_metrics)

        # Convert save_path to Path if provided
        self.save_path: Path | None = Path(save_path) if save_path is not None else None

        # Ensure parent directory exists if save_path is provided
        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Best model will be saved to: {self.save_path}")

        # Determine actual device (may fall back to CPU if CUDA unavailable)
        if self.config.device == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but not available. Falling back to CPU. "
                "Training will be slower."
            )
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

        logger.info(f"Trainer initialized with device: {self.device}")

        # Move model to device
        self.model.to(self.device)

        # Validate data loaders and store their lengths for averaging
        self._train_batches = self._validate_loader(self.train_loader, "train")
        self._val_batches = self._validate_loader(self.val_loader, "validation")

        # Initialize optimizer, loss function, and scheduler from config
        self.optimizer = self._build_optimizer()
        self.loss_fn: nn.Module = self._build_loss_fn()
        self.scheduler = self._build_scheduler()

    def _train_epoch(self) -> float:
        """
        Run a single training epoch.

        Iterates through the training data, computes loss, performs
        backpropagation, and updates model weights.

        Returns
        -------
        float
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            features, mask, labels = self._prepare_batch(batch, "train")

            self.optimizer.zero_grad()
            logits = self.model(features, mask)
            loss = self.loss_fn(logits, labels)

            loss.backward()
            self._maybe_clip_gradients()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / self._train_batches

    def _evaluate_epoch(self) -> tuple[float, dict[str, float]]:
        """
        Run a single validation epoch.

        Iterates through the validation data and computes loss without
        gradient calculations or weight updates.

        Returns
        -------
        tuple[float, dict[str, float]]
            Average validation loss and the configured validation metrics.
        """
        gather_metrics = bool(self._metric_set)
        val_loss, labels_list, probs_list = self._collect_validation_outputs(
            gather_probs=gather_metrics,
            desc="Validation",
        )

        metrics: dict[str, float] = {}
        if gather_metrics:
            metrics = self._compute_metric_values(
                labels_list,
                probs_list,
                requested_metrics=self._metric_set,
            )

        return val_loss, metrics

    def fit(self) -> tuple[dict[str, Any], Path | None]:
        """
        Run the full training and evaluation loop.

        Executes the complete training process for the configured number of
        epochs, tracking both training and validation loss at each epoch.
        If save_path was provided during initialization, saves the model with
        the best validation loss.

        This is the main public interface of the Trainer.

        Returns
        -------
        tuple[dict[str, Any], Path | None]
            A tuple containing:
            - history: Dictionary with 'train_loss', 'val_loss', and optional
              'val_metrics' entries (per TrainConfig.val_metrics)
            - best_model_path: Path to the saved best model, or None if saving
              was disabled

        Examples
        --------
        >>> trainer = Trainer(
        ...     model, train_loader, val_loader, config, save_path="best.pt"
        ... )
        >>> history, best_path = trainer.fit()
        >>> print(f"Trained for {len(history['train_loss'])} epochs")
        >>> print(f"Best validation loss: {min(history['val_loss']):.4f}")
        >>> print(f"Best model saved at: {best_path}")
        """
        logger.info("Starting training on device: %s", self.device)
        history: dict[str, Any] = {"train_loss": [], "val_loss": []}
        if self._metric_set:
            history["val_metrics"] = {metric: [] for metric in self._metric_set}
        best_val_loss = float("inf")
        best_model_path: Path | None = None

        for epoch in range(self.config.epochs):
            train_loss = self._train_epoch()
            val_loss, metrics = self._evaluate_epoch()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if self._metric_set:
                for metric, value in metrics.items():
                    history["val_metrics"][metric].append(value)

            logger.info(
                "Epoch %s/%s | Train Loss: %.4f | Val Loss: %.4f",
                epoch + 1,
                self.config.epochs,
                train_loss,
                val_loss,
            )
            if metrics:
                metric_summary = " | ".join(
                    f"{name.upper()}: {value:.4f}" for name, value in metrics.items()
                )
                logger.info("Validation metrics -> %s", metric_summary)

            # Save model if this is the best validation loss so far
            if self.save_path is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = self.save_path
                torch.save(self.model.state_dict(), self.save_path)
                logger.info(
                    "New best model saved with val_loss=%.4f at %s",
                    val_loss,
                    self.save_path,
                )

            if self.scheduler is not None:
                self.scheduler.step()

        logger.info("Training completed successfully")
        if best_model_path is not None:
            logger.info(f"Best model available at: {best_model_path}")

        return history, best_model_path

    def _validate_loader(self, loader: DataLoader[Batch], name: str) -> int:
        """
        Validate loader against config contract and return its length.

        Raises
        ------
        ValueError
            If batch size mismatches config or loader has no batches.
        TypeError
            If loader does not implement __len__ (required for averaging).
        """
        batch_size = getattr(loader, "batch_size", None)
        if batch_size is None:
            raise ValueError(
                f"{name} loader must define batch_size to validate against TrainConfig."
            )
        if batch_size != self.config.batch_size:
            raise ValueError(
                f"{name} loader batch_size ({batch_size}) does not match "
                f"TrainConfig.batch_size ({self.config.batch_size}). "
                "Please construct your DataLoader using the config."
            )

        try:
            length = len(loader)
        except TypeError as exc:
            raise TypeError(
                f"{name} loader must be sized (implement __len__) for the Trainer to "
                "compute average losses."
            ) from exc

        if length <= 0:
            raise ValueError(f"{name} loader must contain at least one batch.")

        return length

    def _prepare_batch(self, batch: Batch, loader_name: str) -> Batch:
        """Validate, unpack, and move a batch to the configured device."""
        features, mask, labels = self._unpack_batch(batch, loader_name)
        return (
            features.to(self.device),
            mask.to(self.device),
            labels.to(self.device),
        )

    def _unpack_batch(self, batch: Batch, loader_name: str) -> Batch:
        """Ensure the batch follows the (features, mask, labels) contract."""
        if not isinstance(batch, (tuple, list)):
            raise ValueError(
                f"{loader_name} loader must return a tuple of "
                f"(features, mask, labels). Got {type(batch)} instead."
            )
        if len(batch) != 3:
            raise ValueError(
                f"{loader_name} loader must return exactly three tensors. "
                f"Got {len(batch)} items."
            )

        features, mask, labels = batch

        if not isinstance(features, Tensor) or features.ndim != 3:
            raise ValueError(
                f"{loader_name} batch features must be a 3D Tensor "
                "[batch, seq_len, features]."
            )
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
            raise ValueError(
                f"{loader_name} batch mask must be a boolean Tensor "
                f"with dtype torch.bool."
            )
        if mask.shape[0] != features.shape[0] or mask.shape[1] != features.shape[1]:
            raise ValueError(
                f"{loader_name} batch mask shape {mask.shape} must match the first two "
                f"dimensions of features {features.shape}."
            )
        if not isinstance(labels, Tensor):
            raise ValueError(
                f"{loader_name} batch labels must be a Tensor. Got {type(labels)}."
            )
        if labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"{loader_name} batch labels must have the same batch dimension as "
                "features."
            )

        return features, mask, labels

    def _build_optimizer(self) -> Optimizer:
        """Create the optimizer specified in the TrainConfig."""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
            )
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.optimizer_momentum,
            )

        raise ValueError(f"Unsupported optimizer '{self.config.optimizer}'.")

    def _build_loss_fn(self) -> nn.Module:
        """Create the loss function specified in the TrainConfig."""
        if self.config.loss == "bce_logits":
            return nn.BCEWithLogitsLoss()
        if self.config.loss == "cross_entropy":
            return nn.CrossEntropyLoss()
        if self.config.loss == "mse":
            return nn.MSELoss()

        raise ValueError(f"Unsupported loss '{self.config.loss}'.")

    def _build_scheduler(self) -> LRScheduler | None:
        """Create the configured learning rate scheduler, if any."""
        if self.config.scheduler == "none":
            return None
        if self.config.scheduler == "step":
            if (
                self.config.scheduler_step_size is None
                or self.config.scheduler_gamma is None
            ):
                raise ValueError(
                    "scheduler_step_size and scheduler_gamma must be provided when "
                    "scheduler='step'."
                )
            return StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma,
            )

        raise ValueError(f"Unsupported scheduler '{self.config.scheduler}'.")

    def _maybe_clip_gradients(self) -> None:
        """Apply gradient clipping if configured."""
        if self.config.max_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

    def compute_val_metric(self, metric: str = "auprc") -> float:
        """
        Compute a classification metric on the validation set.

        This method is primarily used for hyperparameter optimization with Optuna,
        where we need to return a single scalar metric for each trial. It runs
        inference on the entire validation set and computes the requested metric.

        Parameters
        ----------
        metric : str, default="auprc"
            The metric to compute. Supported values:
            - "auprc": Area Under Precision-Recall Curve
            - "auroc": Area Under ROC Curve

        Returns
        -------
        float
            The computed metric value.

        Raises
        ------
        ValueError
            If an unsupported metric is requested.

        Examples
        --------
        >>> trainer = Trainer(model, train_loader, val_loader, config)
        >>> history, _ = trainer.fit()
        >>> final_auprc = trainer.compute_val_metric("auprc")
        >>> print(f"Final validation AUPRC: {final_auprc:.4f}")
        """
        if metric not in {"auprc", "auroc"}:
            raise ValueError(
                f"Unsupported metric '{metric}'. Must be one of: 'auprc', 'auroc'"
            )

        was_training = self.model.training
        try:
            _, labels_list, probs_list = self._collect_validation_outputs(
                gather_probs=True,
                desc=f"Computing {metric.upper()}",
            )
        finally:
            if was_training:
                self.model.train()
            else:
                self.model.eval()

        metrics = self._compute_metric_values(
            labels_list,
            probs_list,
            requested_metrics={metric},
        )
        score = metrics[metric]
        logger.info(f"Validation {metric.upper()}: {score:.4f}")
        return score

    def _collect_validation_outputs(
        self,
        *,
        gather_probs: bool,
        desc: str,
    ) -> tuple[float, list[Tensor], list[Tensor]]:
        """Iterate through validation loader and optionally cache probabilities."""
        self.model.eval()
        total_loss = 0.0
        labels_list: list[Tensor] = []
        probs_list: list[Tensor] = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=desc, leave=False):
                features, mask, labels = self._prepare_batch(batch, "validation")
                logits = self.model(features, mask)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

                if gather_probs:
                    probs = torch.sigmoid(logits)
                    labels_list.append(labels.detach().cpu())
                    probs_list.append(probs.detach().cpu())

        average_loss = total_loss / self._val_batches
        return average_loss, labels_list, probs_list

    def _compute_metric_values(
        self,
        labels_list: list[Tensor],
        probs_list: list[Tensor],
        *,
        requested_metrics: Collection[str],
    ) -> dict[str, float]:
        """Compute validation metrics from cached tensors."""
        requested = set(requested_metrics)
        if not requested:
            return {}

        from sklearn.metrics import average_precision_score, roc_auc_score

        labels_tensor = torch.cat(labels_list).squeeze()
        probs_tensor = torch.cat(probs_list).squeeze()
        labels_np: NDArray[np.float32] = labels_tensor.numpy()
        probs_np: NDArray[np.float32] = probs_tensor.numpy()

        metrics: dict[str, float] = {}
        if "auprc" in requested:
            metrics["auprc"] = float(average_precision_score(labels_np, probs_np))
        if "auroc" in requested:
            metrics["auroc"] = float(roc_auc_score(labels_np, probs_np))
        return metrics
