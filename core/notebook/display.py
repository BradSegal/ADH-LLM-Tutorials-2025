"""Display utilities for notebook visualizations and hyperparameter presentation.

This module provides functions to create rich, informative displays of training
progress and model configurations in the tutorial notebooks.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.config import GRUConfig, LSTMConfig, TrainConfig, TransformerConfig


def display_hyperparameter_table(
    model_config: GRUConfig | LSTMConfig | TransformerConfig,
    train_config: TrainConfig,
    model_type: str,
) -> None:
    """Display hyperparameters in a formatted, annotated table.

    Args:
        model_config: Model configuration (GRUConfig, LSTMConfig, or TransformerConfig)
        train_config: Training configuration (TrainConfig)
        model_type: "GRU", "LSTM", or "Transformer"
    """
    # Define parameter metadata (descriptions, modification safety)
    param_info = {
        # Common model parameters
        "input_size": {
            "category": "Data",
            "description": "Number of input features (from dataset)",
            "modifiable": "❌ No - Fixed by data",
        },
        "hidden_size": {
            "category": "Architecture",
            "description": "Size of hidden state vector",
            "modifiable": "⚠️ Yes (try 32-256)",
        },
        "num_layers": {
            "category": "Architecture",
            "description": "Number of stacked layers",
            "modifiable": "⚠️ Yes (try 1-3)",
        },
        "num_encoder_layers": {
            "category": "Architecture",
            "description": "Number of Transformer encoder layers",
            "modifiable": "⚠️ Yes (try 1-3)",
        },
        "dropout": {
            "category": "Regularization",
            "description": "Dropout probability between layers",
            "modifiable": "✅ Yes (try 0.1-0.5)",
        },
        # Transformer-specific
        "nhead": {
            "category": "Architecture",
            "description": "Number of attention heads",
            "modifiable": "⚠️ Yes (must divide input_size)",
        },
        "dim_feedforward": {
            "category": "Architecture",
            "description": "Feedforward network dimension",
            "modifiable": "⚠️ Yes (try 128-512)",
        },
        "max_seq_length": {
            "category": "Architecture",
            "description": "Maximum sequence length supported",
            "modifiable": "⚠️ Yes (must exceed longest patient stay)",
        },
        # Training parameters
        "learning_rate": {
            "category": "Optimization",
            "description": "Step size for weight updates",
            "modifiable": "✅ Yes (try 0.0001-0.01)",
        },
        "batch_size": {
            "category": "Training",
            "description": "Samples per gradient update",
            "modifiable": "⚠️ Yes (32, 64, 128)",
        },
        "epochs": {
            "category": "Training",
            "description": "Complete passes through data",
            "modifiable": "✅ Yes",
        },
        "optimizer": {
            "category": "Optimization",
            "description": "Optimization algorithm",
            "modifiable": "✅ Yes (adam or sgd)",
        },
        "max_grad_norm": {
            "category": "Optimization",
            "description": "Gradient clipping threshold",
            "modifiable": "✅ Yes (try 0.5-5.0)",
        },
        "train_val_split": {
            "category": "Data",
            "description": "Train/validation split ratio",
            "modifiable": "❌ No - Keep for fair comparison",
        },
        "split_seed": {
            "category": "Data",
            "description": "Random seed for reproducibility",
            "modifiable": "❌ No - Keep at 42",
        },
    }

    # Build table rows
    rows = []

    # Model parameters
    for param, value in model_config.model_dump().items():
        if param in param_info:
            info = param_info[param]
            rows.append(
                {
                    "Parameter": param,
                    "Value": value,
                    "Category": info["category"],
                    "Description": info["description"],
                    "Modifiable?": info["modifiable"],
                }
            )

    # Key training parameters
    important_train_params = [
        "learning_rate",
        "batch_size",
        "epochs",
        "optimizer",
        "max_grad_norm",
        "train_val_split",
        "split_seed",
    ]
    for param in important_train_params:
        if hasattr(train_config, param):
            value = getattr(train_config, param)
            if param in param_info:
                info = param_info[param]
                rows.append(
                    {
                        "Parameter": param,
                        "Value": value,
                        "Category": info["category"],
                        "Description": info["description"],
                        "Modifiable?": info["modifiable"],
                    }
                )

    # Create DataFrame and display
    df = pd.DataFrame(rows)

    print(f"\n{'=' * 90}")
    print(f"{model_type} HYPERPARAMETER CONFIGURATION".center(90))
    print(f"{'=' * 90}\n")

    # Display with nice formatting
    print(
        df.to_string(
            index=False, max_colwidth=50, col_space=None, justify="left", header=True
        )
    )

    print(f"\n{'=' * 90}")
    print("\nModification Guide:")
    print("  ✅ Safe to experiment - good learning opportunity")
    print("  ⚠️  Moderate impact - understand implications first")
    print("  ❌ Do not modify - determined by data/architecture constraints")
    print(
        f"{'=' * 90}\n"
    )


def plot_training_dashboard(
    history: dict[str, Any], model_name: str = "Model", save_path: str | None = None
) -> None:
    """Create a comprehensive training visualization dashboard.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', and optionally 'val_metrics'
        model_name: Name of the model for plot titles
        save_path: Optional path where the best model was saved
    """
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    epochs = range(1, len(history["train_loss"]) + 1)
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    # Plot 1: Loss curves with best model marker
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, "b-", label="Training Loss", linewidth=2, alpha=0.8)
    ax1.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2, alpha=0.8)

    # Mark best model
    best_epoch = np.argmin(val_loss) + 1
    best_loss = min(val_loss)
    ax1.axvline(
        best_epoch,
        color="green",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label=f"Best Model (Epoch {best_epoch})",
    )
    ax1.plot(best_epoch, best_loss, "g*", markersize=20, markeredgecolor="darkgreen")

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss (Binary Cross-Entropy)", fontsize=11)
    ax1.set_title(f"{model_name}: Training Progress", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation metrics (if available)
    ax2 = fig.add_subplot(gs[0, 1])
    if "val_metrics" in history and history["val_metrics"]:
        for metric_name, values in history["val_metrics"].items():
            ax2.plot(
                epochs, values, marker="o", label=metric_name.upper(), linewidth=2
            )
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Score", fontsize=11)
        ax2.set_title("Validation Metrics Over Time", fontsize=13, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        # Mark best epoch for metrics too
        ax2.axvline(
            best_epoch, color="green", linestyle="--", alpha=0.5, linewidth=2
        )
    else:
        # No metrics available, show message
        ax2.text(
            0.5,
            0.5,
            "Validation metrics not enabled.\n\n"
            "To enable, add to config YAML:\n"
            "  val_metrics: ['auprc', 'auroc']",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax2.transAxes,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
        ax2.set_title(
            "Validation Metrics (Not Enabled)", fontsize=13, fontweight="bold"
        )
        ax2.axis("off")

    # Plot 3: Train-Val gap (overfitting indicator)
    ax3 = fig.add_subplot(gs[1, 0])
    gap = np.array(val_loss) - np.array(train_loss)
    ax3.plot(epochs, gap, "purple", linewidth=2, label="Val - Train Loss")
    ax3.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax3.fill_between(
        epochs,
        0,
        gap,
        where=(gap >= 0),
        alpha=0.3,
        color="red",
        label="Overfitting Risk",
    )
    ax3.fill_between(
        epochs,
        gap,
        0,
        where=(gap < 0),
        alpha=0.3,
        color="green",
        label="Good Generalization",
    )
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Validation Loss - Training Loss", fontsize=11)
    ax3.set_title("Generalization Gap Analysis", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Mark best epoch
    ax3.axvline(best_epoch, color="green", linestyle="--", alpha=0.5, linewidth=2)

    # Plot 4: Summary statistics (text)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    improvement_pct = (val_loss[0] - best_loss) / val_loss[0] * 100

    summary_text = (
        f"TRAINING SUMMARY\n"
        f"{'=' * 45}\n\n"
        f"Best Epoch:       {best_epoch:>3} / {len(epochs)}\n"
        f"Best Val Loss:    {best_loss:>8.4f}\n"
        f"Final Train Loss: {train_loss[-1]:>8.4f}\n"
        f"Final Val Loss:   {val_loss[-1]:>8.4f}\n\n"
        f"Improvement:      {improvement_pct:>7.1f}%\n"
        f"(from epoch 1 to best)\n\n"
    )

    if "val_metrics" in history and history["val_metrics"]:
        summary_text += "Best Validation Metrics:\n"
        summary_text += "-" * 45 + "\n"
        for metric, values in history["val_metrics"].items():
            best_metric = max(values)
            best_metric_epoch = np.argmax(values) + 1
            summary_text += (
                f"  {metric.upper():<8} {best_metric:>6.4f}  (epoch {best_metric_epoch})\n"
            )

    if save_path:
        summary_text += f"\n{'=' * 45}\n"
        summary_text += f"Model saved to:\n{save_path}"

    ax4.text(
        0.1,
        0.5,
        summary_text,
        fontsize=10,
        family="monospace",
        verticalalignment="center",
    )

    plt.suptitle(
        f"{model_name} Training Dashboard", fontsize=15, fontweight="bold", y=0.98
    )
    plt.show()


def print_pre_training_summary(
    model: Any, train_loader: Any, val_loader: Any, train_config: TrainConfig
) -> None:
    """Print a pre-training checkpoint summary.

    Args:
        model: The model to be trained
        train_loader: Training data loader
        val_loader: Validation data loader
        train_config: Training configuration
    """
    print("=" * 70)
    print("PRE-TRAINING CHECKPOINT".center(70))
    print("=" * 70)
    print(f"\nModel:            {model.__class__.__name__}")
    print(f"Device:           {train_config.device}")
    print(f"Epochs:           {train_config.epochs}")
    print(f"Batch Size:       {train_config.batch_size}")
    print(f"Learning Rate:    {train_config.learning_rate}")
    print(f"Optimizer:        {train_config.optimizer}")
    print(f"Max Grad Norm:    {train_config.max_grad_norm}")

    print("\n" + "-" * 70)
    print("DATASET INFORMATION".center(70))
    print("-" * 70)
    print(f"\nTraining Samples:   {len(train_loader.dataset):>6,}")
    print(f"Validation Samples: {len(val_loader.dataset):>6,}")
    print(f"Batches per Epoch:  {len(train_loader):>6,}")

    # Estimate training time
    estimated_time_per_epoch = len(train_loader) * 0.1  # Rough estimate: 0.1s per batch
    total_estimated_time = estimated_time_per_epoch * train_config.epochs
    print(
        f"\nEstimated Time per Epoch: ~{estimated_time_per_epoch:.0f} seconds"
    )
    print(f"Total Estimated Time:     ~{total_estimated_time / 60:.1f} minutes")
    print("(Actual time depends on hardware; GPU is much faster)")

    print("\n" + "=" * 70)
    print("✅ Ready to train! Run the next cell to begin.".center(70))
    print("=" * 70 + "\n")
