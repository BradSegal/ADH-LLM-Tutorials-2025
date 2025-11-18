## Training Engine Guide

This document explains how the reusable training stack works so notebook authors and contributors can rely on it without re-implementing PyTorch boilerplate.

### 1. Components

- **`TrainConfig` (`core.config.TrainConfig`)** – a Pydantic model that validates every hyperparameter before training begins. Any invalid combination (negative learning rates, unsupported optimizers, missing scheduler params, etc.) raises immediately.
- **`Trainer` (`core.train.Trainer`)** – the unified training/evaluation harness. It consumes a validated `TrainConfig`, enforces strict contracts on the incoming `DataLoader`s, manages device placement, and executes the training loop with consistent logging and metrics.
- **`BaseSequenceModel`** – the abstract interface all sequence models must implement (`forward(features, mask)`), ensuring the trainer can operate on any concrete model.

### 2. Creating a `TrainConfig`

```python
from core.config import TrainConfig

train_config = TrainConfig(
    learning_rate=1e-3,
    epochs=10,
    batch_size=64,
    device="cuda",
    optimizer="adam",            # or "sgd"
    optimizer_momentum=0.9,      # used only for SGD
    loss="bce_logits",           # "cross_entropy" or "mse" also available
    scheduler="step",
    scheduler_step_size=2,
    scheduler_gamma=0.5,
    max_grad_norm=1.0,
)
```

Field summary:

| Field | Description |
| --- | --- |
| `learning_rate` | Positive float applied to the optimizer. |
| `epochs` | Number of full passes over the training set. |
| `batch_size` | Must match both `train_loader.batch_size` and `val_loader.batch_size`; the trainer enforces this. |
| `device` | `"cuda"` or `"cpu"`. If CUDA is requested but unavailable, the trainer falls back to CPU and logs a warning. |
| `optimizer` + `optimizer_momentum` | Select between Adam and SGD. Momentum is validated to stay within `[0,1)` and is ignored for Adam. |
| `loss` | Choose the loss module. Use `bce_logits` for binary classification, `cross_entropy` for multi-class, and `mse` for regression. |
| `scheduler`, `scheduler_step_size`, `scheduler_gamma` | Optional StepLR configuration. Supplying scheduler params while `scheduler="none"` raises immediately. |
| `max_grad_norm` | Optional gradient clipping threshold applied every step. |

### 3. DataLoader Contract

The trainer expects both loaders to yield batches shaped exactly as `(features, mask, labels)`:

- `features`: `Tensor[batch, seq_len, num_features]`
- `mask`: Boolean tensor with shape `[batch, seq_len]`
- `labels`: `Tensor[batch, ...]` (binary labels `[batch, 1]`, class indices, etc.)

Additional rules enforced at runtime:

- Each `DataLoader` must expose a `batch_size` attribute matching `TrainConfig.batch_size`.
- Loaders must be sized (`len(loader) > 0`). Empty splits produce a `ValueError` before training starts.
- Mask dtype must be `torch.bool`, and batch dimensions must align. Violations raise clear exceptions so issues are fixed before lengthy training runs.

#### 3.1 Building Sepsis DataLoaders

Use `core.data.create_dataloaders(train_config, df=None)` to construct deterministic train/validation splits for the PhysioNet dataset. The helper:

- Caches each patient’s sequence once (via `SepsisDataset`) to keep epoch iteration fast.
- Applies the requested `train_val_split` and `split_seed`, failing fast if either split would be empty.
- Logs the resulting patient counts so notebooks can display them for students.
- Pads variable-length sequences inside a custom `collate_fn`, ensuring every batch satisfies the Trainer contract.

`core.data.build_sepsis_dataloaders` remains as a backwards-compatible alias but new code must import `create_dataloaders` directly.

### 4. Running the Trainer

```python
from core.train import Trainer

trainer = Trainer(
    model=my_sequence_model,
    train_loader=train_loader,  # must follow the contract above
    val_loader=val_loader,
    config=train_config,
)
history, best_model_path = trainer.fit()
```

During `.fit()` the trainer:

1. Logs device selection and epoch progress via the module logger (`core.train.trainer`).
2. Applies the requested optimizer, loss, scheduler, and optional gradient clipping.
3. Validates every batch before moving it to the target device, ensuring silent shape mismatches never propagate.
4. Returns a tuple `(history, best_model_path)` so notebooks can both visualize convergence (`history`) and reload the best checkpoint later via the saved path.

### 5. Notebook Workflow

Every modeling notebook should follow this pattern:

1. Load YAML / environment config (data paths, hyperparameters).
2. Instantiate `TrainConfig` with the desired hyperparameters.
3. Build data loaders via the helpers in `core.data` (these helpers should already respect the requested batch size).
4. Instantiate the specific `BaseSequenceModel` implementation.
5. Initialize `Trainer(model, train_loader, val_loader, config)` and call `.fit()` to obtain `history, best_model_path`.

By keeping the training logic centralized, notebooks can stay narrative-driven and focus on pedagogy rather than PyTorch plumbing. Refer back to this guide whenever you need to expose a new knob—extend `TrainConfig` + `Trainer` together, add tests, and document the behavior here.
