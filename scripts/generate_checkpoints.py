#!/usr/bin/env python
"""Utility to (re)generate tutorial model checkpoints.

This script mirrors the training that happens in notebooks 02–04 but runs it in
pure Python so CI or developers can refresh the saved weights without opening
the notebooks themselves.
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
import yaml

from core.config import GRUConfig, LSTMConfig, TrainConfig, TransformerConfig
from core.data import create_dataloaders
from core.data.physionet_sepsis import get_sepsis_data
from core.models import GRUModel, LSTMModel, TransformerModel
from core.models.base import BaseSequenceModel
from core.notebook import ensure_project_root
from core.train import Trainer

ModelRegistryEntry = tuple[type[Any], type[BaseSequenceModel]]
MODEL_REGISTRY: dict[str, ModelRegistryEntry] = {
    "gru": (GRUConfig, GRUModel),
    "lstm": (LSTMConfig, LSTMModel),
    "transformer": (TransformerConfig, TransformerModel),
}


_DATA_CACHE: pd.DataFrame | None = None


def _get_data() -> pd.DataFrame:
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = get_sepsis_data()
    return _DATA_CACHE


def train_model(name: str, skip_existing: bool) -> None:
    config_cls, model_cls = MODEL_REGISTRY[name]
    root = ensure_project_root()
    config_path = root / "configs" / f"{name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config_raw = yaml.safe_load(config_path.read_text())
    model_config = config_cls(**config_raw["model"])
    train_config = TrainConfig(**config_raw["training"])

    data = _get_data()
    train_loader, val_loader = create_dataloaders(train_config=train_config, df=data)

    model = model_cls(model_config)
    checkpoint_path = root / "models" / f"{name}_best.pt"
    if skip_existing and checkpoint_path.exists():
        print(f"[{name}] skipping existing checkpoint at {checkpoint_path}")
        return
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        train_config,
        save_path=checkpoint_path,
    )
    history, best_path = trainer.fit()
    best_val = min(history["val_loss"])
    print(f"[{name}] epochs={len(history['train_loss'])} | best_val={best_val:.4f}")
    print(f"[{name}] checkpoint saved to {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        choices=sorted(MODEL_REGISTRY.keys()),
        default=sorted(MODEL_REGISTRY.keys()),
        help="Subset of models to train (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training if the checkpoint already exists.",
    )
    args = parser.parse_args()
    for model_name in args.models:
        train_model(model_name, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
