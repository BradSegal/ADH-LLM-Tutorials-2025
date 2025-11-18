"""Tests for Trainer validation metric helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from core.config import TrainConfig
from core.models.base import BaseSequenceModel
from core.train import Trainer


class _TinySequenceDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, samples: Sequence[tuple[Tensor, Tensor, Tensor]]) -> None:
        self._samples = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self._samples[index]


class _DummySequenceModel(BaseSequenceModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch_indices = torch.arange(x.size(0), device=x.device)
        seq_lengths = mask.sum(dim=1) - 1
        last_hidden = x[batch_indices, seq_lengths]
        logits: Tensor = self.linear(last_hidden)
        return logits.squeeze(-1)


def _build_dataset() -> _TinySequenceDataset:
    samples: list[tuple[Tensor, Tensor, Tensor]] = []
    feature_values = [0.0, 1.0, 2.0, 3.0]
    for value in feature_values:
        feature = torch.tensor([[value]], dtype=torch.float32)
        mask = torch.ones(1, dtype=torch.bool)
        label = torch.tensor(float(value > 1.5), dtype=torch.float32)
        samples.append((feature, mask, label))
    return _TinySequenceDataset(samples)


def _build_loader(
    dataset: _TinySequenceDataset,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    return DataLoader(dataset, batch_size=2, shuffle=True)


def test_compute_val_metric_restores_training_mode() -> None:
    torch.manual_seed(0)
    dataset = _build_dataset()
    train_loader = _build_loader(dataset)
    val_loader = _build_loader(dataset)
    model = _DummySequenceModel()
    config = TrainConfig(
        learning_rate=0.05,
        epochs=3,
        batch_size=2,
        device="cpu",
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        save_path=None,
    )

    history, _ = trainer.fit()
    trainer.model.train()

    score = trainer.compute_val_metric("auprc")

    assert trainer.model.training
    assert abs(score - history["val_metrics"]["auprc"][-1]) < 1e-6
