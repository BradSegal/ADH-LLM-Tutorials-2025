#!/usr/bin/env python3
"""Run the full PhysioNet 2019 Sepsis data pipeline end-to-end."""

from __future__ import annotations

from pathlib import Path

from core.data.physionet_sepsis import get_sepsis_data


def main() -> None:
    """Download, preprocess, and report on the Sepsis dataset."""
    df = get_sepsis_data(config_path=Path("configs/data.yaml"))
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"No NaNs: {df.isna().sum().sum() == 0}")


if __name__ == "__main__":
    main()
