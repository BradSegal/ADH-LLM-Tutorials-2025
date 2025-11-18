"""Shared runtime helpers for tutorial notebooks.

Each notebook is expected to call :func:`ensure_project_root` in its first
import cell. This keeps the execution context predictable regardless of where
the user launches the notebook (Colab, local editor, etc.).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from core.data.physionet_sepsis import FEATURE_COLUMNS


def _detect_project_root(start: Path | None = None) -> Path:
    """Return the repository root (directory containing ``pyproject.toml``)."""

    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate project root. Ensure pyproject.toml exists."
    )


def ensure_project_root(start: Path | None = None) -> Path:
    """Change the working directory to the project root and update sys.path."""

    if "PYPROJECT_ROOT" in os.environ:
        root = Path(os.environ["PYPROJECT_ROOT"]).resolve()
    else:
        root = _detect_project_root(start)
        os.environ["PYPROJECT_ROOT"] = str(root)

    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def get_feature_names() -> list[str]:
    """Return the canonical feature ordering for attribution plots."""

    return list(FEATURE_COLUMNS)
