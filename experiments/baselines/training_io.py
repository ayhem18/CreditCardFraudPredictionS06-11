"""OOF table and training config files. Experiment-local, not a package API."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OOF_ID_COL = "id"
OOF_FOLD_COL = "fold_id"
OOF_TARGET_COL = "is_fraud"
OOF_PROB_COL = "y_prob"
OOF_COLUMNS = (OOF_ID_COL, OOF_FOLD_COL, OOF_TARGET_COL, OOF_PROB_COL)


def write_config(*, path: Path, payload: dict[str, object]) -> None:
    """Write the experiment configuration as indented JSON."""
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_config(*, path: Path) -> dict[str, object]:
    """Load the experiment configuration."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if type(payload) is not dict:
        raise TypeError(f"config must be a JSON object, got {type(payload)!r}")
    return payload


def write_oof(*, path: Path, frame: pd.DataFrame) -> None:
    """Write the OOF table with a fixed column order."""
    missing = [name for name in OOF_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"OOF table missing columns: {missing}")
    frame.loc[:, list(OOF_COLUMNS)].to_csv(path, index=False)


def read_oof(*, path: Path) -> pd.DataFrame:
    """Load the OOF table and require the frozen columns."""
    if not path.is_file():
        raise FileNotFoundError(f"OOF table not found: {path}")
    frame = pd.read_csv(path)
    missing = [name for name in OOF_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"OOF table missing columns: {missing}")
    if frame.empty:
        raise ValueError("OOF table is empty")
    return frame.loc[:, list(OOF_COLUMNS)]
