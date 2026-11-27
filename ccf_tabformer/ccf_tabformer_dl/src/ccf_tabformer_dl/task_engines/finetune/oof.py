"""Assemble held-out OOF scores and the mean-of-folds test submission."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from dl_toolbox.dl_engine.exp_engine.train_val_engine.default_layout import (
    DefaultLayoutTrainValEngine,
)

STACK_OOF_AUC = 0.95062
VAL_ROW_IDS_FILE = "val_row_ids.npy"
VAL_LOGITS_FILE = "val_logits.npy"
VAL_LABELS_FILE = "val_labels.npy"
TEST_ROW_IDS_FILE = "test_row_ids.npy"
TEST_LOGITS_FILE = "test_logits.npy"
SUBMISSION_COLUMNS = ("id", "is_fraud")
OOF_COLUMNS = ("id", "fold_id", "is_fraud", "y_prob")


def sigmoid_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def inspection_payload_directory(artifacts_dir: Path) -> Path:
    """Newest ``inspection_runs/epoch=N`` directory under a finished run."""
    root = artifacts_dir / DefaultLayoutTrainValEngine.INSPECTION_RUNS_DIRNAME
    if not root.is_dir():
        raise FileNotFoundError(f"inspection_runs not found under {artifacts_dir}")
    best: Path | None = None
    best_epoch = -1
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("epoch="):
            continue
        epoch = int(name.split("=", 1)[1])
        if epoch >= best_epoch:
            best_epoch = epoch
            best = child
    if best is None:
        raise FileNotFoundError(f"no epoch=* inspect payload under {root}")
    return best


def assemble_oof(*, fold_payloads: Sequence[Path]) -> dict:
    """Concatenate per-fold val predictions. Each train row appears once."""
    n_folds = len(fold_payloads)
    if n_folds < 1:
        raise ValueError("fold_payloads must be non-empty")
    row_ids_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    scores_parts: list[np.ndarray] = []
    fold_ids_parts: list[np.ndarray] = []
    for fold in range(n_folds):
        payload = Path(fold_payloads[fold])
        row_ids = np.load(payload / VAL_ROW_IDS_FILE)
        labels = np.load(payload / VAL_LABELS_FILE)
        scores = sigmoid_logits(np.load(payload / VAL_LOGITS_FILE))
        if row_ids.ndim != 1 or labels.ndim != 1 or scores.ndim != 1:
            raise ValueError(
                f"fold {fold} val arrays must be 1-D, got "
                f"{row_ids.shape} {labels.shape} {scores.shape}"
            )
        if not (row_ids.shape == labels.shape == scores.shape):
            raise ValueError(
                f"fold {fold} val lengths disagree: "
                f"{row_ids.shape} {labels.shape} {scores.shape}"
            )
        row_ids_parts.append(np.asarray(row_ids, dtype=np.int64))
        labels_parts.append(np.asarray(labels, dtype=np.float64))
        scores_parts.append(np.asarray(scores, dtype=np.float64))
        fold_ids_parts.append(np.full(int(row_ids.shape[0]), fold, dtype=np.int64))
    row_ids = np.concatenate(row_ids_parts, axis=0)
    labels = np.concatenate(labels_parts, axis=0)
    scores = np.concatenate(scores_parts, axis=0)
    fold_ids = np.concatenate(fold_ids_parts, axis=0)
    unique = np.unique(row_ids)
    if int(unique.shape[0]) != int(row_ids.shape[0]):
        raise ValueError("OOF val_row_ids overlap across folds")
    per_fold_auc = [
        float(roc_auc_score(labels[fold_ids == fold], scores[fold_ids == fold]))
        for fold in range(n_folds)
    ]
    oof_auc = float(roc_auc_score(labels, scores))
    return {
        "n_folds": n_folds,
        "n_oof": int(row_ids.shape[0]),
        "oof_auc": oof_auc,
        "per_fold_auc": per_fold_auc,
        "stack_oof_auc": STACK_OOF_AUC,
        "oof_minus_stack": oof_auc - STACK_OOF_AUC,
        "row_ids": row_ids,
        "fold_ids": fold_ids,
        "labels": labels,
        "scores": scores,
    }


def average_test_scores(
    *, fold_payloads: Sequence[Path]
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the 5 fold models' test sigmoid scores. Same row order."""
    n_folds = len(fold_payloads)
    if n_folds < 1:
        raise ValueError("fold_payloads must be non-empty")
    first_ids = None
    stacked: list[np.ndarray] = []
    for fold in range(n_folds):
        payload = Path(fold_payloads[fold])
        row_ids = np.asarray(np.load(payload / TEST_ROW_IDS_FILE), dtype=np.int64)
        scores = sigmoid_logits(np.load(payload / TEST_LOGITS_FILE))
        if row_ids.ndim != 1 or scores.ndim != 1:
            raise ValueError(
                f"fold {fold} test arrays must be 1-D, got "
                f"{row_ids.shape} {scores.shape}"
            )
        if int(row_ids.shape[0]) != int(scores.shape[0]):
            raise ValueError(
                f"fold {fold} test id/score lengths disagree: "
                f"{row_ids.shape} {scores.shape}"
            )
        if first_ids is None:
            first_ids = row_ids
        elif not np.array_equal(first_ids, row_ids):
            raise ValueError("test_row_ids disagree across folds")
        stacked.append(scores)
    if first_ids is None:
        raise RuntimeError("average_test_scores had no folds")
    mean_scores = np.mean(np.stack(stacked, axis=0), axis=0)
    return first_ids, mean_scores


def write_oof_csv(assembled: dict, *, dest: Path) -> None:
    _write_csv(
        dest,
        header=OOF_COLUMNS,
        columns=(
            assembled["row_ids"],
            assembled["fold_ids"],
            assembled["labels"],
            assembled["scores"],
        ),
        integer_columns=frozenset({"id", "fold_id", "is_fraud"}),
    )


def write_submission_csv(
    *, row_ids: np.ndarray, scores: np.ndarray, dest: Path
) -> None:
    _write_csv(
        dest,
        header=SUBMISSION_COLUMNS,
        columns=(row_ids, scores),
        integer_columns=frozenset({"id"}),
    )


def _write_csv(
    dest: Path,
    *,
    header: tuple[str, ...],
    columns: tuple[np.ndarray, ...],
    integer_columns: frozenset[str],
) -> None:
    n_cols = len(header)
    if n_cols != len(columns):
        raise ValueError("header and columns length disagree")
    n_rows = int(columns[0].shape[0])
    for i in range(n_cols):
        if int(columns[i].shape[0]) != n_rows:
            raise ValueError("csv columns must have the same length")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(header) + "\n")
        for row in range(n_rows):
            cells: list[str] = []
            for col in range(n_cols):
                value = columns[col][row]
                name = header[col]
                if name in integer_columns:
                    cells.append(str(int(value)))
                else:
                    cells.append(f"{float(value):.10g}")
            handle.write(",".join(cells) + "\n")
