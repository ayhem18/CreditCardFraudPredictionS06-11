"""Score a frozen OOF table: ROC AUC, precision, recall.

Reads ``storage/artifacts/training/<exp_name>/oof.csv``. Does not train.

Also writes a precision/recall curve on a uniform threshold grid.

Usage::

    python experiments/baselines/eval_oof.py --exp-name hgb_baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ccf_ml.file_utils import (
    get_repo_root,
    get_training_config_path,
    get_training_eval_dir,
    get_training_oof_path,
)
from training_io import (
    OOF_FOLD_COL,
    OOF_PROB_COL,
    OOF_TARGET_COL,
    read_config,
    read_oof,
)

OVERALL_FOLD_LABEL = "oof"
DEFAULT_THRESHOLD_STEP = 0.05
EVAL_METRICS_FILENAME = "metrics.csv"
EVAL_PR_CURVE_TABLE_FILENAME = "pr_curve.csv"
EVAL_PR_CURVE_PLOT_FILENAME = "pr_curve.png"
_CURVE_COLUMNS = (
    "threshold",
    "n_pred_pos",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "fn",
    "tn",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-name",
        required=True,
        help="Training experiment identity whose oof.csv should be scored.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for the per-fold table. Default: 0.5",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=DEFAULT_THRESHOLD_STEP,
        help=(
            "Step of the precision/recall curve on [0, 1]. "
            f"Default: {DEFAULT_THRESHOLD_STEP}"
        ),
    )
    args = parser.parse_args()
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {args.threshold}")

    exp_name = args.exp_name
    config_path = get_training_config_path(exp_name)
    oof_path = get_training_oof_path(exp_name)
    config = read_config(path=config_path)
    oof = read_oof(path=oof_path)
    y_true = oof[OOF_TARGET_COL].to_numpy()
    y_true_int = y_true.astype(np.int64)
    if not np.all(y_true == y_true_int):
        raise ValueError("OOF is_fraud is not integer-valued")
    y_prob = oof[OOF_PROB_COL].to_numpy(dtype=np.float64)
    fold_ids = oof[OOF_FOLD_COL].to_numpy()
    unique_folds = np.sort(np.unique(fold_ids))
    n_folds = len(unique_folds)
    print("eval_oof")
    print(f"exp_name: {exp_name}")
    print(f"model: {config.get('model')}")
    print(f"args: {config.get('args')}")
    print(f"oof: {_repo_relative(oof_path)}")
    print(f"n_rows: {len(oof)}")
    print(f"threshold: {args.threshold}")

    rows: list[dict[str, object] | None] = [None] * (n_folds + 1)
    for i in range(n_folds):
        fold_id = int(unique_folds[i])
        mask = fold_ids == fold_id
        metrics = binary_metrics(
            y_true=y_true_int[mask],
            y_prob=y_prob[mask],
            threshold=args.threshold,
        )
        metrics["fold_id"] = fold_id
        rows[i] = metrics
        print(
            f"fold {fold_id}: roc_auc={metrics['roc_auc']:.5f} "
            f"precision={metrics['precision']:.5f} "
            f"recall={metrics['recall']:.5f} "
            f"n_pos={metrics['n_pos']} n_pred_pos={metrics['n_pred_pos']}"
        )
    overall = binary_metrics(
        y_true=y_true_int,
        y_prob=y_prob,
        threshold=args.threshold,
    )
    overall["fold_id"] = OVERALL_FOLD_LABEL
    rows[n_folds] = overall
    print(
        f"oof: roc_auc={overall['roc_auc']:.5f} "
        f"precision={overall['precision']:.5f} "
        f"recall={overall['recall']:.5f} "
        f"n_pos={overall['n_pos']} n_pred_pos={overall['n_pred_pos']}"
    )
    filled = [row for row in rows if row is not None]
    table = pd.DataFrame(filled)
    column_order = (
        "fold_id",
        "n",
        "n_pos",
        "n_pred_pos",
        "roc_auc",
        "threshold",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "tn",
    )
    table = table.loc[:, list(column_order)]
    print(table.to_string(index=False))
    eval_dir = get_training_eval_dir(exp_name)
    out_path = eval_dir / EVAL_METRICS_FILENAME
    table.to_csv(out_path, index=False)
    print(f"wrote: {_repo_relative(out_path)}")

    curve = precision_recall_curve_table(
        y_true=y_true_int,
        y_prob=y_prob,
        step=args.threshold_step,
    )
    print("precision/recall curve (OOF)")
    print(curve.to_string(index=False))
    curve_path = eval_dir / EVAL_PR_CURVE_TABLE_FILENAME
    curve.to_csv(curve_path, index=False)
    print(f"wrote: {_repo_relative(curve_path)}")
    fig, _axes = plot_precision_recall_curve(curve)
    plot_path = eval_dir / EVAL_PR_CURVE_PLOT_FILENAME
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"wrote: {_repo_relative(plot_path)}")
    print("done")


def threshold_grid(*, step: float) -> np.ndarray:
    """Return ``0, step, ..., 1``. ``step`` must divide ``1``."""
    step_f = float(step)
    if step_f <= 0.0 or step_f > 1.0:
        raise ValueError(f"step must be in (0, 1], got {step_f}")
    n_steps = int(round(1.0 / step_f))
    if not np.isclose(n_steps * step_f, 1.0):
        raise ValueError(f"step {step_f} does not divide 1")
    return np.linspace(0.0, 1.0, n_steps + 1)


def precision_recall_curve_table(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step: float,
) -> pd.DataFrame:
    """Precision and recall on a uniform threshold grid."""
    thresholds = threshold_grid(step=step)
    n_thresholds = len(thresholds)
    rows: list[dict[str, object] | None] = [None] * n_thresholds
    for i in range(n_thresholds):
        metrics = binary_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=float(thresholds[i]),
        )
        rows[i] = {name: metrics[name] for name in _CURVE_COLUMNS}
    filled = [row for row in rows if row is not None]
    return pd.DataFrame(filled).loc[:, list(_CURVE_COLUMNS)]


def plot_precision_recall_curve(curve: pd.DataFrame) -> tuple[Figure, tuple[Axes, Axes]]:
    """Threshold sweep and precision-vs-recall at those operating points."""
    missing = [name for name in ("threshold", "precision", "recall") if name not in curve.columns]
    if missing:
        raise ValueError(f"curve table missing columns: {missing}")
    if curve.empty:
        raise ValueError("curve table is empty")
    threshold = curve["threshold"].to_numpy(dtype="float64")
    precision = curve["precision"].to_numpy(dtype="float64")
    recall = curve["recall"].to_numpy(dtype="float64")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    vs_threshold, pr = axes
    vs_threshold.plot(threshold, precision, marker="o", label="precision")
    vs_threshold.plot(threshold, recall, marker="o", label="recall")
    vs_threshold.set_xlabel("threshold")
    vs_threshold.set_ylabel("score")
    vs_threshold.set_xlim(0.0, 1.0)
    vs_threshold.set_ylim(0.0, 1.05)
    vs_threshold.set_xticks(np.linspace(0.0, 1.0, 11))
    vs_threshold.grid(True, alpha=0.3)
    vs_threshold.legend()
    pr.plot(recall, precision, marker="o")
    n_points = len(threshold)
    for i in range(n_points):
        if i % 2 != 0:
            continue
        pr.annotate(
            f"{threshold[i]:.2f}",
            (recall[i], precision[i]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    pr.set_xlabel("recall")
    pr.set_ylabel("precision")
    pr.set_xlim(0.0, 1.05)
    pr.set_ylim(0.0, 1.05)
    pr.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, (vs_threshold, pr)


def binary_metrics(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    """ROC AUC plus precision/recall at ``threshold``."""
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot score an empty fold")
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true == 1)),
        "n_pred_pos": int(np.sum(y_pred == 1)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(get_repo_root()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
