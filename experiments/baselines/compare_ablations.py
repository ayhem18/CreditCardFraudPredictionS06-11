"""Compare OOF tables across ablation runs: pooled, per-region, per-fold.

Reads each experiment's ``oof.csv`` (same frozen folds), joins the alarm-flag
mask from ``train.csv``, and reports:

  * pooled ROC AUC and delta vs the baseline experiment
  * alarm-region and no-alarm-region ROC AUC
  * ROI occupancy: no-alarm rows with ``y_prob <= --roi-threshold``
    (rows, frauds, fraud rate, ROC AUC)
  * per-fold pooled AUCs and deltas vs the baseline experiment

Usage::

    python experiments/baselines/compare_ablations.py \
        --exp-name hgb_baseline --exp-name hgb_no_time \
        --exp-name hgb_no_time_no_card_type --exp-name hgb_no_time_no_channel \
        --exp-name hgb_no_time_no_channel_no_card_type
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ccf_ml.cross_validation.alarm import alarm_flag_mask
from ccf_ml.cross_validation.constants import (
    ID_COLUMN,
    TARGET_COLUMN,
)
from ccf_ml.file_utils import (
    get_repo_root,
    get_script_table_dir,
    get_train_csv_path,
    get_training_oof_path,
)
from training_io import (
    OOF_FOLD_COL,
    OOF_ID_COL,
    OOF_PROB_COL,
    OOF_TARGET_COL,
    read_oof,
)

ROI_DEFAULT_THRESHOLD = 0.02
SUMMARY_FILENAME = "summary.csv"
FOLD_AUC_FILENAME = "fold_auc.csv"
REGION_LABELS = ("alarm", "no_alarm")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-name",
        action="append",
        required=True,
        help="Training experiment whose oof.csv should be compared (repeatable).",
    )
    parser.add_argument(
        "--roi-threshold",
        type=float,
        default=ROI_DEFAULT_THRESHOLD,
        help=(
            "ROI definition: no-alarm rows with y_prob <= this. "
            f"Default: {ROI_DEFAULT_THRESHOLD}"
        ),
    )
    args = parser.parse_args()
    if not (0.0 <= args.roi_threshold <= 1.0):
        raise ValueError(
            f"roi-threshold must be in [0, 1], got {args.roi_threshold}"
        )
    exp_names = list(dict.fromkeys(args.exp_name))

    train = pd.read_csv(get_train_csv_path())
    alarm = alarm_flag_mask(train).to_numpy(dtype=bool)
    alarm_by_id = pd.Series(alarm, index=train[ID_COLUMN].to_numpy())

    tables: dict[str, pd.DataFrame] = {}
    for exp_name in exp_names:
        oof = read_oof(path=get_training_oof_path(exp_name))
        tables[exp_name] = _with_regions(oof, alarm_by_id=alarm_by_id)
    baseline_name = exp_names[0]
    baseline = tables[baseline_name]

    summary_rows = []
    for exp_name in exp_names:
        oof = tables[exp_name]
        row: dict[str, object] = {
            "exp_name": exp_name,
            "pooled_auc": _auc(oof),
            "delta_vs_baseline": _auc(oof) - _auc(baseline),
            "alarm_auc": _auc(oof[oof["region"] == "alarm"]),
            "no_alarm_auc": _auc(oof[oof["region"] == "no_alarm"]),
        }
        roi = oof[(oof["region"] == "no_alarm") & (oof[OOF_PROB_COL] <= args.roi_threshold)]
        n_roi_fraud = int(roi[OOF_TARGET_COL].sum())
        row["roi_threshold"] = args.roi_threshold
        row["roi_rows"] = len(roi)
        row["roi_frauds"] = n_roi_fraud
        row["roi_fraud_rate"] = n_roi_fraud / len(roi) if len(roi) else np.nan
        row["roi_auc"] = _auc(roi)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    print(f"regions: alarm share={alarm.mean():.4f} (joined from train.csv)")
    print(f"baseline exp: {baseline_name}")
    print(summary.to_string(index=False))

    fold_rows = []
    for exp_name in exp_names:
        oof = tables[exp_name]
        for fold_id in sorted(oof[OOF_FOLD_COL].unique()):
            fold_oof = oof[oof[OOF_FOLD_COL] == fold_id]
            fold_rows.append(
                {
                    "exp_name": exp_name,
                    "fold_id": int(fold_id),
                    "auc": _auc(fold_oof),
                }
            )
    fold_auc = pd.DataFrame(fold_rows)
    fold_wide = fold_auc.pivot(index="fold_id", columns="exp_name", values="auc")
    for exp_name in exp_names[1:]:
        fold_wide[f"delta_{exp_name}"] = fold_wide[exp_name] - fold_wide[baseline_name]
    print("\nper-fold pooled AUC (rows = fold ids)")
    print(fold_wide.to_string())

    table_dir = get_script_table_dir(Path(__file__).name, group="baselines")
    summary_path = table_dir / SUMMARY_FILENAME
    summary.to_csv(summary_path, index=False)
    fold_path = table_dir / FOLD_AUC_FILENAME
    fold_wide.to_csv(fold_path)
    print(f"\nwrote: {_repo_relative(summary_path)}")
    print(f"wrote: {_repo_relative(fold_path)}")
    print("done")


def _with_regions(
    oof: pd.DataFrame,
    *,
    alarm_by_id: pd.Series,
) -> pd.DataFrame:
    """Attach the boolean alarm region as a column."""
    flags = oof[OOF_ID_COL].map(alarm_by_id)
    if flags.isna().any():
        raise ValueError("OOF ids missing from train.csv alarm join")
    frame = oof.copy()
    frame["region"] = np.where(flags.to_numpy(dtype=bool), "alarm", "no_alarm")
    return frame


def _auc(frame: pd.DataFrame) -> float:
    if len(frame) == 0 or frame[OOF_TARGET_COL].nunique() < 2:
        return float("nan")
    return float(
        roc_auc_score(
            frame[OOF_TARGET_COL].to_numpy(),
            frame[OOF_PROB_COL].to_numpy(dtype=np.float64),
        )
    )


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(get_repo_root()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
