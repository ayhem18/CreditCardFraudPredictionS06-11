"""Compare global and per-region models on the unified alarm-profile CV.

Every experiment must have been trained with ``--cv alarm_profile``: the
config's cv scheme is checked, and the script asserts that all OOF tables
carry the same fold id per row. That is what makes the numbers below
directly comparable - global and region models validate on the same rows
with the same fold assignment, and per-row blends of their OOF
probabilities are fold-coherent (each row's scores come from models that
never trained on it).

Reports, per experiment: pooled and per-fold ROC AUC on all rows (global
models), on the alarm rows, and on the no-alarm rows - the region models'
home ground. Then blends of each (global, region) pair on the no-alarm
rows, and the assembled two-stage score on all rows using the best blend.

Usage::

    python experiments/baselines/compare_regimes.py \
        --global-exp hgbU_baseline --global-exp xgbU_baseline \
        --global-exp xgbU_tuned_v1 --region-exp rgnU_v4_quantiles_xgb
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ccf_ml.cross_validation.alarm import alarm_flag_mask
from ccf_ml.cross_validation.constants import (
    ALARM_FLAG_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
)
from ccf_ml.file_utils import (
    get_train_csv_path,
    get_training_config_path,
    get_training_oof_path,
)
from training_io import (
    OOF_FOLD_COL,
    OOF_ID_COL,
    OOF_PROB_COL,
    OOF_TARGET_COL,
    read_config,
    read_oof,
)

UNIFIED_SCHEME = "alarm_profile_stratified"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--global-exp",
        action="append",
        required=True,
        help="Global experiment name (all rows). Repeatable.",
    )
    parser.add_argument(
        "--region-exp",
        action="append",
        required=True,
        help="Region experiment name (no-alarm rows). Repeatable.",
    )
    args = parser.parse_args()

    train = pd.read_csv(get_train_csv_path())
    alarm_by_id = pd.Series(
        alarm_flag_mask(train, columns=ALARM_FLAG_COLUMNS).to_numpy(),
        index=train[ID_COLUMN].to_numpy(),
    )
    y_by_id = train.set_index(ID_COLUMN)[TARGET_COLUMN]

    tables: dict[str, pd.DataFrame] = {}
    for exp in list(args.global_exp) + list(args.region_exp):
        config = read_config(path=get_training_config_path(exp))
        scheme = config.get("cv", {}).get("scheme")
        if scheme != UNIFIED_SCHEME:
            raise ValueError(
                f"{exp} was trained on cv scheme {scheme!r}, not "
                f"{UNIFIED_SCHEME!r} - OOF from different schemes must "
                "never be compared"
            )
        tables[exp] = read_oof(path=get_training_oof_path(exp)).set_index(OOF_ID_COL)

    anchor = tables[args.global_exp[0]]
    for exp in args.global_exp:
        if not tables[exp].index.equals(anchor.index):
            raise ValueError(f"{exp} covers different rows than {args.global_exp[0]}")
    region_ids = anchor.index[~alarm_by_id.loc[anchor.index].to_numpy(dtype=bool)]
    for exp in args.region_exp:
        table = tables[exp]
        if set(table.index) - set(anchor.index):
            raise ValueError(f"{exp} has ids outside the global table")
        if not table.index.equals(region_ids):
            raise ValueError(
                f"{exp} does not cover exactly the no-alarm rows "
                f"({len(table.index)} ids vs {len(region_ids)})"
            )
    for exp, table in tables.items():
        aligned_fold = table[OOF_FOLD_COL]
        anchor_fold = anchor[OOF_FOLD_COL].loc[table.index]
        if not aligned_fold.equals(anchor_fold):
            raise ValueError(f"{exp} fold assignment disagrees with {args.global_exp[0]}")

    y = y_by_id.loc[anchor.index].to_numpy(dtype=np.int64)
    fold = anchor[OOF_FOLD_COL].to_numpy()
    alarm_aligned = alarm_by_id.loc[anchor.index].to_numpy(dtype=bool)
    region_aligned = ~alarm_aligned
    folds = np.sort(np.unique(fold))

    rows: list[dict[str, float | str]] = []
    for exp in args.global_exp:
        prob = tables[exp][OOF_PROB_COL].loc[anchor.index].to_numpy()
        rows.append(
            {
                "exp": exp,
                "scope": "all rows",
                "auc": _pooled(y, prob),
                **_per_fold(y, prob, fold, folds, mask=None),
            }
        )
        rows.append(
            {
                "exp": exp,
                "scope": "alarm rows",
                "auc": _pooled(y[alarm_aligned], prob[alarm_aligned]),
                **_per_fold(y, prob, fold, folds, mask=alarm_aligned),
            }
        )
        rows.append(
            {
                "exp": exp,
                "scope": "no-alarm rows",
                "auc": _pooled(y[region_aligned], prob[region_aligned]),
                **_per_fold(y, prob, fold, folds, mask=region_aligned),
            }
        )
    for exp in args.region_exp:
        in_region_ids = anchor.index[region_aligned]
        prob_full = tables[exp][OOF_PROB_COL].reindex(anchor.index)
        prob = prob_full.loc[in_region_ids].to_numpy()
        rows.append(
            {
                "exp": exp,
                "scope": "no-alarm rows",
                "auc": _pooled(y[region_aligned], prob),
                **_per_fold(y, prob_full.to_numpy(), fold, folds, mask=region_aligned),
            }
        )
    table = pd.DataFrame(rows)
    fold_cols = [f"fold_{f}" for f in folds]
    print("\n== per-regime AUC (unified alarm_profile_stratified folds) ==")
    print(table.loc[:, ["exp", "scope", "auc"] + fold_cols].to_string(index=False))

    best_global_no_alarm = max(
        (row for row in rows if row["scope"] == "no-alarm rows" and row["exp"] in args.global_exp),
        key=lambda row: float(row["auc"]),
    )
    best_global_name = str(best_global_no_alarm["exp"])
    print(
        f"\nbest global on no-alarm rows: {best_global_name} "
        f"({float(best_global_no_alarm['auc']):.5f})"
    )

    print("\n== blends on the no-alarm rows (fold-coherent) ==")
    blend_rows: list[dict[str, object]] = []
    region_y = y[region_aligned]
    for region_exp in args.region_exp:
        region_prob = (
            tables[region_exp][OOF_PROB_COL].reindex(anchor.index).to_numpy()[region_aligned]
        )
        for global_exp in args.global_exp:
            global_prob = (
                tables[global_exp][OOF_PROB_COL].loc[anchor.index].to_numpy()[region_aligned]
            )
            for label, score in (
                ("mean", 0.5 * global_prob + 0.5 * region_prob),
                ("rank_mean", 0.5 * _pct(global_prob) + 0.5 * _pct(region_prob)),
                ("g.75/r.25", 0.75 * global_prob + 0.25 * region_prob),
                ("g.25/r.75", 0.25 * global_prob + 0.75 * region_prob),
            ):
                blend_rows.append(
                    {
                        "global": global_exp,
                        "region": region_exp,
                        "weights": label,
                        "no_alarm_auc": float(roc_auc_score(region_y, score)),
                    }
                )
    blends = pd.DataFrame(blend_rows)
    print(blends.to_string(index=False))

    best_blend = blends.sort_values("no_alarm_auc").iloc[-1]
    best_region_exp = str(best_blend["region"])
    best_global_exp = str(best_blend["global"])
    region_prob = (
        tables[best_region_exp][OOF_PROB_COL].reindex(anchor.index).to_numpy()
    )
    global_prob = tables[best_global_exp][OOF_PROB_COL].loc[anchor.index].to_numpy()
    weights = str(best_blend["weights"])
    if weights == "rank_mean":
        no_alarm_score = 0.5 * _pct(global_prob[region_aligned]) + 0.5 * _pct(
            region_prob[region_aligned]
        )
    else:
        w_global = {"mean": 0.5, "g.75/r.25": 0.75, "g.25/r.75": 0.25}[weights]
        no_alarm_score = w_global * global_prob[region_aligned] + (
            1.0 - w_global
        ) * region_prob[region_aligned]
    assembled = global_prob.copy()
    assembled[region_aligned] = no_alarm_score
    print(
        f"\nassembled two-stage (alarm: {best_global_exp}; no-alarm: "
        f"{best_global_exp} + {best_region_exp} [{weights}]): "
        f"overall AUC {float(roc_auc_score(y, assembled)):.5f}"
    )
    best_overall = max(
        float(row["auc"])
        for row in rows
        if row["scope"] == "all rows" and row["exp"] in args.global_exp
    )
    print(f"best single global overall AUC: {best_overall:.5f}")


def _pooled(y: np.ndarray, prob: np.ndarray) -> float:
    return float(roc_auc_score(y, prob))


def _per_fold(
    y: np.ndarray,
    prob: np.ndarray,
    fold: np.ndarray,
    folds: np.ndarray,
    *,
    mask: np.ndarray | None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for f in folds:
        sel = fold == f if mask is None else (fold == f) & mask
        out[f"fold_{f}"] = float(roc_auc_score(y[sel], prob[sel]))
    return out


def _pct(values: np.ndarray) -> np.ndarray:
    """Percentile ranks (average ties), as in a rank-based blend."""
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


if __name__ == "__main__":
    main()
