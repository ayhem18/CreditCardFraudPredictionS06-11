"""OOF-guided weight search for the two-stage blend; optionally build it.

Searches probability weights on the OOF tables: (w_hgb, w_xgb) on the alarm
side, (w_hgb, w_xgb, w_region) on the no-alarm side, grid step 0.1 with a
0.05 refinement around the best. Selection rule (pre-registered): if the
best grid gain over the best "simple" configuration (weights in multiples
of 0.25) is under 0.0003 AUC, keep the simple one - the grid has hundreds
of candidates and small gains are selection noise.

With ``--build <exp_name>`` the chosen weights are applied to the test
submissions and a submission experiment directory is written.

Usage::

    python experiments/baselines/blend_weights.py
    python experiments/baselines/blend_weights.py --build blend_weighted_v1
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

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
    claim_training_exp_dir,
    get_repo_root,
    get_train_csv_path,
    get_training_config_path,
    get_training_oof_path,
)
from training_io import write_config

HGB = "hgb_baseline"
XGB = "xgb_baseline"
REGION = "rgn_v4_quantiles_xgb"
SIMPLE_GAIN_FLOOR = 0.0003


def _simplex(*, n_parts: int, step: float) -> list[tuple[float, ...]]:
    """All weight tuples of length n_parts summing to 1 at the given step."""
    levels = np.arange(0, 1.0 + step / 2, step)
    out = []
    for combo in itertools.product(levels, repeat=n_parts):
        if abs(sum(combo) - 1.0) < step / 2:
            out.append(tuple(round(float(w), 4) for w in combo))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default=None, help="Experiment name to write.")
    args = parser.parse_args()

    train = pd.read_csv(get_train_csv_path())
    alarm = alarm_flag_mask(train, columns=ALARM_FLAG_COLUMNS).to_numpy()
    oof = None
    for exp in (HGB, XGB):
        part = pd.read_csv(get_training_oof_path(exp))[["id", "y_prob"]]
        part = part.rename(columns={"y_prob": exp})
        oof = part if oof is None else oof.merge(part, on="id", validate="one_to_one")
    region_oof = pd.read_csv(get_training_oof_path(REGION))[["id", "y_prob"]]
    region_oof = region_oof.rename(columns={"y_prob": REGION})
    oof = oof.merge(region_oof, on="id", how="left", validate="one_to_one")
    y = train.loc[:, [ID_COLUMN, TARGET_COLUMN]]
    oof = oof.merge(y, on="id", validate="one_to_one")
    region_ids = set(region_oof["id"])
    in_region = oof["id"].isin(region_ids).to_numpy()

    y_arr = oof[TARGET_COLUMN].to_numpy()
    hgb = oof[HGB].to_numpy()
    xgb = oof[XGB].to_numpy()
    region = oof[REGION].to_numpy()
    alarm_mask = alarm
    na_mask = in_region

    def assembly_auc(w_alarm: tuple[float, float], w_na: tuple[float, float, float]) -> float:
        score = np.empty(len(oof), dtype=np.float64)
        score[alarm_mask] = w_alarm[0] * hgb[alarm_mask] + w_alarm[1] * xgb[alarm_mask]
        score[na_mask] = (
            w_na[0] * hgb[na_mask]
            + w_na[1] * xgb[na_mask]
            + w_na[2] * region[na_mask]
        )
        return float(roc_auc_score(y_arr, score))

    def search(step: float) -> tuple[tuple[float, float], tuple[float, float, float], float]:
        best = (None, None, -1.0)
        for w_alarm in _simplex(n_parts=2, step=step):
            for w_na in _simplex(n_parts=3, step=step):
                auc = assembly_auc(w_alarm, w_na)
                if auc > best[2]:
                    best = (w_alarm, w_na, auc)
        return best  # type: ignore[return-value]

    print("coarse grid (step 0.1)")
    best = search(0.1)
    print(f"  best: alarm hgb/xgb {best[0]}, no_alarm hgb/xgb/region {best[1]} -> {best[2]:.5f}")

    simple_step = 0.25
    print(f"simple grid (step {simple_step})")
    best_simple = search(simple_step)
    print(f"  best: alarm {best_simple[0]}, no_alarm {best_simple[1]} -> {best_simple[2]:.5f}")

    references = {
        "xgb alone": ((0.0, 1.0), (0.0, 1.0, 0.0)),
        "two_stage_v1 (equal weights)": ((0.5, 0.5), (1 / 3, 1 / 3, 1 / 3)),
        "xgb + region only": ((0.0, 1.0), (0.0, 0.5, 0.5)),
    }
    for name, (w_alarm, w_na) in references.items():
        print(f"  reference {name}: {assembly_auc(w_alarm, w_na):.5f}")

    if best[2] - best_simple[2] < SIMPLE_GAIN_FLOOR:
        w_alarm, w_na, chosen_auc, rule = (*best_simple, "simple (robustness rule)")
    else:
        w_alarm, w_na, chosen_auc, rule = (*best, "grid")
    print(f"\nchosen [{rule}]: alarm hgb/xgb {w_alarm}, no_alarm hgb/xgb/region {w_na} -> OOF {chosen_auc:.5f}")

    if args.build is None:
        return
    sub = None
    for exp in (HGB, XGB):
        path = get_repo_root() / "storage" / "artifacts" / "training" / exp / "submission.csv"
        part = pd.read_csv(path).rename(columns={TARGET_COLUMN: exp})
        sub = part if sub is None else sub.merge(part, on=ID_COLUMN, validate="one_to_one")
    region_path = (
        get_repo_root() / "storage" / "artifacts" / "training" / REGION / "test_no_alarm.csv"
    )
    region_sub = pd.read_csv(region_path).rename(columns={"y_prob": REGION})
    sub = sub.merge(region_sub, on=ID_COLUMN, how="left", validate="one_to_one")
    test_alarm = alarm_flag_mask(
        pd.read_csv(get_repo_root() / "storage" / "raw_data" / "test.csv"),
        columns=ALARM_FLAG_COLUMNS,
    ).to_numpy()
    in_region_test = sub[ID_COLUMN].isin(set(region_sub[ID_COLUMN])).to_numpy()
    if not np.array_equal(in_region_test, ~test_alarm):
        raise ValueError("region test predictions do not cover exactly the no-alarm test rows")
    score = np.empty(len(sub), dtype=np.float64)
    score[test_alarm] = w_alarm[0] * sub.loc[test_alarm, HGB].to_numpy() + w_alarm[1] * sub.loc[test_alarm, XGB].to_numpy()
    score[in_region_test] = (
        w_na[0] * sub.loc[in_region_test, HGB].to_numpy()
        + w_na[1] * sub.loc[in_region_test, XGB].to_numpy()
        + w_na[2] * sub.loc[in_region_test, REGION].to_numpy()
    )
    exp_dir = claim_training_exp_dir(args.build)
    submission_path = exp_dir / "submission.csv"
    pd.DataFrame({ID_COLUMN: sub[ID_COLUMN].to_numpy(), TARGET_COLUMN: score}).to_csv(
        submission_path, index=False
    )
    write_config(
        path=get_training_config_path(args.build),
        payload={
            "exp_name": args.build,
            "model": "weighted_two_stage",
            "weights": {
                "alarm_hgb_xgb": list(w_alarm),
                "no_alarm_hgb_xgb_region": list(w_na),
            },
            "oof_auc": chosen_auc,
            "selection_rule": rule,
        },
    )
    print(f"wrote {submission_path} n={len(sub)}")


if __name__ == "__main__":
    try:
        main()
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
