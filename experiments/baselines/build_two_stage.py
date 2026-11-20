"""Build the two-stage ensemble: report OOF, write the submission.

Assembly: alarm test rows score the mean probability of the two global
models (hgb_baseline, xgb_baseline); no-alarm test rows score the mean of
the two global models plus the region model's test predictions. The OOF
report mirrors the same recipe on training rows (each row's score is
out-of-fold in every contributing model).

Usage::

    python experiments/baselines/build_two_stage.py --region-exp rgn_v4_quantiles_xgb \
        --exp-name two_stage_ensemble_v1
"""

from __future__ import annotations

import argparse
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
    get_test_csv_path,
    get_train_csv_path,
    get_training_config_path,
    get_training_oof_path,
)
from training_io import write_config

GLOBAL_EXPS = ("hgb_baseline", "xgb_baseline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region-exp", default="rgn_v4_quantiles_xgb")
    parser.add_argument("--exp-name", default="two_stage_ensemble_v1")
    args = parser.parse_args()

    train = pd.read_csv(get_train_csv_path())
    train_alarm = alarm_flag_mask(train, columns=ALARM_FLAG_COLUMNS).to_numpy()
    test = pd.read_csv(get_test_csv_path())
    test_alarm = alarm_flag_mask(test, columns=ALARM_FLAG_COLUMNS).to_numpy()

    # ---- OOF side: same recipe on training rows -----------------------------
    oof = None
    for exp in GLOBAL_EXPS:
        part = pd.read_csv(get_training_oof_path(exp))[["id", "y_prob"]]
        part = part.rename(columns={"y_prob": exp})
        oof = part if oof is None else oof.merge(part, on="id", validate="one_to_one")
    region_oof = pd.read_csv(get_training_oof_path(args.region_exp))[["id", "y_prob"]]
    region_oof = region_oof.rename(columns={"y_prob": args.region_exp})
    oof = oof.merge(region_oof, on="id", how="left", validate="one_to_one")
    y = pd.read_csv(get_train_csv_path()).loc[:, [ID_COLUMN, TARGET_COLUMN]]
    oof = oof.merge(y, on="id", validate="one_to_one")
    region_train_ids = set(region_oof["id"])
    in_region = oof["id"].isin(region_train_ids).to_numpy()

    oof["score"] = oof.loc[:, list(GLOBAL_EXPS)].mean(axis=1)
    oof.loc[in_region, "score"] = oof.loc[in_region, list(GLOBAL_EXPS) + [args.region_exp]].mean(axis=1)
    report_rows = []
    for name, score in (
        ("hgb_baseline", oof["hgb_baseline"]),
        ("xgb_baseline", oof["xgb_baseline"]),
        ("global mean (hgb+xgb)", oof.loc[:, list(GLOBAL_EXPS)].mean(axis=1)),
        ("two_stage (+region on no_alarm)", oof["score"]),
    ):
        report_rows.append(
            {
                "model": name,
                "overall_auc": round(float(roc_auc_score(oof[TARGET_COLUMN], score)), 5),
                "alarm_auc": round(float(roc_auc_score(oof.loc[train_alarm, TARGET_COLUMN], score[train_alarm])), 5),
                "no_alarm_auc": round(float(roc_auc_score(oof.loc[~train_alarm, TARGET_COLUMN], score[~train_alarm])), 5),
            }
        )
    report = pd.DataFrame(report_rows)
    print(report.to_string(index=False))

    # ---- test side -----------------------------------------------------------
    sub = None
    for exp in GLOBAL_EXPS:
        path = get_repo_root() / "storage" / "artifacts" / "training" / exp / "submission.csv"
        part = pd.read_csv(path).rename(columns={TARGET_COLUMN: exp})
        sub = part if sub is None else sub.merge(part, on=ID_COLUMN, validate="one_to_one")
    region_test_path = (
        get_repo_root() / "storage" / "artifacts" / "training" / args.region_exp / "test_no_alarm.csv"
    )
    region_sub = pd.read_csv(region_test_path).rename(columns={"y_prob": args.region_exp})
    sub = sub.merge(region_sub, on=ID_COLUMN, how="left", validate="one_to_one")
    if len(sub) != len(test):
        raise ValueError(f"submission rows {len(sub)} do not match test rows {len(test)}")
    region_test_ids = set(region_sub[ID_COLUMN])
    in_region_test = sub[ID_COLUMN].isin(region_test_ids).to_numpy()
    if not np.array_equal(in_region_test, ~test_alarm):
        raise ValueError("region test predictions do not cover exactly the no-alarm test rows")

    sub[TARGET_COLUMN] = sub.loc[:, list(GLOBAL_EXPS)].mean(axis=1)
    sub.loc[in_region_test, TARGET_COLUMN] = (
        sub.loc[in_region_test, list(GLOBAL_EXPS) + [args.region_exp]].mean(axis=1)
    )
    exp_dir = claim_training_exp_dir(args.exp_name)
    submission_path = exp_dir / "submission.csv"
    sub.loc[:, [ID_COLUMN, TARGET_COLUMN]].to_csv(submission_path, index=False)
    write_config(
        path=get_training_config_path(args.exp_name),
        payload={
            "exp_name": args.exp_name,
            "model": "two_stage_probability_mean",
            "components": {
                "global": list(GLOBAL_EXPS),
                "region_no_alarm": list(GLOBAL_EXPS) + [args.region_exp],
            },
            "oof_report": report.to_dict(orient="records"),
        },
    )
    print(f"wrote {submission_path} n={len(sub)}")
    print(f"mean prob: alarm {sub.loc[test_alarm, TARGET_COLUMN].mean():.6f} | "
          f"no_alarm {sub.loc[~test_alarm, TARGET_COLUMN].mean():.6f}")


if __name__ == "__main__":
    try:
        main()
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
