"""Freeze the alarm x profile x target stratified k-fold over ALL rows.

The unified CV shared by the global and region trainers: one fold
assignment stratified on alarm x profile x target (2 x 90 x 2 = 360
theoretical cells), so OOF numbers from both modeling approaches are
directly comparable. See ``ccf_ml.region_modeling.alarm_profile_cv``.

Usage::

    python scripts/compute_alarm_profile_cv.py [--overwrite]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ccf_ml.cross_validation.alarm import flag_target_stratum, stratum_count_names
from ccf_ml.file_utils import get_train_csv_path
from ccf_ml.region_modeling import (
    AlarmProfileStratifiedCv,
    alarm_profile_cv_path,
    alarm_profile_fold_summaries,
    alarm_target_totals,
    build_alarm_profile_stratified_cv,
    load_alarm_profile_stratified_cv,
    with_profile_columns,
    write_alarm_profile_stratified_cv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = alarm_profile_cv_path()
    if path.exists() and not args.overwrite:
        cv = load_alarm_profile_stratified_cv()
        print(f"frozen partition already exists: {path} (n_rows={cv.n_rows})")
        return
    frame = with_profile_columns(pd.read_csv(get_train_csv_path()))
    cv = build_alarm_profile_stratified_cv(frame)
    _report(cv, frame)
    write_alarm_profile_stratified_cv(cv, overwrite=args.overwrite)
    print(f"wrote {path}")


def _report(cv: AlarmProfileStratifiedCv, frame: pd.DataFrame) -> None:
    totals = alarm_target_totals(cv.stratum_counts)
    flag_totals = stratum_count_names(flag_target_stratum(frame))
    print(f"rows: {cv.n_rows}")
    print(f"strata: {len(cv.stratum_counts)} (min_stratum_rows={cv.min_stratum_rows})")
    print(f"alarm x target totals: {totals}")
    if totals != flag_totals:
        raise ValueError(
            f"alarm x target totals disagree with the flag stratum: {flag_totals}"
        )
    print("alarm x target totals match the flag-stratified scheme: ok")
    summaries = alarm_profile_fold_summaries(cv, frame)
    for summary in summaries:
        fold_totals = alarm_target_totals(summary.stratum_counts_val)
        occupancy = " ".join(
            f"{name}={fold_totals[name]}" for name in sorted(fold_totals)
        )
        print(
            f"fold {summary.fold_id}: n_train={summary.n_train} n_val={summary.n_val} "
            f"strata_val={len(summary.stratum_counts_val)} | {occupancy}"
        )


if __name__ == "__main__":
    try:
        main()
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
