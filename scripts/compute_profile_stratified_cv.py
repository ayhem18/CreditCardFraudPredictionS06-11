"""Freeze the profile x target stratified k-fold for the no-alarm region.

Usage::

    python scripts/compute_profile_stratified_cv.py [--overwrite]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ccf_ml.cross_validation.flag_stratified import load_flag_stratified_cv
from ccf_ml.file_utils import get_cv_splits_dir, get_train_csv_path
from ccf_ml.region_modeling import (
    ProfileStratifiedCv,
    build_no_alarm_frame,
    build_profile_stratified_cv,
    dump_to_path,
    fold_summaries,
    load_profile_stratified_cv,
)
from ccf_ml.region_modeling.constants import (
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    PROFILE_CV_FILENAME,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    path = get_cv_splits_dir() / PROFILE_CV_FILENAME
    if path.exists() and not args.overwrite:
        cv = load_profile_stratified_cv()
        print(f"frozen partition already exists: {path} (n_rows={cv.n_rows})")
        return
    frame = pd.read_csv(get_train_csv_path())
    region = build_no_alarm_frame(frame)
    print(f"region rows: {len(region)}")
    cv = build_profile_stratified_cv(
        region, n_splits=DEFAULT_N_SPLITS, random_state=DEFAULT_RANDOM_STATE
    )
    summaries = fold_summaries(cv, region)
    for summary in summaries:
        print(
            f"fold {summary.fold_id}: n_train={summary.n_train} n_val={summary.n_val} "
            f"strata_train={len(summary.stratum_counts_train)} "
            f"strata_val={len(summary.stratum_counts_val)}"
        )
    dump_to_path(cv)
    print(f"wrote {path}")


if __name__ == "__main__":
    try:
        main()
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
