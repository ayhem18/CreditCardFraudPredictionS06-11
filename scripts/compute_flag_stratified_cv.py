"""Freeze flag × target stratified k-fold ids into package_artifacts.

Usage::

    python scripts/compute_flag_stratified_cv.py
    python scripts/compute_flag_stratified_cv.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ccf_ml.cross_validation.constants import (
    ALARM_FLAG_COLUMNS,
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
)
from ccf_ml.cross_validation.flag_stratified import (
    FlagStratifiedCv,
    build_flag_stratified_cv,
    flag_stratified_cv_path,
    load_flag_stratified_cv,
    n_alarm_from_counts,
    n_fraud_from_counts,
    write_flag_stratified_cv,
)
from ccf_ml.file_utils import get_repo_root, get_train_csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help=f"Number of folds. Default: {DEFAULT_N_SPLITS}",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"StratifiedKFold seed. Default: {DEFAULT_RANDOM_STATE}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing flag_stratified_cv.json.",
    )
    args = parser.parse_args()
    dest = flag_stratified_cv_path()
    if dest.exists() and not args.force:
        print(f"already exists: {_repo_relative(dest)}")
        print("pass --force to recompute")
        document = load_flag_stratified_cv(path=dest)
        _print_document(document)
        return

    train_path = get_train_csv_path()
    print(f"train_csv: {_repo_relative(train_path)}")
    frame = pd.read_csv(train_path)
    print(f"n_rows: {len(frame)}")
    print(f"n_splits: {args.n_splits}")
    print(f"random_state: {args.random_state}")
    print(f"alarm_flag_columns: {list(ALARM_FLAG_COLUMNS)}")

    document = build_flag_stratified_cv(
        frame,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    written = write_flag_stratified_cv(
        document,
        path=dest,
        overwrite=args.force,
    )
    print(f"wrote: {_repo_relative(written)}")
    _print_document(document)
    print("done")


def _print_document(document: FlagStratifiedCv) -> None:
    n_fraud = n_fraud_from_counts(document.stratum_counts)
    n_alarm = n_alarm_from_counts(document.stratum_counts)
    n_rows = document.n_rows
    n_alarm_fraud = document.stratum_counts["alarm_fraud"]
    base_rate = n_fraud / n_rows
    alarm_rate = n_alarm_fraud / n_alarm if n_alarm else 0.0
    print(f"scheme: {document.n_splits}-fold flag_stratified")
    print(f"n_fraud: {n_fraud} rate={base_rate:.4f}")
    print(
        f"n_alarm: {n_alarm} share_of_rows={n_alarm / n_rows:.4f} "
        f"share_of_fraud={n_alarm_fraud / n_fraud:.4f} "
        f"rate={alarm_rate:.4f} lift={alarm_rate / base_rate:.2f}"
    )
    print("global stratum counts")
    print(_counts_frame(document.stratum_counts).to_string(index=False))
    n_folds = len(document.fold_summaries)
    rows: list[dict[str, object] | None] = [None] * n_folds
    for i in range(n_folds):
        summary = document.fold_summaries[i]
        val_counts = summary.val_stratum_counts
        n_val_fraud = n_fraud_from_counts(val_counts)
        n_val_alarm = n_alarm_from_counts(val_counts)
        row: dict[str, object] = {
            "fold_id": summary.fold_id,
            "n_train": summary.n_train,
            "n_val": summary.n_val,
            "val_n_fraud": n_val_fraud,
            "val_n_alarm": n_val_alarm,
            "val_fraud_rate": n_val_fraud / summary.n_val,
            "val_alarm_rate": n_val_alarm / summary.n_val,
        }
        for name, count in val_counts.items():
            row[name] = count
        rows[i] = row
    filled_rows = [row for row in rows if row is not None]
    print("val occupancy by fold")
    print(pd.DataFrame(filled_rows).to_string(index=False))


def _counts_frame(counts: dict[str, int]) -> pd.DataFrame:
    names = list(counts)
    n_names = len(names)
    rows: list[dict[str, object] | None] = [None] * n_names
    for i in range(n_names):
        name = names[i]
        rows[i] = {"stratum": name, "n": counts[name]}
    filled = [row for row in rows if row is not None]
    return pd.DataFrame(filled)


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(get_repo_root()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    try:
        main()
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
