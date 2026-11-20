"""Build the preset dataset versions on the frozen flag-stratified folds.

Usage::

    python scripts/build_dataset_versions.py
    python scripts/build_dataset_versions.py --version no_time
    python scripts/build_dataset_versions.py --overwrite
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from ccf_ml.cross_validation.flag_stratified import load_flag_stratified_cv
from ccf_ml.dataset_versions.builder import build_dataset_version
from ccf_ml.dataset_versions.loader import (
    list_dataset_versions,
    load_dataset_version,
)
from ccf_ml.dataset_versions.specs import DATASET_VERSION_SPECS
from ccf_ml.file_utils import get_repo_root, get_train_csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="append",
        choices=[spec.name for spec in DATASET_VERSION_SPECS],
        help="Build only this preset (repeatable). Default: all presets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing version directories.",
    )
    args = parser.parse_args()

    names = args.version or [spec.name for spec in DATASET_VERSION_SPECS]
    specs = [spec for spec in DATASET_VERSION_SPECS if spec.name in names]

    folds = load_flag_stratified_cv()
    print(f"folds: {folds.n_splits}-fold {folds.id_column}/"
          f"{folds.target_column} flag_stratified, n_rows={folds.n_rows}")
    frame = pd.read_csv(get_train_csv_path())
    print(f"train: {len(frame)} rows, {len(frame.columns)} columns")

    for spec in specs:
        print(f"\n[{spec.name}]")
        print(f"  drop: {list(spec.drop_columns)}")
        version_dir = build_dataset_version(
            frame, spec, folds=folds, overwrite=args.overwrite
        )
        version = load_dataset_version(spec.name)
        print(f"  kept columns: {len(version.manifest.kept_columns)}")
        print(f"  rows: {version.manifest.n_rows}")
        print(f"  data sha256: {version.manifest.data_sha256}")
        print(f"  dir: {_repo_relative(version_dir)}")

    print(f"\nversions on disk: {list(list_dataset_versions())}")
    print("done")


def _repo_relative(path) -> str:
    try:
        return str(path.resolve().relative_to(get_repo_root()))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    try:
        main()
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        print("pass --overwrite to replace existing versions", file=sys.stderr)
        raise SystemExit(1) from exc
