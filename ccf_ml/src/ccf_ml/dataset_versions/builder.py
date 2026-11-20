"""Build a dataset version: drop columns, freeze folds, write payloads.

A version directory contains three files::

    manifest.json   # what this version is + sha256 fingerprints
    folds.json      # the frozen flag-stratified partition (same format as the
                    # package cv_splits artifact), self-contained per version
    data.parquet    # id + target + feature columns minus the spec's drops

Folds are not part of the spec: every version is built on the same frozen
partition so ablations are comparable fold by fold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.alarm import as_record_ids
from ccf_ml.cross_validation.constants import (
    FLAG_STRATIFIED_SCHEME,
    ID_COLUMN,
    TARGET_COLUMN,
)
from ccf_ml.cross_validation.flag_stratified import (
    FlagStratifiedCv,
    dump_flag_stratified_cv,
    load_flag_stratified_cv,
)
from ccf_ml.dataset_versions.constants import (
    DATASET_VERSION_DATA_FILENAME,
    DATASET_VERSION_FOLDS_FILENAME,
    DATASET_VERSION_MANIFEST_FILENAME,
    DATASET_VERSION_SCHEME,
)
from ccf_ml.dataset_versions.manifest import (
    DatasetVersionManifest,
    sha256_file,
    write_dataset_version_manifest,
)
from ccf_ml.dataset_versions.specs import (
    DatasetVersionSpec,
    kept_columns_from,
    validate_spec,
    validate_version_name,
)
from ccf_ml.dataset_versions.transforms import apply_transform
from ccf_ml.file_utils import (
    claim_dataset_version_dir,
    get_dataset_versions_dir,
)


def build_dataset_version(
    frame: pd.DataFrame,
    spec: DatasetVersionSpec,
    *,
    folds: FlagStratifiedCv | None = None,
    overwrite: bool = False,
    versions_dir: Path | None = None,
) -> Path:
    """Materialize ``spec`` from ``frame``. Return the version directory.

    ``folds`` defaults to the frozen flag-stratified partition. The frame's
    ids must match the partition's id universe exactly. Writes the three
    version files; refuses an existing version unless ``overwrite`` is set.
    """
    drop_columns = validate_spec(spec, frame_columns=tuple(frame.columns))
    if folds is None:
        folds = load_flag_stratified_cv()
    if folds.id_column != ID_COLUMN or folds.target_column != TARGET_COLUMN:
        raise ValueError(
            f"frozen partition must use id column {ID_COLUMN!r} and target "
            f"column {TARGET_COLUMN!r}, got {folds.id_column!r} and "
            f"{folds.target_column!r}"
        )
    record_ids = as_record_ids(frame[ID_COLUMN], column=ID_COLUMN)
    _require_partition_match(record_ids=record_ids, folds=folds)

    if spec.transform:
        working_frame, extras = apply_transform(spec.transform, frame, folds)
    else:
        working_frame, extras = frame, {}
    kept_columns = kept_columns_from(tuple(working_frame.columns), drop_columns)
    version_frame = working_frame.loc[:, list(kept_columns)]

    version_dir = _claim_version_dir(
        spec.name, overwrite=overwrite, versions_dir=versions_dir
    )

    folds_text = json.dumps(
        dump_flag_stratified_cv(folds),
        separators=(",", ":"),
        allow_nan=False,
    )
    folds_path = version_dir / DATASET_VERSION_FOLDS_FILENAME
    folds_path.write_text(folds_text, encoding="utf-8")

    data_path = version_dir / DATASET_VERSION_DATA_FILENAME
    version_frame.to_parquet(data_path, index=False)
    for filename, extra_frame in extras.items():
        extra_frame.to_parquet(version_dir / filename, index=False)

    manifest = DatasetVersionManifest(
        scheme=DATASET_VERSION_SCHEME,
        name=spec.name,
        description=spec.description,
        drop_columns=drop_columns,
        kept_columns=kept_columns,
        id_column=ID_COLUMN,
        target_column=TARGET_COLUMN,
        n_rows=len(version_frame),
        n_columns=len(kept_columns),
        n_splits=folds.n_splits,
        cv_scheme=FLAG_STRATIFIED_SCHEME,
        cv_filename=DATASET_VERSION_FOLDS_FILENAME,
        cv_sha256=sha256_file(folds_path),
        data_filename=DATASET_VERSION_DATA_FILENAME,
        data_sha256=sha256_file(data_path),
    )
    write_dataset_version_manifest(
        manifest,
        path=version_dir / DATASET_VERSION_MANIFEST_FILENAME,
        overwrite=overwrite,
    )
    return version_dir


def _require_partition_match(
    *,
    record_ids: np.ndarray,
    folds: FlagStratifiedCv,
) -> None:
    """Check the frame's ids are exactly the partition's id universe."""
    if folds.n_rows != len(record_ids):
        raise ValueError(
            f"frame has {len(record_ids)} rows; frozen partition has "
            f"n_rows={folds.n_rows}"
        )
    universe: set[int] = set()
    for fold in folds.folds:
        universe.update(fold.val_ids)
    if len(universe) != len(record_ids) or not np.array_equal(
        np.sort(record_ids), np.array(sorted(universe), dtype=np.int64)
    ):
        raise ValueError(
            "frame ids do not match the frozen partition's id universe"
        )


def _claim_version_dir(
    version_name: str,
    *,
    overwrite: bool,
    versions_dir: Path | None,
) -> Path:
    """Create (or reuse when ``overwrite``) the version directory."""
    name = validate_version_name(version_name)
    if versions_dir is not None:
        base = Path(versions_dir)
        base.mkdir(parents=True, exist_ok=True)
        path = base / name
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"dataset version {name!r} already exists: {path}"
            )
        path.mkdir(parents=False, exist_ok=True)
        return path
    path = get_dataset_versions_dir() / name
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"dataset version {name!r} already exists: {path}"
            )
        path.mkdir(parents=False, exist_ok=True)
        return path
    return claim_dataset_version_dir(name)
