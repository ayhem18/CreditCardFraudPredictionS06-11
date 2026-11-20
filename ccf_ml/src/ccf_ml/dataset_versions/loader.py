"""Load dataset versions: name in, folds and frames out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.flag_stratified import (
    FlagStratifiedCv,
    parse_flag_stratified_cv,
)
from ccf_ml.cross_validation.stratified import Fold
from ccf_ml.dataset_versions.constants import (
    DATASET_VERSION_DATA_FILENAME,
    DATASET_VERSION_FOLDS_FILENAME,
    DATASET_VERSION_MANIFEST_FILENAME,
)
from ccf_ml.dataset_versions.manifest import (
    DatasetVersionManifest,
    load_dataset_version_manifest,
    sha256_file,
)
from ccf_ml.dataset_versions.specs import validate_version_name
from ccf_ml.file_utils import get_dataset_version_dir, get_dataset_versions_dir


class DatasetVersion(NamedTuple):
    """A loaded version: manifest, frozen folds, and the version frame."""

    manifest: DatasetVersionManifest
    folds: FlagStratifiedCv
    frame: pd.DataFrame
    dir: Path


class VersionFold(NamedTuple):
    """One fold of a dataset version: train frame and val frame."""

    fold_id: int
    train: pd.DataFrame
    val: pd.DataFrame


def dataset_version_exists(version_name: str) -> bool:
    """True when ``<dataset_versions>/<version_name>`` exists on disk."""
    name = validate_version_name(version_name)
    return (get_dataset_versions_dir() / name).is_dir()


def list_dataset_versions(
    *,
    versions_dir: Path | None = None,
) -> tuple[str, ...]:
    """Return sorted version names found under the versions directory."""
    base = versions_dir if versions_dir is not None else get_dataset_versions_dir()
    if not base.is_dir():
        return ()
    return tuple(sorted(entry.name for entry in base.iterdir() if entry.is_dir()))


def load_dataset_version(
    version_name: str,
    *,
    verify: bool = True,
    versions_dir: Path | None = None,
) -> DatasetVersion:
    """Load the version named ``version_name``.

    With ``verify`` (default), the sha256 of ``data.parquet`` and
    ``folds.json`` are checked against the manifest.
    """
    name = validate_version_name(version_name)
    version_dir = (
        Path(versions_dir) / name
        if versions_dir is not None
        else get_dataset_version_dir(name)
    )
    manifest = load_dataset_version_manifest(
        name, path=version_dir / DATASET_VERSION_MANIFEST_FILENAME
    )
    folds_path = version_dir / DATASET_VERSION_FOLDS_FILENAME
    folds_text = folds_path.read_text(encoding="utf-8")
    folds_payload = json.loads(folds_text)
    folds = parse_flag_stratified_cv(folds_payload)
    frame = pd.read_parquet(version_dir / DATASET_VERSION_DATA_FILENAME)
    _require_consistent(
        manifest=manifest,
        folds=folds,
        folds_scheme=folds_payload["scheme"],
        frame=frame,
    )
    if verify:
        data_path = version_dir / manifest.data_filename
        if sha256_file(data_path) != manifest.data_sha256:
            raise ValueError(
                f"data payload fingerprint mismatch for {name!r}: "
                f"{data_path}"
            )
        digest = sha256_file(folds_path)
        if digest != manifest.cv_sha256:
            raise ValueError(
                f"folds payload fingerprint mismatch for {name!r}: {folds_path}"
            )
    return DatasetVersion(
        manifest=manifest, folds=folds, frame=frame, dir=version_dir
    )


def iter_folds(
    version: DatasetVersion,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
) -> Iterator[VersionFold]:
    """Yield each fold's train/val frames from a loaded version."""
    for fold in version.folds.folds:
        yield _fold_frames(version, fold, columns=columns)


def load_fold(
    version: DatasetVersion,
    fold_id: int,
    *,
    columns: tuple[str, ...] | list[str] | None = None,
) -> VersionFold:
    """Return one fold's train/val frames by ``fold_id``."""
    fold = version.folds.fold(fold_id)
    return _fold_frames(version, fold, columns=columns)


def _fold_frames(
    version: DatasetVersion,
    fold: Fold,
    *,
    columns: tuple[str, ...] | list[str] | None,
) -> VersionFold:
    frame = version.frame
    if columns is None:
        selected = list(frame.columns)
    else:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(
                f"columns not in version {version.manifest.name!r}: {missing}"
            )
        selected = list(columns)
    ids = frame[version.manifest.id_column].to_numpy(dtype=np.int64)
    val_ids = np.asarray(fold.val_ids, dtype=np.int64)
    train_ids = np.asarray(fold.train_ids, dtype=np.int64)
    val_mask = np.isin(ids, val_ids)
    train_mask = np.isin(ids, train_ids)
    if np.any(train_mask & val_mask) or not np.all(train_mask | val_mask):
        raise ValueError(
            f"fold {fold.fold_id} of version {version.manifest.name!r} does "
            f"not partition the version frame"
        )
    return VersionFold(
        fold_id=fold.fold_id,
        train=frame.loc[train_mask, selected],
        val=frame.loc[val_mask, selected],
    )


def _require_consistent(
    *,
    manifest: DatasetVersionManifest,
    folds: FlagStratifiedCv,
    folds_scheme: str,
    frame: pd.DataFrame,
) -> None:
    """Manifest, frozen partition, and payload must agree with each other."""
    if manifest.id_column not in frame.columns:
        raise ValueError(
            f"version {manifest.name!r} payload is missing id column "
            f"{manifest.id_column!r}"
        )
    if manifest.target_column not in frame.columns:
        raise ValueError(
            f"version {manifest.name!r} payload is missing target column "
            f"{manifest.target_column!r}"
        )
    columns = tuple(frame.columns)
    if columns != manifest.kept_columns:
        raise ValueError(
            f"version {manifest.name!r} payload columns do not match "
            f"manifest kept_columns"
        )
    if len(frame) != manifest.n_rows or len(columns) != manifest.n_columns:
        raise ValueError(
            f"version {manifest.name!r} payload shape "
            f"{len(frame)}x{len(columns)} does not match manifest "
            f"n_rows={manifest.n_rows} n_columns={manifest.n_columns}"
        )
    if folds.n_splits != manifest.n_splits or folds_scheme != manifest.cv_scheme:
        raise ValueError(
            f"version {manifest.name!r} folds.json does not match manifest"
        )
