"""Profile x target stratified k-fold, frozen as a package artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.constants import ID_COLUMN, TARGET_COLUMN
from ccf_ml.cross_validation.stratified import (
    Fold,
    require_partition,
    stratified_kfold_ids,
)
from ccf_ml.file_utils import get_cv_splits_dir
from ccf_ml.region_modeling.constants import (
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    PROFILE_COLUMNS,
    PROFILE_CV_FILENAME,
    PROFILE_CV_MIN_STRATUM_PROFILE,
    PROFILE_CV_SCHEME,
    REGION_NAME,
)
from ccf_ml.region_modeling.profiles import profile_stratum

_DOCUMENT_KEYS = (
    "scheme",
    "region",
    "n_splits",
    "random_state",
    "shuffle",
    "id_column",
    "target_column",
    "profile_columns",
    "min_stratum_rows",
    "n_rows",
    "stratum_counts",
    "folds",
)
_FOLD_KEYS = ("fold_id", "n_train", "n_val", "stratum_counts_train", "stratum_counts_val", "val_ids")


class ProfileStratifiedCv(NamedTuple):
    """Frozen profile x target stratified k-fold partition of the region."""

    n_splits: int
    random_state: int
    shuffle: bool
    id_column: str
    target_column: str
    profile_columns: tuple[str, ...]
    min_stratum_rows: int
    n_rows: int
    stratum_counts: dict[str, int]
    folds: tuple[Fold, ...]

    def fold(self, fold_id: int) -> Fold:
        """Return the fold with id ``fold_id``."""
        for fold in self.folds:
            if fold.fold_id == fold_id:
                return fold
        raise KeyError(f"no fold with id {fold_id}")


class FoldSummary(NamedTuple):
    """Occupancy of one fold, derived from the profile x target stratum."""

    fold_id: int
    n_train: int
    n_val: int
    stratum_counts_train: dict[str, int]
    stratum_counts_val: dict[str, int]


def build_profile_stratified_cv(
    region_frame: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    shuffle: bool = True,
    min_stratum_rows: int = PROFILE_CV_MIN_STRATUM_PROFILE,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> ProfileStratifiedCv:
    """Split ``region_frame`` (already carrying ``profile_id``) into folds."""
    record_ids = region_frame[id_column].to_numpy(dtype=np.int64)
    stratum = profile_stratum(
        region_frame,
        min_stratum_rows=min_stratum_rows,
        id_column=id_column,
        target_column=target_column,
    ).to_numpy()
    counts = {str(key): int(value) for key, value in pd.Series(stratum).value_counts().items()}
    folds = stratified_kfold_ids(
        record_ids=record_ids,
        stratum=stratum,
        n_splits=n_splits,
        random_state=random_state if shuffle else None,
        shuffle=shuffle,
    )
    return ProfileStratifiedCv(
        n_splits=n_splits,
        random_state=random_state,
        shuffle=shuffle,
        id_column=id_column,
        target_column=target_column,
        profile_columns=PROFILE_COLUMNS,
        min_stratum_rows=min_stratum_rows,
        n_rows=len(record_ids),
        stratum_counts=counts,
        folds=folds,
    )


def fold_summaries(
    cv: ProfileStratifiedCv, region_frame: pd.DataFrame
) -> tuple[FoldSummary, ...]:
    """Per-fold stratum occupancy of ``cv`` against its region frame."""
    stratum = profile_stratum(
        region_frame,
        min_stratum_rows=cv.min_stratum_rows,
        id_column=cv.id_column,
        target_column=cv.target_column,
    )
    ids = region_frame[cv.id_column].to_numpy(dtype=np.int64)
    pos_by_id = pd.Series(np.arange(len(ids)), index=ids)
    summaries: list[FoldSummary] = []
    for fold in cv.folds:
        val_pos = pos_by_id.loc[list(fold.val_ids)].to_numpy(dtype=np.int64)
        train_pos = pos_by_id.loc[list(fold.train_ids)].to_numpy(dtype=np.int64)
        summaries.append(
            FoldSummary(
                fold_id=fold.fold_id,
                n_train=len(train_pos),
                n_val=len(val_pos),
                stratum_counts_train={
                    str(k): int(v)
                    for k, v in stratum.iloc[train_pos].value_counts().items()
                },
                stratum_counts_val={
                    str(k): int(v)
                    for k, v in stratum.iloc[val_pos].value_counts().items()
                },
            )
        )
    return tuple(summaries)


def dump_profile_stratified_cv(cv: ProfileStratifiedCv) -> dict[str, object]:
    """JSON-ready document for the frozen partition."""
    return {
        "scheme": PROFILE_CV_SCHEME,
        "region": REGION_NAME,
        "n_splits": cv.n_splits,
        "random_state": cv.random_state,
        "shuffle": cv.shuffle,
        "id_column": cv.id_column,
        "target_column": cv.target_column,
        "profile_columns": list(cv.profile_columns),
        "min_stratum_rows": cv.min_stratum_rows,
        "n_rows": cv.n_rows,
        "stratum_counts": cv.stratum_counts,
        "folds": [
            {
                "fold_id": fold.fold_id,
                "n_train": len(fold.train_ids),
                "n_val": len(fold.val_ids),
                "stratum_counts_train": {},
                "stratum_counts_val": {},
                "val_ids": list(fold.val_ids),
            }
            for fold in cv.folds
        ],
    }


def parse_profile_stratified_cv(payload: object) -> ProfileStratifiedCv:
    """Rebuild the frozen partition from a JSON object."""
    if type(payload) is not dict:
        raise TypeError("profile cv document must be a dict")
    if set(payload) != set(_DOCUMENT_KEYS):
        raise ValueError("profile cv document keys do not match the scheme")
    if payload["scheme"] != PROFILE_CV_SCHEME:
        raise ValueError(f"unexpected scheme {payload['scheme']!r}")
    folds_raw = payload["folds"]
    if type(folds_raw) is not list or not folds_raw:
        raise ValueError("folds must be a non-empty list")
    folds: list[Fold] = []
    for i, fold_payload in enumerate(folds_raw):
        if set(fold_payload) != set(_FOLD_KEYS):
            raise ValueError(f"fold {i} keys do not match the scheme")
        if fold_payload["fold_id"] != i:
            raise ValueError(f"fold_id {fold_payload['fold_id']!r} does not match index {i}")
        folds.append(
            Fold(
                fold_id=int(fold_payload["fold_id"]),
                train_ids=(),
                val_ids=tuple(int(x) for x in fold_payload["val_ids"]),
            )
        )
    cv = ProfileStratifiedCv(
        n_splits=int(payload["n_splits"]),
        random_state=int(payload["random_state"]),
        shuffle=bool(payload["shuffle"]),
        id_column=str(payload["id_column"]),
        target_column=str(payload["target_column"]),
        profile_columns=tuple(str(c) for c in payload["profile_columns"]),
        min_stratum_rows=int(payload["min_stratum_rows"]),
        n_rows=int(payload["n_rows"]),
        stratum_counts={str(k): int(v) for k, v in payload["stratum_counts"].items()},
        folds=tuple(folds),
    )
    return _rebuild_train_ids(cv)


def _rebuild_train_ids(cv: ProfileStratifiedCv) -> ProfileStratifiedCv:
    """Return the cv with each fold's train_ids as the complement of the
    other folds' val lists."""
    all_ids = sorted({int(x) for fold in cv.folds for x in fold.val_ids})
    if len(all_ids) != cv.n_rows:
        raise ValueError(
            f"val ids count {len(all_ids)} does not match n_rows {cv.n_rows}"
        )
    val_sets = [set(fold.val_ids) for fold in cv.folds]
    rebuilt = []
    for i, fold in enumerate(cv.folds):
        others = set().union(*(val_sets[j] for j in range(len(val_sets)) if j != i))
        rebuilt.append(
            Fold(
                fold_id=fold.fold_id,
                train_ids=tuple(sorted(others)),
                val_ids=fold.val_ids,
            )
        )
    record_ids = np.array(all_ids, dtype=np.int64)
    require_partition(tuple(rebuilt), record_ids=record_ids)
    return cv._replace(folds=tuple(rebuilt))


def dump_to_path(cv: ProfileStratifiedCv, *, path: Path | None = None) -> Path:
    """Write the frozen partition JSON (compact, NaN-free)."""
    dest = path if path is not None else get_cv_splits_dir() / PROFILE_CV_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(dump_profile_stratified_cv(cv), separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return dest


def load_profile_stratified_cv(*, path: Path | None = None) -> ProfileStratifiedCv:
    """Load the frozen partition from disk."""
    source = path if path is not None else get_cv_splits_dir() / PROFILE_CV_FILENAME
    return parse_profile_stratified_cv(json.loads(source.read_text(encoding="utf-8")))
