"""Alarm x profile x target stratified k-fold over ALL rows, frozen artifact.

One fold assignment for both modeling approaches. A global model trains on
fold ``i``'s train rows (all rows); a no-alarm region model trains on the
no-alarm subset of those train rows and validates on the no-alarm subset of
fold ``i``'s val rows (:func:`restrict_to_no_alarm`). Because every row
carries the same fold id in both lines, their OOF tables are directly
comparable and per-row blends of the two OOF vectors are fold-coherent
(both predictions of a row come from models that never saw it).

The stratum is ``alarm|profile_id|target`` - 2 x 90 x 2 = 360 theoretical
cells. Two inheritance properties hold by construction:

- Aggregated to alarm x target (drop the profile field) the stratum is the
  flag-stratified scheme's stratum, so the alarm-region / fraud balance of
  every fold matches what the frozen flag-stratified CV guarantees.
- Within the no-alarm region (alarm constant) the stratum is the profile x
  target stratum with the same per-cell counts as the region-only scheme,
  so small-cell collapse decisions are identical there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.alarm import alarm_flag_mask
from ccf_ml.cross_validation.constants import (
    ALARM_FLAG_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
)
from ccf_ml.cross_validation.stratified import (
    Fold,
    require_partition,
    stratified_kfold_ids,
)
from ccf_ml.file_utils import get_cv_splits_dir
from ccf_ml.region_modeling.constants import (
    ALARM_LABEL,
    ALARM_PROFILE_CV_FILENAME,
    ALARM_PROFILE_CV_MIN_STRATUM,
    ALARM_PROFILE_CV_SCHEME,
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    NO_ALARM_LABEL,
    PROFILE_COLUMNS,
    SMALL_PROFILE_LABEL,
)
from ccf_ml.region_modeling.profile_cv import FoldSummary, ProfileStratifiedCv
from ccf_ml.region_modeling.profiles import profile_stratum

_DOCUMENT_KEYS = (
    "scheme",
    "region",
    "n_splits",
    "random_state",
    "shuffle",
    "id_column",
    "target_column",
    "alarm_flag_columns",
    "profile_columns",
    "min_stratum_rows",
    "n_rows",
    "stratum_counts",
    "folds",
)
_FOLD_KEYS = ("fold_id", "n_train", "n_val", "val_ids")
_ALL_REGIONS = "all"


class AlarmProfileStratifiedCv(NamedTuple):
    """Frozen alarm x profile x target stratified k-fold of every row."""

    n_splits: int
    random_state: int
    shuffle: bool
    id_column: str
    target_column: str
    alarm_flag_columns: tuple[str, ...]
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


def alarm_profile_stratum(
    frame: pd.DataFrame,
    *,
    min_stratum_rows: int,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
    alarm_flag_columns: tuple[str, ...] = ALARM_FLAG_COLUMNS,
) -> pd.Series:
    """Stratum labels for the unified CV.

    Labels are ``alarm|profile_id|target``. A cell with fewer than
    ``min_stratum_rows`` rows collapses to ``alarm|_small_|target`` - only
    the profile field is dropped, so the alarm x target balance every fold
    inherits is exactly the flag-stratified scheme's.
    """
    has_alarm = alarm_flag_mask(frame, columns=alarm_flag_columns)
    alarm_name = has_alarm.map({True: ALARM_LABEL, False: NO_ALARM_LABEL})
    target_name = frame[target_column].astype(str)
    labels = alarm_name + "|" + frame["profile_id"].astype(str) + "|" + target_name
    counts = labels.value_counts()
    small = counts.index[counts < min_stratum_rows]
    return labels.where(~labels.isin(small), alarm_name + "|" + SMALL_PROFILE_LABEL + "|" + target_name)


def build_alarm_profile_stratified_cv(
    frame: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    shuffle: bool = True,
    min_stratum_rows: int = ALARM_PROFILE_CV_MIN_STRATUM,
    id_column: str = ID_COLUMN,
    target_column: str = TARGET_COLUMN,
    alarm_flag_columns: tuple[str, ...] = ALARM_FLAG_COLUMNS,
) -> AlarmProfileStratifiedCv:
    """Split ``frame`` (already carrying ``profile_id``) into folds."""
    record_ids = frame[id_column].to_numpy(dtype=np.int64)
    stratum = alarm_profile_stratum(
        frame,
        min_stratum_rows=min_stratum_rows,
        id_column=id_column,
        target_column=target_column,
        alarm_flag_columns=alarm_flag_columns,
    ).to_numpy()
    counts = {str(key): int(value) for key, value in pd.Series(stratum).value_counts().items()}
    folds = stratified_kfold_ids(
        record_ids=record_ids,
        stratum=stratum,
        n_splits=n_splits,
        random_state=random_state if shuffle else None,
        shuffle=shuffle,
    )
    return AlarmProfileStratifiedCv(
        n_splits=n_splits,
        random_state=random_state,
        shuffle=shuffle,
        id_column=id_column,
        target_column=target_column,
        alarm_flag_columns=tuple(alarm_flag_columns),
        profile_columns=PROFILE_COLUMNS,
        min_stratum_rows=min_stratum_rows,
        n_rows=len(record_ids),
        stratum_counts=counts,
        folds=folds,
    )


def alarm_target_totals(stratum_counts: dict[str, int]) -> dict[str, int]:
    """Aggregate stratum labels into the four alarm x target cells.

    Returned keys are the flag-stratified names (``no_alarm_not_fraud`` ...),
    so a unified fold table can be compared with the frozen flag-stratified
    artifact directly.
    """
    totals = {
        "no_alarm_not_fraud": 0,
        "no_alarm_fraud": 0,
        "alarm_not_fraud": 0,
        "alarm_fraud": 0,
    }
    for label, count in stratum_counts.items():
        fields = str(label).split("|")
        alarm_name = fields[0]
        if alarm_name == ALARM_LABEL:
            prefix = "alarm"
        elif alarm_name == NO_ALARM_LABEL:
            prefix = "no_alarm"
        else:
            raise ValueError(f"stratum label {label!r} does not start with an alarm bit")
        target = float(fields[-1])
        if target == 1.0:
            suffix = "fraud"
        elif target == 0.0:
            suffix = "not_fraud"
        else:
            raise ValueError(f"stratum label {label!r} does not end with a binary target")
        totals[f"{prefix}_{suffix}"] += int(count)
    return totals


def fold_summaries(
    cv: AlarmProfileStratifiedCv, frame: pd.DataFrame
) -> tuple[FoldSummary, ...]:
    """Per-fold stratum occupancy of ``cv`` against its (profiled) frame."""
    stratum = alarm_profile_stratum(
        frame,
        min_stratum_rows=cv.min_stratum_rows,
        id_column=cv.id_column,
        target_column=cv.target_column,
        alarm_flag_columns=cv.alarm_flag_columns,
    )
    ids = frame[cv.id_column].to_numpy(dtype=np.int64)
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


def restrict_to_no_alarm(
    cv: AlarmProfileStratifiedCv, region_frame: pd.DataFrame
) -> ProfileStratifiedCv:
    """The no-alarm partition induced by ``cv`` on ``region_frame`` rows.

    Each region row keeps its unified fold id: fold ``i``'s val list is the
    region rows that fold ``i`` holds, its train list the region rows of the
    other folds. The result is a :class:`ProfileStratifiedCv`, so the region
    variant ladder (including the fold-safe within-profile quantiles) runs
    on it unchanged. Stratum counts come from the region-only profile x
    target labels - identical collapse decisions, since alarm is part of
    the unified stratum key.
    """
    ids = region_frame[cv.id_column].to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("region frame has duplicate ids")
    universe = {int(x) for fold in cv.folds for x in fold.val_ids}
    unknown = sorted({int(x) for x in ids} - universe)
    if unknown:
        raise ValueError(
            f"region frame has {len(unknown)} ids outside the unified cv, "
            f"e.g. {unknown[:5]}"
        )
    stratum = profile_stratum(
        region_frame,
        min_stratum_rows=cv.min_stratum_rows,
        id_column=cv.id_column,
        target_column=cv.target_column,
    )
    counts = {str(k): int(v) for k, v in stratum.value_counts().items()}
    region_set = {int(x) for x in ids}
    folds: list[Fold] = []
    for fold in cv.folds:
        val_ids = tuple(x for x in fold.val_ids if x in region_set)
        train_ids = tuple(sorted(region_set.difference(val_ids)))
        folds.append(Fold(fold_id=fold.fold_id, train_ids=train_ids, val_ids=val_ids))
    partition = tuple(folds)
    require_partition(partition, record_ids=ids)
    return ProfileStratifiedCv(
        n_splits=cv.n_splits,
        random_state=cv.random_state,
        shuffle=cv.shuffle,
        id_column=cv.id_column,
        target_column=cv.target_column,
        profile_columns=cv.profile_columns,
        min_stratum_rows=cv.min_stratum_rows,
        n_rows=len(ids),
        stratum_counts=counts,
        folds=partition,
    )


def alarm_profile_cv_path() -> Path:
    """Return ``<package_artifacts>/cv_splits/alarm_profile_stratified_cv.json``."""
    return get_cv_splits_dir() / ALARM_PROFILE_CV_FILENAME


def dump_alarm_profile_stratified_cv(cv: AlarmProfileStratifiedCv) -> dict[str, object]:
    """JSON-ready document for the frozen partition."""
    return {
        "scheme": ALARM_PROFILE_CV_SCHEME,
        "region": _ALL_REGIONS,
        "n_splits": cv.n_splits,
        "random_state": cv.random_state,
        "shuffle": cv.shuffle,
        "id_column": cv.id_column,
        "target_column": cv.target_column,
        "alarm_flag_columns": list(cv.alarm_flag_columns),
        "profile_columns": list(cv.profile_columns),
        "min_stratum_rows": cv.min_stratum_rows,
        "n_rows": cv.n_rows,
        "stratum_counts": cv.stratum_counts,
        "folds": [
            {
                "fold_id": fold.fold_id,
                "n_train": len(fold.train_ids),
                "n_val": len(fold.val_ids),
                "val_ids": list(fold.val_ids),
            }
            for fold in cv.folds
        ],
    }


def parse_alarm_profile_stratified_cv(payload: object) -> AlarmProfileStratifiedCv:
    """Rebuild the frozen partition from a JSON object."""
    if type(payload) is not dict:
        raise TypeError("alarm profile cv document must be a dict")
    if set(payload) != set(_DOCUMENT_KEYS):
        raise ValueError("alarm profile cv document keys do not match the scheme")
    if payload["scheme"] != ALARM_PROFILE_CV_SCHEME:
        raise ValueError(f"unexpected scheme {payload['scheme']!r}")
    if payload["region"] != _ALL_REGIONS:
        raise ValueError(f"unexpected region {payload['region']!r}")
    n_splits = payload["n_splits"]
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError(f"n_splits must be an int >= 2, got {n_splits!r}")
    random_state = payload["random_state"]
    if type(random_state) is not int:
        raise TypeError(f"random_state must be an int, got {random_state!r}")
    shuffle = payload["shuffle"]
    if type(shuffle) is not bool:
        raise TypeError(f"shuffle must be a bool, got {shuffle!r}")
    n_rows = payload["n_rows"]
    if type(n_rows) is not int or n_rows <= 0:
        raise ValueError(f"n_rows must be a positive int, got {n_rows!r}")
    min_stratum_rows = payload["min_stratum_rows"]
    if type(min_stratum_rows) is not int or min_stratum_rows < n_splits:
        raise ValueError(
            f"min_stratum_rows must be an int >= n_splits, got {min_stratum_rows!r}"
        )
    id_column = payload["id_column"]
    target_column = payload["target_column"]
    if type(id_column) is not str or type(target_column) is not str:
        raise TypeError("id_column and target_column must be strings")
    alarm_flag_columns = tuple(payload["alarm_flag_columns"])
    profile_columns = tuple(payload["profile_columns"])
    if not all(type(name) is str for name in alarm_flag_columns + profile_columns):
        raise TypeError("column lists must contain strings")
    stratum_counts_raw = payload["stratum_counts"]
    if type(stratum_counts_raw) is not dict:
        raise TypeError("stratum_counts must be a dict")
    stratum_counts: dict[str, int] = {}
    for key, value in stratum_counts_raw.items():
        if type(key) is not str or type(value) is not int or value < 0:
            raise ValueError(f"bad stratum_counts entry {key!r}: {value!r}")
        stratum_counts[key] = value
    if sum(stratum_counts.values()) != n_rows:
        raise ValueError("stratum_counts do not sum to n_rows")
    alarm_target_totals(stratum_counts)
    folds_raw = payload["folds"]
    if type(folds_raw) is not list or len(folds_raw) != n_splits:
        raise ValueError(f"folds must be a list of {n_splits} entries")
    folds: list[Fold] = []
    for i, fold_payload in enumerate(folds_raw):
        if type(fold_payload) is not dict:
            raise TypeError(f"folds[{i}] must be a dict")
        if set(fold_payload) != set(_FOLD_KEYS):
            raise ValueError(f"fold {i} keys do not match the scheme")
        if fold_payload["fold_id"] != i:
            raise ValueError(f"fold_id {fold_payload['fold_id']!r} does not match index {i}")
        val_ids_raw = fold_payload["val_ids"]
        if type(val_ids_raw) is not list:
            raise TypeError(f"folds[{i}].val_ids must be a list")
        if not all(type(x) is int for x in val_ids_raw):
            raise TypeError(f"folds[{i}].val_ids must contain ints")
        folds.append(
            Fold(
                fold_id=int(fold_payload["fold_id"]),
                train_ids=(),
                val_ids=tuple(val_ids_raw),
            )
        )
    universe = sorted({int(x) for fold in folds for x in fold.val_ids})
    if len(universe) != n_rows:
        raise ValueError(f"val ids count {len(universe)} does not match n_rows {n_rows}")
    val_sets = [set(fold.val_ids) for fold in folds]
    rebuilt: list[Fold] = []
    for i, fold in enumerate(folds):
        others = set().union(*(val_sets[j] for j in range(len(val_sets)) if j != i))
        rebuilt.append(
            Fold(
                fold_id=fold.fold_id,
                train_ids=tuple(sorted(others)),
                val_ids=fold.val_ids,
            )
        )
    partition = tuple(rebuilt)
    require_partition(partition, record_ids=np.array(universe, dtype=np.int64))
    return AlarmProfileStratifiedCv(
        n_splits=n_splits,
        random_state=random_state,
        shuffle=shuffle,
        id_column=id_column,
        target_column=target_column,
        alarm_flag_columns=alarm_flag_columns,
        profile_columns=profile_columns,
        min_stratum_rows=min_stratum_rows,
        n_rows=n_rows,
        stratum_counts=stratum_counts,
        folds=partition,
    )


def write_alarm_profile_stratified_cv(
    cv: AlarmProfileStratifiedCv,
    *,
    path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write the frozen partition. Refuses to replace an existing file unless asked."""
    dest = path if path is not None else alarm_profile_cv_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"cv splits already exist at {dest}. Pass overwrite=True to replace them."
        )
    dest.write_text(
        json.dumps(
            dump_alarm_profile_stratified_cv(cv),
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return dest


def load_alarm_profile_stratified_cv(*, path: Path | None = None) -> AlarmProfileStratifiedCv:
    """Load the frozen partition from disk."""
    source = path if path is not None else alarm_profile_cv_path()
    return parse_alarm_profile_stratified_cv(json.loads(source.read_text(encoding="utf-8")))
