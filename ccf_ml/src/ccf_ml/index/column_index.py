"""Per-column inverted index: a key maps to the record ids that share it.

Context
-------
Finding rows with a similar value on one column is a dictionary lookup if
the column is already inverted. A linear scan of the frame is the thing
this object replaces. Intersection across columns is a later AND of the
id sets this index returns.

A categorical key is the category value. A numeric column with fewer than
``RAW_UNIQUE_LIMIT`` distinct values uses those values as keys. Otherwise
the key is a percentile range. ``[1, 2[`` is the set of values whose
empirical percentile is in ``[1, 2)``: from the 1st percentile, up to the
2nd, 2nd excluded. The last occupied bin still uses that half-open label;
the column maximum is assigned to ``[99, 100[``.

Terminology
-----------
``PercentileRange`` is the numeric key ``[start, stop[``.
``Keying`` is how one column chooses keys: ``CATEGORICAL``,
``NUMERIC_RAW``, or ``NUMERIC_PERCENTILE``.
``ColumnRole`` is the caller's claim: ``CATEGORICAL`` or ``NUMERIC``.
``ColumnIndex`` is one column's postings. ``FrameIndex`` is one
``ColumnIndex`` per named column, over the same ids.
``ids_for`` returns the posting. ``key_of`` maps a value to a key.
``ids_for_value`` is ``ids_for(key_of(value))``.
``ids_matching`` is the AND of ``ids_for_value`` across named columns.

Input Specification
-------------------
The frame has a unique integer id column and the named feature columns.
Those feature columns have no NA. Numeric columns are finite. A column
name appears in at most one of ``categorical_columns`` and
``numeric_columns``. ``ColumnRole`` is supplied by the caller; unique
count then chooses raw values versus percentile ranges.

Main functionalities
--------------------
``column_index_from_frame`` / ``frame_index_from_frame`` parse. After
parse, ``ids_for``, ``key_of``, ``ids_for_value``, and ``ids_matching``
do not rebuild the postings.

Secondary functionalities
-------------------------
Ids in a posting are a ``frozenset`` of Python ``int``. Empty bins are
omitted. A missing key returns an empty frozenset. ``ids_matching``
intersects those sets, smallest posting first. Percentile edges are
those of the indexed frame. A value outside ``[min, max]`` has no key.
Percentile keys have width 1.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ccf_ml.cross_validation.alarm import as_record_ids

RAW_UNIQUE_LIMIT = 100
PERCENTILE_RANK_MAX = 100
PERCENTILE_EDGES = PERCENTILE_RANK_MAX + 1


class ColumnRole(Enum):
    """Caller claim for one column: category values, or numeric keying."""

    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


class Keying(Enum):
    """How ``ColumnIndex`` keys are formed after parse."""

    CATEGORICAL = "categorical"
    NUMERIC_RAW = "numeric_raw"
    NUMERIC_PERCENTILE = "numeric_percentile"


class PercentileRange(NamedTuple):
    """Half-open percentile key ``[start, stop[``."""

    start: int
    stop: int

    def __str__(self) -> str:
        return f"[{self.start}, {self.stop}["


class ColumnIndex(NamedTuple):
    """Inverted index of one column: key to a set of record ids."""

    column: str
    keying: Keying
    postings: dict[object, frozenset[int]]
    percentile_edges: np.ndarray | None

    def keys(self) -> tuple[object, ...]:
        """Keys that have at least one id."""
        return tuple(self.postings.keys())

    def ids_for(self, key: object) -> frozenset[int]:
        """Ids for ``key``. Empty if that key has no posting."""
        resolved = _resolve_key(key, keying=self.keying)
        found = self.postings.get(resolved)
        if found is None:
            return frozenset()
        return found

    def key_of(self, value: object) -> object:
        """Key that would hold ``value`` in this index."""
        if self.keying is Keying.NUMERIC_PERCENTILE:
            if self.percentile_edges is None:
                raise ValueError(
                    f"column {self.column!r} is percentile-keyed but has no edges"
                )
            return percentile_range_of(
                _as_finite_float(value),
                edges=self.percentile_edges,
            )
        return _python_key(value)

    def ids_for_value(self, value: object) -> frozenset[int]:
        """Ids that share a key with ``value``."""
        return self.ids_for(self.key_of(value))


class FrameIndex(NamedTuple):
    """One ``ColumnIndex`` per named column, over the same record ids."""

    ids: frozenset[int]
    columns: tuple[ColumnIndex, ...]

    def column_index(self, column: str) -> ColumnIndex:
        """Return the index for ``column``."""
        n_columns = len(self.columns)
        for i in range(n_columns):
            item = self.columns[i]
            if item.column == column:
                return item
        raise KeyError(f"no index for column {column!r}")

    def ids_for(self, column: str, key: object) -> frozenset[int]:
        """Ids for ``key`` on ``column``."""
        return self.column_index(column).ids_for(key)

    def ids_for_value(self, column: str, value: object) -> frozenset[int]:
        """Ids that share a key with ``value`` on ``column``."""
        return self.column_index(column).ids_for_value(value)

    def ids_matching(self, *, values: dict[str, object]) -> frozenset[int]:
        """Ids that match every named column value.

        Each name must be a column in this index. The result is the
        intersection of ``ids_for_value`` postings, smallest posting first.
        """
        n_keys = len(values)
        if n_keys == 0:
            raise ValueError("values is empty")
        names = tuple(values.keys())
        parts: list[frozenset[int] | None] = [None] * n_keys
        for i in range(n_keys):
            name = names[i]
            parts[i] = self.ids_for_value(name, values[name])
        filled = [part for part in parts if part is not None]
        if len(filled) != n_keys:
            raise ValueError("failed to read every posting")
        decorated: list[tuple[int, int] | None] = [None] * n_keys
        for i in range(n_keys):
            decorated[i] = (len(filled[i]), i)
        filled_dec = [item for item in decorated if item is not None]
        if len(filled_dec) != n_keys:
            raise ValueError("failed to score postings")
        filled_dec.sort()
        ordered: list[frozenset[int] | None] = [None] * n_keys
        for i in range(n_keys):
            ordered[i] = filled[filled_dec[i][1]]
        sorted_parts = [part for part in ordered if part is not None]
        if len(sorted_parts) != n_keys:
            raise ValueError("failed to order postings")
        return _intersect_sets(tuple(sorted_parts))


def _intersect_sets(parts: tuple[frozenset[int], ...]) -> frozenset[int]:
    n_parts = len(parts)
    if n_parts == 0:
        raise ValueError("parts is empty")
    out = parts[0]
    for i in range(1, n_parts):
        if not out:
            return out
        out = out & parts[i]
    return out


def parse_percentile_range(*, start: int, stop: int) -> PercentileRange:
    """Build a percentile key. ``start < stop``, both in ``[0, 100]``."""
    start_int = _as_rank(start, name="start")
    stop_int = _as_rank(stop, name="stop")
    if start_int >= stop_int:
        raise ValueError(
            f"percentile range must have start < stop (got {start_int}, {stop_int})"
        )
    if stop_int > PERCENTILE_RANK_MAX:
        raise ValueError(
            f"percentile stop must be <= {PERCENTILE_RANK_MAX}, got {stop_int}"
        )
    return PercentileRange(start=start_int, stop=stop_int)


def percentile_range_of(value: float, *, edges: np.ndarray) -> PercentileRange:
    """Percentile key of ``value`` given length-101 edges of the indexed frame."""
    _require_percentile_edges(edges)
    if value < edges[0] or value > edges[-1]:
        raise ValueError(
            f"value {value} is outside the indexed range [{edges[0]}, {edges[-1]}]"
        )
    rank = int(np.searchsorted(edges, value, side="right") - 1)
    if rank < 0:
        rank = 0
    last_start = PERCENTILE_RANK_MAX - 1
    if rank > last_start:
        rank = last_start
    return PercentileRange(start=rank, stop=rank + 1)


def column_index_from_frame(
    frame: pd.DataFrame,
    *,
    column: str,
    id_column: str,
    role: ColumnRole,
) -> ColumnIndex:
    """Invert ``column`` of ``frame`` under ``role``."""
    if column not in frame.columns:
        raise ValueError(f"frame is missing column: {column!r}")
    if id_column not in frame.columns:
        raise ValueError(f"frame is missing id column: {id_column!r}")
    if len(frame) == 0:
        raise ValueError("frame is empty")
    series = frame[column]
    if series.isna().any():
        raise ValueError(f"column {column!r} contains NA")
    ids = as_record_ids(frame[id_column], column=id_column)
    if role is ColumnRole.CATEGORICAL:
        return _index_by_raw_values(
            column=column,
            ids=ids,
            series=series,
            keying=Keying.CATEGORICAL,
        )
    if role is ColumnRole.NUMERIC:
        numeric = _finite_numeric(series, column=column)
        n_unique = int(np.unique(numeric).size)
        if n_unique < RAW_UNIQUE_LIMIT:
            return _index_by_raw_values(
                column=column,
                ids=ids,
                series=series,
                keying=Keying.NUMERIC_RAW,
            )
        return _index_by_percentile(
            column=column,
            ids=ids,
            values=numeric,
        )
    raise ValueError(f"unsupported column role: {role!r}")


def frame_index_from_frame(
    frame: pd.DataFrame,
    *,
    id_column: str,
    categorical_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
) -> FrameIndex:
    """Invert every named column of ``frame``."""
    if id_column not in frame.columns:
        raise ValueError(f"frame is missing id column: {id_column!r}")
    n_categorical = len(categorical_columns)
    n_numeric = len(numeric_columns)
    n_columns = n_categorical + n_numeric
    if n_columns == 0:
        raise ValueError("categorical_columns and numeric_columns are both empty")
    seen: set[str] = set()
    roles: list[tuple[str, ColumnRole] | None] = [None] * n_columns
    for i in range(n_categorical):
        name = categorical_columns[i]
        _require_feature_column(
            frame,
            column=name,
            id_column=id_column,
            seen=seen,
        )
        roles[i] = (name, ColumnRole.CATEGORICAL)
    for j in range(n_numeric):
        name = numeric_columns[j]
        _require_feature_column(
            frame,
            column=name,
            id_column=id_column,
            seen=seen,
        )
        roles[n_categorical + j] = (name, ColumnRole.NUMERIC)
    filled_roles = [item for item in roles if item is not None]
    if len(filled_roles) != n_columns:
        raise ValueError("failed to read every column role")
    items: list[ColumnIndex | None] = [None] * n_columns
    for i in tqdm(range(n_columns), desc="column_index"):
        name, role = filled_roles[i]
        items[i] = column_index_from_frame(
            frame,
            column=name,
            id_column=id_column,
            role=role,
        )
    filled = [item for item in items if item is not None]
    if len(filled) != n_columns:
        raise ValueError("failed to index every column")
    ids = as_record_ids(frame[id_column], column=id_column)
    return FrameIndex(ids=_id_set(ids), columns=tuple(filled))


def _require_feature_column(
    frame: pd.DataFrame,
    *,
    column: str,
    id_column: str,
    seen: set[str],
) -> None:
    if column == id_column:
        raise ValueError(f"id column {id_column!r} cannot be indexed as a feature")
    if column in seen:
        raise ValueError(f"column {column!r} is listed more than once")
    if column not in frame.columns:
        raise ValueError(f"frame is missing column: {column!r}")
    seen.add(column)


def _index_by_raw_values(
    *,
    column: str,
    ids: np.ndarray,
    series: pd.Series,
    keying: Keying,
) -> ColumnIndex:
    codes, uniques = pd.factorize(series, sort=True)
    if uniques.size == 0:
        raise ValueError(f"column {column!r} has no keys")
    n_keys = int(uniques.size)
    keys: list[object | None] = [None] * n_keys
    for i in range(n_keys):
        keys[i] = _python_key(uniques[i])
    filled_keys = [key for key in keys if key is not None]
    if len(filled_keys) != n_keys:
        raise ValueError(f"column {column!r} failed to convert every key")
    postings = _postings_for_codes(
        ids,
        codes=codes.astype(np.int64),
        keys=tuple(filled_keys),
    )
    return ColumnIndex(
        column=column,
        keying=keying,
        postings=postings,
        percentile_edges=None,
    )


def _index_by_percentile(
    *,
    column: str,
    ids: np.ndarray,
    values: np.ndarray,
) -> ColumnIndex:
    edges = np.percentile(values, np.arange(PERCENTILE_EDGES)).astype(np.float64)
    _require_percentile_edges(edges)
    n_bins = PERCENTILE_RANK_MAX
    keys: list[object | None] = [None] * n_bins
    for i in range(n_bins):
        keys[i] = PercentileRange(start=i, stop=i + 1)
    filled_keys = [key for key in keys if key is not None]
    if len(filled_keys) != n_bins:
        raise ValueError(f"column {column!r} failed to build percentile keys")
    if not np.any(np.diff(edges) > 0):
        raise ValueError(f"column {column!r} percentile edges collapsed")
    ranks = np.searchsorted(edges, values, side="right") - 1
    ranks = np.clip(ranks, 0, n_bins - 1).astype(np.int64)
    postings = _postings_for_codes(ids, codes=ranks, keys=tuple(filled_keys))
    return ColumnIndex(
        column=column,
        keying=Keying.NUMERIC_PERCENTILE,
        postings=postings,
        percentile_edges=edges,
    )


def _postings_for_codes(
    ids: np.ndarray,
    *,
    codes: np.ndarray,
    keys: tuple[object, ...],
) -> dict[object, frozenset[int]]:
    n_keys = len(keys)
    if n_keys == 0:
        raise ValueError("keys is empty")
    if codes.shape != ids.shape:
        raise ValueError(
            f"codes and ids must have the same shape (got {codes.shape} and {ids.shape})"
        )
    n_rows = int(ids.size)
    buckets: list[set[int] | None] = [None] * n_keys
    for i in range(n_keys):
        buckets[i] = set()
    filled_buckets = [item for item in buckets if item is not None]
    if len(filled_buckets) != n_keys:
        raise ValueError("failed to allocate buckets")
    for i in range(n_rows):
        code = int(codes[i])
        if code < 0 or code >= n_keys:
            raise ValueError("codes are outside [0, n_keys)")
        filled_buckets[code].add(int(ids[i]))
    postings: dict[object, frozenset[int]] = {}
    for i in range(n_keys):
        bucket = filled_buckets[i]
        if len(bucket) == 0:
            continue
        postings[keys[i]] = frozenset(bucket)
    return postings


def _id_set(ids: np.ndarray) -> frozenset[int]:
    n_ids = int(ids.size)
    items: list[int | None] = [None] * n_ids
    for i in range(n_ids):
        items[i] = int(ids[i])
    filled = [item for item in items if item is not None]
    if len(filled) != n_ids:
        raise ValueError("failed to convert every id")
    return frozenset(filled)


def _finite_numeric(series: pd.Series, *, column: str) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(
            f"numeric column {column!r} must be numeric, got {series.dtype}"
        )
    values = series.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"column {column!r} contains non-finite values")
    return values


def _require_percentile_edges(edges: np.ndarray) -> None:
    if edges.ndim != 1:
        raise ValueError(f"percentile edges must be 1-d, got ndim={edges.ndim}")
    if edges.size != PERCENTILE_EDGES:
        raise ValueError(
            f"percentile edges must have length {PERCENTILE_EDGES}, got {edges.size}"
        )
    if not np.isfinite(edges).all():
        raise ValueError("percentile edges contain non-finite values")
    if np.any(np.diff(edges) < 0):
        raise ValueError("percentile edges are not non-decreasing")


def _resolve_key(key: object, *, keying: Keying) -> object:
    if keying is Keying.NUMERIC_PERCENTILE:
        if not isinstance(key, PercentileRange):
            raise TypeError(
                f"percentile index keys are PercentileRange, got {type(key)!r}"
            )
        parsed = parse_percentile_range(start=key.start, stop=key.stop)
        if parsed.stop - parsed.start != 1:
            raise ValueError(
                "percentile index keys have width 1 "
                f"(got {parsed})"
            )
        return parsed
    return _python_key(key)


def _python_key(value: object) -> object:
    if isinstance(value, PercentileRange):
        return parse_percentile_range(start=value.start, stop=value.stop)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, np.floating)):
        return _as_finite_float(value)
    if isinstance(value, (str, np.str_)):
        return str(value)
    raise TypeError(f"unsupported index key type: {type(value)!r}")


def _as_finite_float(value: object) -> float:
    number = float(np.asarray(value))
    if not np.isfinite(number):
        raise ValueError("value is not finite")
    return number


def _as_rank(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be int, got {type(value)!r}")
    if isinstance(value, (int, np.integer)):
        rank = int(value)
        if rank < 0:
            raise ValueError(f"{name} must be >= 0, got {rank}")
        return rank
    raise TypeError(f"{name} must be int, got {type(value)!r}")
