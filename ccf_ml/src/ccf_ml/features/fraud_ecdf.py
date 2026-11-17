"""Fraud-conditional ECDF of a numeric column.

``F(t) = P(x <= t | is_fraud = 1)``. Estimated from the fraud rows of a
labeled sample: the count of those rows with ``x <= t``, divided by the
number of fraud rows.

Apply is a monotone map of ``x`` onto ``[0, 1]``. Estimate on a fold's
train ids; apply to that fold's val ids (and to test with the same curve).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from ccf_ml.cross_validation.alarm import as_binary_label
from ccf_ml.cross_validation.constants import TARGET_COLUMN


class FraudEcdf(NamedTuple):
    """Sorted fraud values of one numeric column.

    ``evaluate(t)`` is ``(# values <= t) / n``.
    """

    column: str
    values: np.ndarray

    @property
    def n(self) -> int:
        """Number of fraud rows used to estimate the curve."""
        return int(self.values.size)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Return ``F(t)`` for each entry of ``x``."""
        _require_finite_1d(x, name="x")
        n = self.n
        ranks = np.searchsorted(self.values, x, side="right")
        return ranks.astype(np.float64) / float(n)

    def curve_table(self) -> pd.DataFrame:
        """Unique ``x`` and the ECDF at those points. Last ``cdf`` is 1."""
        unique, counts = np.unique(self.values, return_counts=True)
        cdf = np.cumsum(counts, dtype=np.float64) / float(self.n)
        return pd.DataFrame({self.column: unique, "cdf": cdf})


def fraud_ecdf_from_arrays(
    *,
    column: str,
    x: np.ndarray,
    y: np.ndarray,
) -> FraudEcdf:
    """Estimate ``F`` from aligned ``x`` and binary ``y``."""
    _require_finite_1d(x, name="x")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-d, got ndim={y.ndim}")
    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape (got {x.shape} and {y.shape})"
        )
    y_int = y.astype(np.int64)
    if not np.all(y == y_int):
        raise ValueError("y is not integer-valued")
    if not np.all((y_int == 0) | (y_int == 1)):
        raise ValueError("y is not binary")
    fraud_x = x[y_int == 1]
    if fraud_x.size == 0:
        raise ValueError(f"no fraud rows to estimate {column!r}")
    values = np.sort(fraud_x)
    return FraudEcdf(column=column, values=values)


def fraud_ecdf_from_frame(
    frame: pd.DataFrame,
    *,
    column: str,
    target_column: str = TARGET_COLUMN,
) -> FraudEcdf:
    """Estimate ``F`` from a labeled table."""
    if column not in frame.columns:
        raise ValueError(f"frame is missing column: {column!r}")
    if target_column not in frame.columns:
        raise ValueError(f"frame is missing target column: {target_column!r}")
    x = frame[column].to_numpy(dtype=np.float64)
    y = as_binary_label(frame[target_column], column=target_column)
    return fraud_ecdf_from_arrays(column=column, x=x, y=y)


def common_grid(curves: tuple[FraudEcdf, ...]) -> np.ndarray:
    """Sorted unique ``x`` values across curves. Enough to compare step ECDFs."""
    n_curves = len(curves)
    if n_curves == 0:
        raise ValueError("curves is empty")
    parts: list[np.ndarray | None] = [None] * n_curves
    for i in range(n_curves):
        parts[i] = np.unique(curves[i].values)
    filled = [part for part in parts if part is not None]
    if len(filled) != n_curves:
        raise ValueError("failed to read every curve")
    return np.unique(np.concatenate(filled))


def max_abs_difference(
    left: FraudEcdf,
    right: FraudEcdf,
    *,
    grid: np.ndarray,
) -> float:
    """``max_t |F_left(t) - F_right(t)|`` on ``grid``."""
    _require_finite_1d(grid, name="grid")
    if grid.size == 0:
        raise ValueError("grid is empty")
    delta = np.abs(left.evaluate(grid) - right.evaluate(grid))
    return float(np.max(delta))


def transform_frame(
    frame: pd.DataFrame,
    *,
    curves: tuple[FraudEcdf, ...],
) -> pd.DataFrame:
    """Replace each curve's column with ``F(x)``. Other columns are copied."""
    n_curves = len(curves)
    if n_curves == 0:
        raise ValueError("curves is empty")
    missing = [curve.column for curve in curves if curve.column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing columns: {missing}")
    out = frame.copy()
    for i in range(n_curves):
        curve = curves[i]
        out[curve.column] = curve.evaluate(
            out[curve.column].to_numpy(dtype=np.float64)
        )
    return out


def _require_finite_1d(values: np.ndarray, *, name: str) -> None:
    if values.ndim != 1:
        raise ValueError(f"{name} must be 1-d, got ndim={values.ndim}")
    if values.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
