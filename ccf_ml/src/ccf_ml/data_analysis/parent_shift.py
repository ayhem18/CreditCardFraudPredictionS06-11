"""Label-blind numeric shift of a cell against its parent."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ccf_ml.data_analysis.constants import (
    SHIFT_KS_FLOOR,
    SHIFT_MARGIN,
    SHIFT_MIN_SIDE,
)
from ccf_ml.data_analysis.windows import Window, age_membership, hour_membership


class ShiftRow(NamedTuple):
    """One (cell, numeric, parent axis) KS comparison."""

    profile_id: str
    spec_name: str
    cell_key: str
    parent_key: str
    parent_axis: str
    numeric: str
    n_cell: int
    n_parent: int
    ks_d: float
    delta_f_p50: float
    delta_f_p90: float
    mean_parent_rank: float
    scramble_d: float
    verdict: str
    distilled: bool
    is_window: bool


def occupancy_ok(*, n_cell: int, n_parent: int, min_side: int = SHIFT_MIN_SIDE) -> bool:
    """Both the cell and its complement inside the parent meet ``min_side``."""
    n_complement = n_parent - n_cell
    return n_cell >= min_side and n_complement >= min_side


def parent_cdf(sorted_parent: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``F_parent(t) = (# parent rows with x <= t) / n_parent``."""
    n_parent = int(sorted_parent.size)
    if n_parent == 0:
        raise ValueError("parent is empty")
    ranks = np.searchsorted(sorted_parent, values, side="right")
    return ranks.astype(np.float64) / float(n_parent)


def ks_d_cell_vs_parent(cell_x: np.ndarray, parent_x: np.ndarray) -> float:
    """``sup |F_cell - F_parent|`` on the usual (right-continuous) ECDFs."""
    if cell_x.size == 0 or parent_x.size == 0:
        raise ValueError("cell_x and parent_x must be non-empty")
    cell_sorted = np.sort(cell_x.astype(np.float64))
    parent_sorted = np.sort(parent_x.astype(np.float64))
    grid = np.unique(np.concatenate([cell_sorted, parent_sorted]))
    f_cell_right = np.searchsorted(cell_sorted, grid, side="right") / float(
        cell_sorted.size
    )
    f_parent_right = np.searchsorted(parent_sorted, grid, side="right") / float(
        parent_sorted.size
    )
    f_cell_left = np.searchsorted(cell_sorted, grid, side="left") / float(
        cell_sorted.size
    )
    f_parent_left = np.searchsorted(parent_sorted, grid, side="left") / float(
        parent_sorted.size
    )
    d_right = float(np.max(np.abs(f_cell_right - f_parent_right)))
    d_left = float(np.max(np.abs(f_cell_left - f_parent_left)))
    return max(d_right, d_left)


def shift_companions(
    cell_x: np.ndarray,
    parent_x: np.ndarray,
) -> tuple[float, float, float]:
    """``ΔF`` at the cell/parent p50 and p90, plus mean parent-rank of the cell."""
    parent_sorted = np.sort(parent_x.astype(np.float64))
    cell_vals = cell_x.astype(np.float64)
    p50_cell = float(np.quantile(cell_vals, 0.5))
    p50_parent = float(np.quantile(parent_x.astype(np.float64), 0.5))
    p90_cell = float(np.quantile(cell_vals, 0.9))
    p90_parent = float(np.quantile(parent_x.astype(np.float64), 0.9))
    f_p50_cell = float(parent_cdf(parent_sorted, np.array([p50_cell]))[0])
    f_p50_parent = float(parent_cdf(parent_sorted, np.array([p50_parent]))[0])
    f_p90_cell = float(parent_cdf(parent_sorted, np.array([p90_cell]))[0])
    f_p90_parent = float(parent_cdf(parent_sorted, np.array([p90_parent]))[0])
    mean_rank = float(parent_cdf(parent_sorted, cell_vals).mean())
    return f_p50_cell - f_p50_parent, f_p90_cell - f_p90_parent, mean_rank


def scramble_window_d(
    *,
    parent_numeric: np.ndarray,
    parent_axis_values: np.ndarray,
    window: Window,
    rng: np.random.Generator,
) -> float | None:
    """KS D of ``X`` after shuffling the window axis inside the parent."""
    shuffled = parent_axis_values.astype(np.float64, copy=True)
    rng.shuffle(shuffled)
    if window.kind == "hour":
        mask = hour_membership(shuffled, window=window)
    elif window.kind == "age":
        mask = age_membership(shuffled, window=window)
    else:
        raise ValueError(f"unknown window kind {window.kind!r}")
    n_cell = int(mask.sum())
    n_parent = int(parent_numeric.size)
    if not occupancy_ok(n_cell=n_cell, n_parent=n_parent):
        return None
    return ks_d_cell_vs_parent(parent_numeric[mask], parent_numeric)


def verdict_for(
    *,
    n_cell: int,
    n_parent: int,
    ks_d: float,
    scramble_d: float | None,
    require_scramble: bool,
    hit_verdict: str,
) -> str:
    """``too_thin`` / ``below_floor`` / ``hit_verdict`` using pre-registered floors."""
    if not occupancy_ok(n_cell=n_cell, n_parent=n_parent):
        return "too_thin"
    if ks_d < SHIFT_KS_FLOOR:
        return "below_floor"
    if require_scramble:
        if scramble_d is None:
            return "below_floor"
        if ks_d < SHIFT_MARGIN * scramble_d:
            return "below_floor"
    return hit_verdict


def measure_shift(
    *,
    cell_x: np.ndarray,
    parent_x: np.ndarray,
    profile_id: str,
    spec_name: str,
    cell_key: str,
    parent_key: str,
    parent_axis: str,
    numeric: str,
    require_scramble: bool,
    hit_verdict: str,
    window: Window | None,
    parent_axis_values: np.ndarray | None,
    rng: np.random.Generator | None,
) -> ShiftRow:
    """Occupancy, KS, companions, optional window scramble, and a verdict."""
    n_cell = int(cell_x.size)
    n_parent = int(parent_x.size)
    if not occupancy_ok(n_cell=n_cell, n_parent=n_parent):
        return ShiftRow(
            profile_id=profile_id,
            spec_name=spec_name,
            cell_key=cell_key,
            parent_key=parent_key,
            parent_axis=parent_axis,
            numeric=numeric,
            n_cell=n_cell,
            n_parent=n_parent,
            ks_d=float("nan"),
            delta_f_p50=float("nan"),
            delta_f_p90=float("nan"),
            mean_parent_rank=float("nan"),
            scramble_d=float("nan"),
            verdict="too_thin",
            distilled=False,
            is_window=window is not None,
        )
    ks_d = ks_d_cell_vs_parent(cell_x, parent_x)
    delta_p50, delta_p90, mean_rank = shift_companions(cell_x, parent_x)
    scramble_d: float | None = None
    if require_scramble:
        if window is None or parent_axis_values is None or rng is None:
            raise ValueError("scramble requires window, parent_axis_values, and rng")
        scramble_d = scramble_window_d(
            parent_numeric=parent_x,
            parent_axis_values=parent_axis_values,
            window=window,
            rng=rng,
        )
    verdict = verdict_for(
        n_cell=n_cell,
        n_parent=n_parent,
        ks_d=ks_d,
        scramble_d=scramble_d,
        require_scramble=require_scramble,
        hit_verdict=hit_verdict,
    )
    scramble_value = float("nan") if scramble_d is None else float(scramble_d)
    return ShiftRow(
        profile_id=profile_id,
        spec_name=spec_name,
        cell_key=cell_key,
        parent_key=parent_key,
        parent_axis=parent_axis,
        numeric=numeric,
        n_cell=n_cell,
        n_parent=n_parent,
        ks_d=ks_d,
        delta_f_p50=delta_p50,
        delta_f_p90=delta_p90,
        mean_parent_rank=mean_rank,
        scramble_d=scramble_value,
        verdict=verdict,
        distilled=False,
        is_window=window is not None,
    )


def mark_distilled(rows: list[ShiftRow]) -> list[ShiftRow]:
    """Keep at most one window per ``(spec_name, parent_key, numeric)``.

    Non-window rows stay ``distilled=False``. Among window rows with
    ``verdict`` equal to the spec's hit label, the largest ``ks_d`` wins.
    """
    n_rows = len(rows)
    distilled_flags = [False] * n_rows
    best_index: dict[tuple[str, str, str], int] = {}
    best_d: dict[tuple[str, str, str], float] = {}
    for i in range(n_rows):
        row = rows[i]
        if row.verdict not in {"fe_candidate", "descriptive_only"}:
            continue
        if not row.is_window:
            continue
        key = (row.spec_name, row.parent_key, row.numeric)
        current = best_d.get(key)
        if current is None or row.ks_d > current:
            best_d[key] = float(row.ks_d)
            best_index[key] = i
    for index in best_index.values():
        distilled_flags[index] = True
    marked = [None] * n_rows
    for i in range(n_rows):
        row = rows[i]
        marked[i] = row._replace(distilled=distilled_flags[i])
    return marked


def shift_as_dict(row: ShiftRow) -> dict[str, object]:
    """Flat record for a vs-parent table."""
    return row._asdict()
