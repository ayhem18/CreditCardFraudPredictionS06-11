"""Categorical × numeric: ``P(x <= t | group)``.

Overlaid ECDFs of a continuous variable, one curve per category.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ccf_ml.visualization.labels import legend_label

_SHARE_ATOL = 1e-6
_X_SCALES = ("linear", "log", "symlog")


def require_conditional_cdf(
    curves: pd.DataFrame,
    *,
    x: str,
    cdf: str,
    group: str,
) -> None:
    """Raise unless ``curves`` is ``P(x <= t | group)`` as an ECDF table.

    Required columns ``x``, ``cdf``, ``group``. Within each group, ``x``
    and ``cdf`` are nondecreasing, ``cdf`` is in ``[0, 1]``, and the last
    ``cdf`` is 1.
    """
    if not isinstance(curves, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(curves)!r}")
    missing = [c for c in (x, cdf, group) if c not in curves.columns]
    if missing:
        raise ValueError(f"conditional CDF table missing columns: {missing}")
    if curves.empty:
        raise ValueError("conditional CDF table is empty")
    if curves[[x, cdf]].isna().to_numpy().any():
        raise ValueError("conditional CDF table contains NA in x or cdf")
    cdf_values = curves[cdf].to_numpy(dtype="float64")
    if np.any(cdf_values < -_SHARE_ATOL) or np.any(cdf_values > 1.0 + _SHARE_ATOL):
        raise ValueError("cdf must be in [0, 1]")
    for group_value, part in curves.groupby(group, sort=False):
        xx = part[x].to_numpy(dtype="float64")
        ff = part[cdf].to_numpy(dtype="float64")
        if xx.size == 0:
            raise ValueError(f"group {group_value!r} has no rows")
        if np.any(np.diff(xx) < -_SHARE_ATOL):
            raise ValueError(
                f"x must be nondecreasing within group {group_value!r}"
            )
        if np.any(np.diff(ff) < -_SHARE_ATOL):
            raise ValueError(
                f"cdf must be nondecreasing within group {group_value!r}"
            )
        if abs(float(ff[-1]) - 1.0) > 1e-3:
            raise ValueError(
                f"last cdf in group {group_value!r} must be 1, got {ff[-1]}"
            )


def plot_conditional_cdf(
    curves: pd.DataFrame,
    *,
    x: str,
    cdf: str,
    group: str,
    x_scale: str = "linear",
    group_labels: Mapping[object, str] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Overlaid ECDFs of a numeric variable given a categorical group.

    ``x_scale`` is a matplotlib axis scale (``linear``, ``log``,
    ``symlog``). Log scale requires ``x > 0``.
    """
    require_conditional_cdf(curves, x=x, cdf=cdf, group=group)
    if x_scale not in _X_SCALES:
        raise ValueError(f"x_scale must be one of {_X_SCALES}, got {x_scale!r}")
    if x_scale == "log" and (curves[x].to_numpy(dtype="float64") <= 0).any():
        raise ValueError("log scale requires x > 0")
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.0, 5.0))
    else:
        fig = ax.figure
    for group_value, part in curves.groupby(group, sort=True):
        label = legend_label(
            group_value, given=group, given_labels=group_labels
        )
        ax.plot(
            part[x].to_numpy(),
            part[cdf].to_numpy(),
            label=label,
            drawstyle="steps-post",
        )
    ax.set_xscale(x_scale)
    ax.set_ylim(0, 1)
    ax.set_xlabel(x)
    ax.set_ylabel(f"P({x} <= t | {group})")
    ax.legend(title=group)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, ax
