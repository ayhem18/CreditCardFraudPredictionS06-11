"""Condition vs one categorical: ``P(condition = positive | feature)``.

The condition is a binary label (fraud / not fraud). The other variable
is a categorical feature.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ccf_ml.visualization.labels import tick_label

_SHARE_ATOL = 1e-6
_BINARY_RATE_COLUMNS = ("n", "n_positive", "rate", "lift")


def require_binary_rate(rates: pd.DataFrame, *, base_rate: float) -> None:
    """Raise unless ``rates`` is ``P(target = positive | feature)``.

    Required columns: ``n``, ``n_positive``, ``rate``, ``lift``. ``rate``
    is in ``[0, 1]``, ``n_positive <= n``, and ``lift`` equals
    ``rate / base_rate``.
    """
    if not isinstance(rates, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(rates)!r}")
    missing = [c for c in _BINARY_RATE_COLUMNS if c not in rates.columns]
    if missing:
        raise ValueError(f"binary rate table missing columns: {missing}")
    if rates.empty:
        raise ValueError("binary rate table is empty")
    if rates[list(_BINARY_RATE_COLUMNS)].isna().to_numpy().any():
        raise ValueError("binary rate table contains NA")
    rate = rates["rate"].to_numpy(dtype="float64")
    if np.any(rate < -_SHARE_ATOL) or np.any(rate > 1.0 + _SHARE_ATOL):
        raise ValueError("rate must be in [0, 1]")
    if (rates["n_positive"] > rates["n"]).any():
        raise ValueError("n_positive cannot exceed n")
    if base_rate <= 0.0:
        raise ValueError("base_rate must be positive")
    if not np.allclose(
        rates["lift"].to_numpy(dtype="float64"),
        rate / base_rate,
        atol=_SHARE_ATOL,
        equal_nan=False,
    ):
        raise ValueError("lift must equal rate / base_rate")


def plot_binary_rate(
    rates: pd.DataFrame,
    *,
    feature: str,
    target: str,
    positive: object,
    base_rate: float,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Horizontal bars of a ``P(target = positive | feature)`` table.

    A vertical line marks ``base_rate``. Rows are drawn in table order
    (first row at the bottom).
    """
    require_binary_rate(rates, base_rate=base_rate)
    n_feature = len(rates)
    if ax is None:
        height = max(3.0, 0.38 * n_feature + 1.4)
        fig, ax = plt.subplots(figsize=(10.0, height))
    else:
        fig = ax.figure
    y = range(n_feature)
    ax.barh(list(y), rates["rate"].to_numpy())
    ax.set_yticks(list(y))
    ax.set_yticklabels([tick_label(v) for v in rates.index])
    ax.axvline(base_rate, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"P({target}={tick_label(positive)} | {feature})")
    ax.set_xlim(0, max(base_rate, float(rates["rate"].max())) * 1.12)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax
