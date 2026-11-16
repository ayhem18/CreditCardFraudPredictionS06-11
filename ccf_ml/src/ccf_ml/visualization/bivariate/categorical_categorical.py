"""Two categorical variables.

``P(feature | group)`` is one panel per group. Heatmaps are the joint
of an event over two categoricals, or ``P(event | row, col)``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from ccf_ml.visualization.labels import legend_label, tick_label

_SHARE_ATOL = 1e-6


def require_conditional_share(share: pd.DataFrame) -> None:
    """Raise unless ``share`` is ``P(feature | given)``.

    Rows are group levels, columns are feature levels, cells are in
    ``[0, 1]``, and each row sums to 1.
    """
    if not isinstance(share, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(share)!r}")
    if share.empty:
        raise ValueError("conditional share table is empty")
    if share.isna().to_numpy().any():
        raise ValueError("conditional share table contains NA")
    try:
        values = share.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("conditional share table must be numeric") from exc
    if values.ndim != 2:
        raise ValueError("conditional share table must be 2-dimensional")
    if np.any(values < -_SHARE_ATOL) or np.any(values > 1.0 + _SHARE_ATOL):
        raise ValueError("shares must be in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=_SHARE_ATOL):
        raise ValueError("each given-level row must sum to 1")


def plot_conditional_share(
    share: pd.DataFrame,
    *,
    feature: str,
    given: str,
    given_labels: Mapping[object, str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """One bar chart per group of a ``P(feature | given)`` table.

    Each panel is one row of the table, so the bars in a panel sum to 1.
    Adjacent panels are two mixes, not a paired fraud / not-fraud reading
    of the same category. Shared y-order and x-limit ``[0, 1]``. Column
    order on the table is the drawing order (first column at the bottom).
    """
    require_conditional_share(share)
    n_feature = len(share.columns)
    n_given = len(share.index)
    height = max(3.0, 0.38 * n_feature + 1.6)
    width = max(4.5, 4.8 * n_given)
    fig, axes = plt.subplots(
        1,
        n_given,
        figsize=(width, height),
        sharey=True,
        squeeze=False,
    )
    y = np.arange(n_feature, dtype="float64")
    y_labels = [tick_label(v) for v in share.columns]
    for ax, given_value in zip(axes[0], share.index):
        ax.barh(y, share.loc[given_value].to_numpy())
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.set_xlim(0, 1)
        ax.set_title(
            legend_label(given_value, given=given, given_labels=given_labels)
        )
        ax.grid(True, axis="x", alpha=0.3)
    axes[0][0].set_ylabel(feature)
    fig.supxlabel(f"P({feature} | {given})")
    fig.tight_layout()
    return fig, axes


def require_count_and_share(counts: pd.DataFrame, share: pd.DataFrame) -> None:
    """Raise unless ``counts`` and ``share`` are aligned matrices.

    Counts are non-negative. Share is in ``[0, 1]`` and sums to 1
    (``P(row, col | event)``).
    """
    if not isinstance(counts, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(counts)!r}")
    if not isinstance(share, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(share)!r}")
    if counts.empty or share.empty:
        raise ValueError("count or share table is empty")
    if not counts.index.equals(share.index):
        raise ValueError("count and share tables must share the same index")
    if not counts.columns.equals(share.columns):
        raise ValueError("count and share tables must share the same columns")
    if counts.isna().to_numpy().any() or share.isna().to_numpy().any():
        raise ValueError("count or share table contains NA")
    try:
        count_values = counts.to_numpy(dtype="float64")
        share_values = share.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("count and share tables must be numeric") from exc
    if np.any(count_values < -_SHARE_ATOL):
        raise ValueError("counts must be non-negative")
    if np.any(share_values < -_SHARE_ATOL) or np.any(
        share_values > 1.0 + _SHARE_ATOL
    ):
        raise ValueError("shares must be in [0, 1]")
    if not np.isclose(share_values.sum(), 1.0, atol=_SHARE_ATOL):
        raise ValueError("shares must sum to 1")


def _plot_annotated_heatmap(
    values: np.ndarray,
    cell_text: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    row: str,
    col: str,
    colorbar_label: str,
    ax: Axes | None = None,
    vmin: float = 0.0,
    vmax: float | None = None,
    colorbar_percent: bool = False,
) -> tuple[Figure, Axes]:
    n_row, n_col = values.shape
    created_fig = ax is None
    if ax is None:
        width = max(8.0, 1.15 * n_col + 2.4)
        height = max(3.6, 0.72 * n_row + 1.8)
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure
    image = ax.imshow(
        values,
        cmap="YlOrRd",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(n_col, dtype="float64"))
    ax.set_yticks(np.arange(n_row, dtype="float64"))
    ax.set_xticklabels(col_labels, rotation=40, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    cmap = image.cmap
    norm = image.norm
    for i in range(n_row):
        for j in range(n_col):
            rgba = cmap(norm(values[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = "black" if luminance > 0.55 else "white"
            ax.text(
                j,
                i,
                cell_text[i, j],
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.5,
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    if colorbar_percent:
        colorbar.ax.yaxis.set_major_formatter(
            PercentFormatter(xmax=1.0, decimals=0)
        )
    if created_fig:
        fig.tight_layout()
    return fig, ax


def plot_count_and_share_heatmap(
    counts: pd.DataFrame,
    share: pd.DataFrame,
    *,
    row: str,
    col: str,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Heatmap of event counts, annotated with count and share of the event.

    Color is the absolute count. Each cell is labeled with that count and
    with the cell's share of all events (``P(row, col | event)``).
    """
    require_count_and_share(counts, share)
    count_values = counts.to_numpy(dtype="float64")
    share_values = share.to_numpy(dtype="float64")
    cell_text = np.empty(count_values.shape, dtype=object)
    for i in range(count_values.shape[0]):
        for j in range(count_values.shape[1]):
            cell_text[i, j] = (
                f"{int(round(count_values[i, j]))}\n"
                f"{100.0 * share_values[i, j]:.1f}%"
            )
    return _plot_annotated_heatmap(
        count_values,
        cell_text,
        row_labels=[tick_label(v) for v in counts.index],
        col_labels=[tick_label(v) for v in counts.columns],
        row=row,
        col=col,
        colorbar_label="n fraud",
        ax=ax,
    )


def require_rate(rate: pd.DataFrame) -> None:
    """Raise unless ``rate`` is a ``P(event | row, col)`` matrix in ``[0, 1]``."""
    if not isinstance(rate, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(rate)!r}")
    if rate.empty:
        raise ValueError("rate table is empty")
    if rate.isna().to_numpy().any():
        raise ValueError("rate table contains NA")
    try:
        values = rate.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise ValueError("rate table must be numeric") from exc
    if values.ndim != 2:
        raise ValueError("rate table must be 2-dimensional")
    if np.any(values < -_SHARE_ATOL) or np.any(values > 1.0 + _SHARE_ATOL):
        raise ValueError("rates must be in [0, 1]")


def plot_rate_heatmap(
    rate: pd.DataFrame,
    *,
    row: str,
    col: str,
    n: pd.DataFrame | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Heatmap of ``P(event | row, col)``.

    Color is the cell rate. Each cell is labeled with that rate. If ``n``
    is given, the second line is the cell size.
    """
    require_rate(rate)
    if n is not None:
        if not isinstance(n, pd.DataFrame):
            raise TypeError(f"expected a DataFrame, got {type(n)!r}")
        if not n.index.equals(rate.index) or not n.columns.equals(rate.columns):
            raise ValueError("n must be aligned with rate")
        if n.isna().to_numpy().any():
            raise ValueError("n table contains NA")
        try:
            n_values = n.to_numpy(dtype="float64")
        except (TypeError, ValueError) as exc:
            raise ValueError("n table must be numeric") from exc
        if np.any(n_values < -_SHARE_ATOL):
            raise ValueError("n must be non-negative")
    else:
        n_values = None
    rate_values = rate.to_numpy(dtype="float64")
    cell_text = np.empty(rate_values.shape, dtype=object)
    for i in range(rate_values.shape[0]):
        for j in range(rate_values.shape[1]):
            rate_label = f"{100.0 * rate_values[i, j]:.1f}%"
            if n_values is None:
                cell_text[i, j] = rate_label
            else:
                cell_text[i, j] = (
                    f"{rate_label}\nn={int(round(n_values[i, j]))}"
                )
    return _plot_annotated_heatmap(
        rate_values,
        cell_text,
        row_labels=[tick_label(v) for v in rate.index],
        col_labels=[tick_label(v) for v in rate.columns],
        row=row,
        col=col,
        colorbar_label="P(fraud | cell)",
        ax=ax,
        colorbar_percent=True,
    )
