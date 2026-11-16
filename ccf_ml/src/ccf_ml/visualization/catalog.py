"""Which table a plot expects.

A row is a (dimension, scales) pair plus the table's semantics. The notebook
(or later an analysis module) builds that table. The plot checks it and
draws. This catalog does not dispatch.
"""

from __future__ import annotations

from typing import NamedTuple

from ccf_ml.visualization.scale import VariableScale

CAT = VariableScale.CATEGORICAL
ORD = VariableScale.ORDINAL
NUM = VariableScale.NUMERIC


class PlotKind(NamedTuple):
    """One EDA question, the table it needs, and the plot that draws it."""

    n_variables: int
    scales: tuple[VariableScale, ...]
    question: str
    table: str
    plot: str
    implemented: bool


CATALOG: tuple[PlotKind, ...] = (
    PlotKind(
        n_variables=1,
        scales=(CAT,),
        question="P(X)",
        table="one column of category masses, summing to 1",
        plot="visualization.univariate.categorical.plot_categorical_mass",
        implemented=False,
    ),
    PlotKind(
        n_variables=1,
        scales=(NUM,),
        question="density / ECDF of X",
        table="a numeric series, or bins with density",
        plot="visualization.univariate.numeric.plot_numeric_distribution",
        implemented=False,
    ),
    PlotKind(
        n_variables=1,
        scales=(ORD,),
        question="P(X) with the given order on the axis",
        table="category masses with a declared order",
        plot="visualization.univariate.ordinal.plot_ordinal_mass",
        implemented=False,
    ),
    PlotKind(
        n_variables=2,
        scales=(CAT, CAT),
        question="P(feature | group)",
        table="row-stochastic matrix: rows=group, columns=feature, cells in [0, 1]; one panel per row",
        plot="visualization.bivariate.categorical_categorical.plot_conditional_share",
        implemented=True,
    ),
    PlotKind(
        n_variables=2,
        scales=(CAT, CAT),
        question="n(event, row, col) and P(row, col | event)",
        table=(
            "two aligned matrices: non-negative counts; shares in [0, 1] "
            "summing to 1"
        ),
        plot=(
            "visualization.bivariate.categorical_categorical"
            ".plot_count_and_share_heatmap"
        ),
        implemented=True,
    ),
    PlotKind(
        n_variables=2,
        scales=(CAT, CAT),
        question="P(event | row, col)",
        table="aligned matrix of rates in [0, 1]; optional aligned n",
        plot=(
            "visualization.bivariate.categorical_categorical"
            ".plot_rate_heatmap"
        ),
        implemented=True,
    ),
    PlotKind(
        n_variables=2,
        scales=(CAT, CAT),
        question="P(condition = positive | feature)",
        table="index=feature, columns n, n_positive, rate, lift; rate in [0, 1]",
        plot="visualization.bivariate.condition_categorical.plot_binary_rate",
        implemented=True,
    ),
    PlotKind(
        n_variables=2,
        scales=(CAT, NUM),
        question="P(x <= t | category)",
        table="long ECDF: columns x, cdf, group; cdf in [0, 1], last cdf per group is 1",
        plot="visualization.bivariate.categorical_numeric.plot_conditional_cdf",
        implemented=True,
    ),
    PlotKind(
        n_variables=2,
        scales=(NUM, NUM),
        question="joint of two numeric variables",
        table="two aligned numeric series",
        plot="visualization.bivariate.numeric_numeric.plot_numeric_numeric",
        implemented=False,
    ),
)


def implemented_plots() -> tuple[PlotKind, ...]:
    """Return catalog rows whose plot function exists."""
    return tuple(kind for kind in CATALOG if kind.implemented)
