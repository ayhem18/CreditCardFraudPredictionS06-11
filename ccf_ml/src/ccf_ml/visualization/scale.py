"""Scale of a plotted variable.

The plot is chosen from (how many variables, the scale of each). Binary is
categorical with two levels, not its own scale.
"""

from __future__ import annotations

from enum import StrEnum


class VariableScale(StrEnum):
    """How a variable is used on an axis, not its pandas dtype."""

    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    NUMERIC = "numeric"
