"""Tick and legend text for category levels."""

from __future__ import annotations

from collections.abc import Mapping


def tick_label(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def legend_label(
    value: object,
    *,
    given: str,
    given_labels: Mapping[object, str] | None,
) -> str:
    if given_labels is None:
        return f"{given}={tick_label(value)}"
    if value in given_labels:
        return given_labels[value]
    if isinstance(value, float) and value.is_integer() and int(value) in given_labels:
        return given_labels[int(value)]
    return f"{given}={tick_label(value)}"
