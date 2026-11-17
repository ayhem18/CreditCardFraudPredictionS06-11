"""Inverted index of a labeled table, one postings list per column."""

from ccf_ml.index.column_index import (
    PERCENTILE_RANK_MAX,
    RAW_UNIQUE_LIMIT,
    ColumnIndex,
    ColumnRole,
    FrameIndex,
    Keying,
    PercentileRange,
    column_index_from_frame,
    frame_index_from_frame,
    parse_percentile_range,
    percentile_range_of,
)

__all__ = (
    "PERCENTILE_RANK_MAX",
    "RAW_UNIQUE_LIMIT",
    "ColumnIndex",
    "ColumnRole",
    "FrameIndex",
    "Keying",
    "PercentileRange",
    "column_index_from_frame",
    "frame_index_from_frame",
    "parse_percentile_range",
    "percentile_range_of",
)
