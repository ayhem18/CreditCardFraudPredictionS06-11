"""Measure ``int_slots`` and ``decimal_slots`` for each continuous column.

A continuous value is tokenized as a fixed-width digit string. The widths
must be read from the CSV *text* (train + test, label-free). Parsing as
float and converting back invents extra decimals and is not the generator's
handwriting.

For each column:

- ``int_slots``: max number of digits in the integral part
- ``decimal_slots``: max number of digits after the decimal point
- ``digit_slots``: ``int_slots + decimal_slots``

Padded form of a value is then ``int.zfill(int_slots) + '.' + dec.ljust(decimal_slots, '0')``.

Run (from ``ccf_tabformer``, project venv activated)::

    python experiments/data/digit_slots.py
    python experiments/data/digit_slots.py --source corrected

``raw`` (default) reads ``storage/raw_data/``. ``corrected`` reads the working
copy at ``storage/data/corrected/``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from ccf_tabformer_dl.file_utils import (
    get_raw_test_csv_path,
    get_raw_train_csv_path,
    get_script_table_dir,
    get_test_csv_path,
    get_train_csv_path,
)

CONTINUOUS_COLUMNS = (
    "amount_usd",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
)

SCRIPT_STEM = "digit_slots"
EXPERIMENT_GROUP = "data"
NUMBER_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")


def split_written_parts(*, texts: pd.Series, column: str) -> tuple[pd.Series, pd.Series]:
    """Split written numbers into unsigned integer digits and decimal digits."""
    stripped = texts.astype(str).str.strip()
    if stripped.eq("").any() or stripped.isna().any():
        raise ValueError(f"{column}: empty or missing written values")
    valid = stripped.str.fullmatch(NUMBER_PATTERN.pattern)
    if not bool(valid.all()):
        bad_index = int((~valid).to_numpy().nonzero()[0][0])
        raise ValueError(
            f"{column}: not a plain decimal number at row {bad_index}: "
            f"{stripped.iloc[bad_index]!r}"
        )
    unsigned = stripped.str.replace(r"^-", "", n=1, regex=True)
    split = unsigned.str.split(".", n=1, expand=True)
    integer_parts = split[0]
    if 1 in split.columns:
        decimal_parts = split[1].fillna("")
    else:
        decimal_parts = pd.Series("", index=unsigned.index)
    if integer_parts.eq("").any():
        raise ValueError(f"{column}: integer part is empty")
    return integer_parts, decimal_parts


def measure_column(*, series: pd.Series, column: str) -> dict[str, object]:
    """Return slot widths and a few diagnostics for one continuous column."""
    texts = series.astype(str)
    n_rows = len(texts)
    integer_parts, decimal_parts = split_written_parts(texts=texts, column=column)
    int_lengths = integer_parts.str.len()
    dec_lengths = decimal_parts.str.len()
    int_slots = int(int_lengths.max())
    decimal_slots = int(dec_lengths.max())
    padded = (
        integer_parts.str.zfill(int_slots)
        + "."
        + decimal_parts.str.ljust(decimal_slots, "0")
    )
    numeric = texts.astype(float)
    padded_numeric = padded.astype(float)
    mismatch = (padded_numeric - numeric).abs() > 0.0
    if bool(mismatch.any()):
        bad_index = int(mismatch.to_numpy().nonzero()[0][0])
        raise ValueError(
            f"{column}: padding changed the value at row {bad_index}: "
            f"{texts.iloc[bad_index]!r} -> {padded.iloc[bad_index]!r}"
        )
    widest_int_index = int(int_lengths.idxmax())
    widest_dec_index = int(dec_lengths.idxmax())
    return {
        "column": column,
        "int_slots": int_slots,
        "decimal_slots": decimal_slots,
        "digit_slots": int_slots + decimal_slots,
        "n_rows": n_rows,
        "n_unique": int(texts.nunique()),
        "min_value": texts.iloc[int(numeric.idxmin())],
        "max_value": texts.iloc[int(numeric.idxmax())],
        "example_max_int": texts.iloc[widest_int_index],
        "example_max_int_padded": padded.iloc[widest_int_index],
        "example_max_decimal": texts.iloc[widest_dec_index],
        "example_max_decimal_padded": padded.iloc[widest_dec_index],
        "n_at_max_int_slots": int((int_lengths == int_slots).sum()),
        "n_at_max_decimal_slots": int((dec_lengths == decimal_slots).sum()),
        "has_sign": bool(texts.str.startswith("-").any()),
    }


def _source_paths(*, source: str) -> tuple[Path, Path, str]:
    if source == "raw":
        return get_raw_train_csv_path(), get_raw_test_csv_path(), "digit_slots"
    if source == "corrected":
        return get_train_csv_path(), get_test_csv_path(), "digit_slots_corrected"
    raise ValueError(f"unknown source: {source!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("raw", "corrected"),
        default="raw",
        help="CSV copy to measure. Default: raw (generator handwriting).",
    )
    args = parser.parse_args()
    train_path, test_path, stem = _source_paths(source=args.source)
    columns = list(CONTINUOUS_COLUMNS)
    train = pd.read_csv(train_path, dtype=str, usecols=columns)
    test = pd.read_csv(test_path, dtype=str, usecols=columns)
    pooled = pd.concat([train, test], ignore_index=True)
    n_columns = len(columns)
    rows: list[dict[str, object] | None] = [None] * n_columns
    for index in range(n_columns):
        column = columns[index]
        rows[index] = measure_column(series=pooled[column], column=column)
    measured = [row for row in rows if row is not None]
    if len(measured) != n_columns:
        raise RuntimeError("slot measurement dropped a column")
    table = pd.DataFrame(measured)
    table_dir = get_script_table_dir(SCRIPT_STEM, group=EXPERIMENT_GROUP)
    csv_path = table_dir / f"{stem}.csv"
    json_path = table_dir / f"{stem}.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(measured, indent=2) + "\n",
        encoding="utf-8",
    )
    display = table[
        [
            "column",
            "int_slots",
            "decimal_slots",
            "digit_slots",
            "n_at_max_decimal_slots",
            "min_value",
            "max_value",
            "example_max_int_padded",
            "example_max_decimal_padded",
        ]
    ]
    print(f"train {len(train)} + test {len(test)} = {len(pooled)} rows")
    print(display.to_string(index=False))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
