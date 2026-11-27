"""Map each categorical column to its unique written values and stable ids.

Vocab is built label-free from the pooled train+test working copy
(``storage/data/corrected/``). Ids are the position in a lexicographic sort,
so the mapping does not depend on row order.

Run (from ``ccf_tabformer``, project venv activated)::

    python experiments/data/categorical_values.py

Writes ``storage/artifacts/data/categorical_values/tables/``.
"""

from __future__ import annotations

import json

import pandas as pd

from ccf_tabformer_dl.file_utils import (
    get_script_table_dir,
    get_test_csv_path,
    get_train_csv_path,
)

CATEGORICAL_COLUMNS = (
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
)

SCRIPT_STEM = "categorical_values"
EXPERIMENT_GROUP = "data"


def map_column(
    *,
    train: pd.Series,
    test: pd.Series,
    column: str,
) -> list[dict[str, object]]:
    """Return one row per unique written value, with a stable id and split counts."""
    train_text = train.astype(str).str.strip()
    test_text = test.astype(str).str.strip()
    if train_text.eq("").any() or train_text.isna().any():
        raise ValueError(f"{column}: empty or missing train values")
    if test_text.eq("").any() or test_text.isna().any():
        raise ValueError(f"{column}: empty or missing test values")
    train_counts = train_text.value_counts()
    test_counts = test_text.value_counts()
    unique_values = sorted(set(train_counts.index).union(set(test_counts.index)))
    n_values = len(unique_values)
    if n_values < 1:
        raise ValueError(f"{column}: no unique values")
    rows: list[dict[str, object] | None] = [None] * n_values
    for index in range(n_values):
        value = unique_values[index]
        n_train = int(train_counts.get(value, 0))
        n_test = int(test_counts.get(value, 0))
        rows[index] = {
            "column": column,
            "id": index,
            "value": value,
            "n_train": n_train,
            "n_test": n_test,
            "n_total": n_train + n_test,
            "train_only": n_train > 0 and n_test == 0,
            "test_only": n_test > 0 and n_train == 0,
        }
    mapped = [row for row in rows if row is not None]
    if len(mapped) != n_values:
        raise RuntimeError(f"{column}: dropped a unique value")
    return mapped


def main() -> None:
    columns = list(CATEGORICAL_COLUMNS)
    train = pd.read_csv(get_train_csv_path(), dtype=str, usecols=columns)
    test = pd.read_csv(get_test_csv_path(), dtype=str, usecols=columns)
    n_columns = len(columns)
    per_column: list[list[dict[str, object]] | None] = [None] * n_columns
    n_rows = 0
    for index in range(n_columns):
        column = columns[index]
        mapped = map_column(train=train[column], test=test[column], column=column)
        per_column[index] = mapped
        n_rows += len(mapped)
    filled = [block for block in per_column if block is not None]
    if len(filled) != n_columns:
        raise RuntimeError("dropped a categorical column")
    table_rows: list[dict[str, object] | None] = [None] * n_rows
    cursor = 0
    vocab: dict[str, dict[str, int]] = {}
    for index in range(n_columns):
        block = filled[index]
        n_block = len(block)
        for offset in range(n_block):
            table_rows[cursor + offset] = block[offset]
        cursor += n_block
        vocab[columns[index]] = {str(row["value"]): int(row["id"]) for row in block}
    measured = [row for row in table_rows if row is not None]
    if len(measured) != n_rows:
        raise RuntimeError("flattened mapping lost a row")
    table = pd.DataFrame(measured)
    table_dir = get_script_table_dir(SCRIPT_STEM, group=EXPERIMENT_GROUP)
    csv_path = table_dir / "categorical_values.csv"
    json_path = table_dir / "categorical_values.json"
    table.to_csv(csv_path, index=False)
    payload = {
        "source": "storage/data/corrected",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "columns": columns,
        "vocab": vocab,
        "rows": measured,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"train {len(train)} + test {len(test)}")
    print(table.to_string(index=False))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
