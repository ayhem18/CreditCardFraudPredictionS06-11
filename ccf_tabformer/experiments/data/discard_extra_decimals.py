"""Write the working CSV copy that drops singleton extra decimal digits.

Decision (see ``documentation/motivation.md``): for each continuous column,
if the max written decimal width is achieved by a single row, discard that
last decimal digit. The original ``storage/raw_data/`` copy is left unchanged.
Later analysis and training read ``storage/data/corrected/``.

Run (from ``ccf_tabformer``, project venv activated)::

    python experiments/data/discard_extra_decimals.py
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from ccf_tabformer_dl.file_utils import (
    get_corrected_manifest_path,
    get_corrected_test_csv_path,
    get_corrected_train_csv_path,
    get_project_root,
    get_raw_test_csv_path,
    get_raw_train_csv_path,
)

sys.path.insert(0, str(get_project_root()))

from experiments.data.digit_slots import CONTINUOUS_COLUMNS, split_written_parts

ID_COLUMN = "id"
CSV_LINE_TERMINATOR = "\n"


def drop_last_decimal_digit(*, text: str) -> str:
    """Drop the last digit of the decimal part of a written number."""
    stripped = text.strip()
    if "." not in stripped:
        raise ValueError(f"no decimal point to trim: {text!r}")
    integer_part, decimal_part = stripped.split(".", 1)
    if len(decimal_part) < 1:
        raise ValueError(f"empty decimal part: {text!r}")
    truncated = decimal_part[:-1]
    if truncated == "":
        return integer_part
    return integer_part + "." + truncated


def apply_singleton_decimal_edits(
    *, train: pd.DataFrame, test: pd.DataFrame
) -> list[dict[str, str]]:
    """Drop last decimal digits whose max width is unique on pooled train+test."""
    if ID_COLUMN not in train.columns or ID_COLUMN not in test.columns:
        raise ValueError(f"missing {ID_COLUMN!r} column")
    n_train = len(train)
    n_columns = len(CONTINUOUS_COLUMNS)
    edit_slots: list[dict[str, str] | None] = [None] * n_columns
    n_edits = 0
    for index in range(n_columns):
        column = CONTINUOUS_COLUMNS[index]
        if column not in train.columns or column not in test.columns:
            raise ValueError(f"missing column {column!r}")
        pooled = pd.concat([train[column], test[column]], ignore_index=True)
        _, decimal_parts = split_written_parts(texts=pooled, column=column)
        dec_lengths = decimal_parts.str.len()
        decimal_slots = int(dec_lengths.max())
        at_max = dec_lengths == decimal_slots
        n_at_max = int(at_max.sum())
        if n_at_max != 1:
            continue
        pooled_index = int(at_max.to_numpy().nonzero()[0][0])
        if pooled_index < n_train:
            split_name = "train"
            frame = train
            row_index = pooled_index
        else:
            split_name = "test"
            frame = test
            row_index = pooled_index - n_train
        before = str(frame.iloc[row_index][column])
        after = drop_last_decimal_digit(text=before)
        frame.iloc[row_index, frame.columns.get_loc(column)] = after
        edit_slots[n_edits] = {
            "split": split_name,
            "id": str(frame.iloc[row_index][ID_COLUMN]),
            "column": column,
            "before": before,
            "after": after,
        }
        n_edits += 1
    edits = [edit for edit in edit_slots[:n_edits] if edit is not None]
    if len(edits) != n_edits:
        raise RuntimeError("edit log lost a row")
    return edits


def _write_csv(*, frame: pd.DataFrame, path: object) -> None:
    frame.to_csv(path, index=False, lineterminator=CSV_LINE_TERMINATOR)


def _assert_only_logged_edits(
    *,
    original: pd.DataFrame,
    corrected: pd.DataFrame,
    edits: list[dict[str, str]],
    split: str,
) -> None:
    if original.shape != corrected.shape:
        raise ValueError(
            f"{split}: shape {original.shape} != corrected {corrected.shape}"
        )
    if list(original.columns) != list(corrected.columns):
        raise ValueError(f"{split}: column names changed")
    changed = original != corrected
    expected = pd.DataFrame(False, index=original.index, columns=original.columns)
    id_to_row = {str(value): index for index, value in original[ID_COLUMN].items()}
    n_edits = len(edits)
    for i in range(n_edits):
        edit = edits[i]
        row_index = id_to_row[edit["id"]]
        column = edit["column"]
        expected.loc[row_index, column] = True
        got = str(corrected.loc[row_index, column])
        if got != edit["after"]:
            raise ValueError(
                f"{split} id={edit['id']} {column}: expected {edit['after']!r}, got {got!r}"
            )
    unexpected = changed & ~expected
    if bool(unexpected.any().any()):
        bad_column = unexpected.any()
        column = str(bad_column.index[bad_column][0])
        raise ValueError(f"{split}: unexpected cell change in {column!r}")
    missing = expected & ~changed
    if bool(missing.any().any()):
        raise ValueError(f"{split}: logged edit did not change a cell")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing corrected copy.",
    )
    args = parser.parse_args()
    train_out = get_corrected_train_csv_path()
    test_out = get_corrected_test_csv_path()
    manifest_path = get_corrected_manifest_path()
    if not args.force and (train_out.is_file() or test_out.is_file()):
        raise FileExistsError(
            f"corrected CSVs already exist under {train_out.parent}. "
            "Pass --force to overwrite."
        )

    train = pd.read_csv(get_raw_train_csv_path(), dtype=str)
    test = pd.read_csv(get_raw_test_csv_path(), dtype=str)
    edits = apply_singleton_decimal_edits(train=train, test=test)
    n_edits = len(edits)
    train_edits = [edit for edit in edits if edit["split"] == "train"]
    test_edits = [edit for edit in edits if edit["split"] == "test"]
    n_train_edits = len(train_edits)
    n_test_edits = len(test_edits)
    if n_train_edits + n_test_edits != n_edits:
        raise RuntimeError("edit split counts do not cover the log")

    _write_csv(frame=train, path=train_out)
    _write_csv(frame=test, path=test_out)

    written_train = pd.read_csv(train_out, dtype=str)
    written_test = pd.read_csv(test_out, dtype=str)
    raw_train = pd.read_csv(get_raw_train_csv_path(), dtype=str)
    raw_test = pd.read_csv(get_raw_test_csv_path(), dtype=str)
    _assert_only_logged_edits(
        original=raw_train, corrected=written_train, edits=train_edits, split="train"
    )
    _assert_only_logged_edits(
        original=raw_test, corrected=written_test, edits=test_edits, split="test"
    )

    manifest = {
        "rule": (
            "For each continuous column, if the max written decimal width is "
            "achieved by a single row, drop that last decimal digit."
        ),
        "source": "storage/raw_data",
        "output": "storage/data/corrected",
        "n_train": int(len(written_train)),
        "n_test": int(len(written_test)),
        "edits": edits,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"train edits: {n_train_edits} | test edits: {n_test_edits}")
    n_print = len(edits)
    for i in range(n_print):
        edit = edits[i]
        print(
            f"  {edit['split']} id={edit['id']} {edit['column']}: "
            f"{edit['before']!r} -> {edit['after']!r}"
        )
    print(f"wrote {train_out}")
    print(f"wrote {test_out}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
