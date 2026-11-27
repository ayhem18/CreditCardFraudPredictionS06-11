#!/usr/bin/env python3
"""Freeze a row-index subset of a frozen inner_table."""

from __future__ import annotations

import argparse

from ccf_tabformer_dl.data_contract.cleanup.named import cleanup_named_freeze
from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.data_contract.row_subset.collection import RowSubsetCollection
from ccf_tabformer_dl.data_contract.row_subset.preparation.config import (
    ALARM_LABEL_1K_SEED0,
    NAMED_SUBSET_CONFIGS,
    load_named_subset,
)
from ccf_tabformer_dl.data_contract.row_subset.preparation.preparation import RowSubsetPreparation
from ccf_tabformer_dl.file_utils import project_hash_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a row_subset collection."
    )
    parser.add_argument(
        "--parent",
        required=True,
        help="Parent inner_table generating_process_name or hash.",
    )
    parser.add_argument(
        "--name",
        default=ALARM_LABEL_1K_SEED0,
        choices=tuple(NAMED_SUBSET_CONFIGS),
        help="generating_process_name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete the named freeze first, including whichever name already "
            "holds this signature. process() still has no overwrite."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generating_process_name = args.name
    parent_hash = InnerTableCollection.load(args.parent).collection_hash
    preparation = load_named_subset(
        inner_table_hash=parent_hash,
        generating_process_name=generating_process_name,
    )
    if args.force:
        cleanup_named_freeze(
            kind="row_subset",
            name=generating_process_name,
            process_dir=RowSubsetPreparation.process_configurations_dir(),
            collection_dir=RowSubsetCollection.configs_dir(),
            payload_dir=RowSubsetCollection.artifacts_dir(generating_process_name),
            configuration_hash=project_hash_manager().hash_payload(
                preparation.generating_process_signature()
            ),
        )
    collection = preparation.process(
        generating_process_name=generating_process_name
    )
    print(f"hash={collection.collection_hash}")
    print(f"n_rows={collection.n_rows()}")
    print(
        "collection_config="
        f"{RowSubsetCollection.configs_dir().path / (generating_process_name + '.json')}"
    )
    print(f"payload={RowSubsetCollection.artifacts_dir(generating_process_name)}")


if __name__ == "__main__":
    main()
