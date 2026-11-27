#!/usr/bin/env python3
"""Freeze the declared field_spec collection from the corrected CSVs."""

from __future__ import annotations

import argparse

from ccf_tabformer_dl.data_contract.cleanup.named import cleanup_named_freeze
from ccf_tabformer_dl.data_contract.field_spec.collection import FieldSpecCollection
from ccf_tabformer_dl.data_contract.field_spec.preparation.config import (
    DECLARED_V1,
    load_declared_v1,
)
from ccf_tabformer_dl.data_contract.field_spec.preparation.preparation import FieldSpecPreparation
from ccf_tabformer_dl.file_utils import project_hash_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a field_spec collection."
    )
    parser.add_argument(
        "--name",
        default=DECLARED_V1,
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
    preparation = load_declared_v1()
    if args.force:
        cleanup_named_freeze(
            kind="field_spec",
            name=generating_process_name,
            process_dir=FieldSpecPreparation.process_configurations_dir(),
            collection_dir=FieldSpecCollection.configs_dir(),
            configuration_hash=project_hash_manager().hash_payload(
                preparation.generating_process_signature()
            ),
        )
    collection = preparation.process(
        generating_process_name=generating_process_name
    )
    print(f"hash={collection.collection_hash}")
    print(f"n_columns={len(collection.columns())}")
    print(
        "collection_config="
        f"{FieldSpecCollection.configs_dir().path / (generating_process_name + '.json')}"
    )


if __name__ == "__main__":
    main()
