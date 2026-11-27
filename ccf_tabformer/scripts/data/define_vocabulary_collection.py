#!/usr/bin/env python3
"""Freeze a vocabulary collection from a field_spec parent."""

from __future__ import annotations

import argparse

from ccf_tabformer_dl.data_contract.cleanup.named import cleanup_named_freeze
from ccf_tabformer_dl.data_contract.field_spec.collection import FieldSpecCollection
from ccf_tabformer_dl.data_contract.vocabulary.collection import VocabularyCollection
from ccf_tabformer_dl.data_contract.vocabulary.preparation.config import (
    UNIVERSAL_DIGITS_V1,
    load_universal_digits_v1,
)
from ccf_tabformer_dl.data_contract.vocabulary.preparation.preparation import VocabularyPreparation
from ccf_tabformer_dl.file_utils import project_hash_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a vocabulary collection."
    )
    parser.add_argument(
        "--parent",
        required=True,
        help="Parent field_spec generating_process_name or hash.",
    )
    parser.add_argument(
        "--name",
        default=UNIVERSAL_DIGITS_V1,
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
    parent_hash = FieldSpecCollection.load(args.parent).collection_hash
    preparation = load_universal_digits_v1(field_spec_hash=parent_hash)
    if args.force:
        cleanup_named_freeze(
            kind="vocabulary",
            name=generating_process_name,
            process_dir=VocabularyPreparation.process_configurations_dir(),
            collection_dir=VocabularyCollection.configs_dir(),
            configuration_hash=project_hash_manager().hash_payload(
                preparation.generating_process_signature()
            ),
        )
    collection = preparation.process(
        generating_process_name=generating_process_name
    )
    inventory = collection.inventory()
    print(f"hash={collection.collection_hash}")
    print(f"vocab_size={inventory.vocab_size}")
    print(f"mask_id={inventory.mask_id}")
    print(
        "collection_config="
        f"{VocabularyCollection.configs_dir().path / (generating_process_name + '.json')}"
    )


if __name__ == "__main__":
    main()
