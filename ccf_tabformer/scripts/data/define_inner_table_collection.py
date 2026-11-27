#!/usr/bin/env python3
"""Tokenize the full corrected table and freeze the inner_table numpy payload."""

from __future__ import annotations

import argparse

from ccf_tabformer_dl.data_contract.cleanup.named import cleanup_named_freeze
from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.data_contract.inner_table.preparation.config import (
    FIXED_WIDTH_V1,
    load_fixed_width_v1,
)
from ccf_tabformer_dl.data_contract.inner_table.preparation.preparation import InnerTablePreparation
from ccf_tabformer_dl.data_contract.vocabulary.collection import VocabularyCollection
from ccf_tabformer_dl.file_utils import project_hash_manager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze an inner_table collection."
    )
    parser.add_argument(
        "--parent",
        required=True,
        help="Parent vocabulary generating_process_name or hash.",
    )
    parser.add_argument(
        "--name",
        default=FIXED_WIDTH_V1,
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
    parent_hash = VocabularyCollection.load(args.parent).collection_hash
    preparation = load_fixed_width_v1(vocabulary_hash=parent_hash)
    if args.force:
        cleanup_named_freeze(
            kind="inner_table",
            name=generating_process_name,
            process_dir=InnerTablePreparation.process_configurations_dir(),
            collection_dir=InnerTableCollection.configs_dir(),
            payload_dir=InnerTableCollection.artifacts_dir(generating_process_name),
            configuration_hash=project_hash_manager().hash_payload(
                preparation.generating_process_signature()
            ),
        )
    collection = preparation.process(
        generating_process_name=generating_process_name
    )
    spec = collection.table_spec()
    print(f"hash={collection.collection_hash}")
    print(f"n_rows={collection.n_rows()}")
    print(f"seq_len={spec.seq_len}")
    print(f"vocab_size={spec.vocab_size}")
    print(
        "collection_config="
        f"{InnerTableCollection.configs_dir().path / (generating_process_name + '.json')}"
    )
    print(f"payload={InnerTableCollection.artifacts_dir(generating_process_name)}")


if __name__ == "__main__":
    main()
