"""One labeled train loader from a frozen inner_table row_subset."""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.data_contract.row_subset.collection import RowSubsetCollection
from ccf_tabformer_dl.tabformer_components.table_spec import FrozenTableSpec
from ccf_tabformer_dl.task_engines.finetune.data import build_indexed_loader
from ccf_tabformer_dl.task_engines.finetune_overfit.config import FineTuneOverfitDataConfig


def load_overfit_collections(
    row_subset: str,
) -> tuple[InnerTableCollection, RowSubsetCollection]:
    subset = RowSubsetCollection.load(row_subset)
    return subset.parent_collection(), subset


def table_spec_from_subset(row_subset: str) -> FrozenTableSpec:
    inner_table, _subset = load_overfit_collections(row_subset)
    return inner_table.table_spec()


def build_finetune_overfit_loader(config: FineTuneOverfitDataConfig) -> DataLoader:
    inner_table, subset = load_overfit_collections(config.row_subset)
    return build_indexed_loader(
        table=inner_table,
        row_index=np.asarray(subset.row_index()),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        with_labels=True,
    )
