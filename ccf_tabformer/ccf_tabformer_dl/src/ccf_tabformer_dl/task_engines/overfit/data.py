"""One train loader from a frozen inner_table row_subset."""

from __future__ import annotations

from torch.utils.data import DataLoader

from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.data_contract.inner_table_dataset import dataset_from_collections
from ccf_tabformer_dl.data_contract.row_subset.collection import RowSubsetCollection
from ccf_tabformer_dl.tabformer_components.table_spec import FrozenTableSpec
from ccf_tabformer_dl.task_engines.overfit.config import OverfitDataConfig


def load_overfit_collections(
    row_subset: str,
) -> tuple[InnerTableCollection, RowSubsetCollection]:
    subset = RowSubsetCollection.load(row_subset)
    return subset.parent_collection(), subset


def table_spec_from_subset(row_subset: str) -> FrozenTableSpec:
    inner_table, _subset = load_overfit_collections(row_subset)
    return inner_table.table_spec()


def build_overfit_loader(config: OverfitDataConfig) -> DataLoader:
    inner_table, subset = load_overfit_collections(config.row_subset)
    dataset = dataset_from_collections(inner_table=inner_table, subset=subset)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
