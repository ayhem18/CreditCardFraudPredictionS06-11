"""Train and val loaders from a frozen inner_table split column."""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from ccf_tabformer_dl.data_contract.corrected_table import SPLIT_TEST, SPLIT_TRAIN
from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.data_contract.inner_table_dataset import InnerTableIndexDataset
from ccf_tabformer_dl.tabformer_components.table_spec import FrozenTableSpec
from ccf_tabformer_dl.task_engines.full_run.config import FullRunDataConfig


def load_inner_table(inner_table: str) -> InnerTableCollection:
    return InnerTableCollection.load(inner_table)


def table_spec_from_inner_table(inner_table: str) -> FrozenTableSpec:
    return load_inner_table(inner_table).table_spec()


def row_index_for_split(split: np.ndarray, *, split_value: int) -> np.ndarray:
    """Row positions whose sidecar split marker equals ``split_value``."""
    if split.ndim != 1:
        raise ValueError(f"split must be 1-D, got {split.shape}")
    index = np.flatnonzero(np.asarray(split) == split_value).astype(np.int64)
    if int(index.shape[0]) < 1:
        raise ValueError(f"split value {split_value} has no rows")
    return index


def build_split_loader(
    *,
    inner_table: InnerTableCollection,
    split_value: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = InnerTableIndexDataset(
        token_ids=inner_table.token_ids(),
        row_index=row_index_for_split(
            inner_table.split(), split_value=split_value
        ),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def build_train_val_loaders(config: FullRunDataConfig) -> tuple[DataLoader, DataLoader]:
    collection = load_inner_table(config.inner_table)
    train = build_split_loader(
        inner_table=collection,
        split_value=SPLIT_TRAIN,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )
    val = build_split_loader(
        inner_table=collection,
        split_value=SPLIT_TEST,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    return train, val
