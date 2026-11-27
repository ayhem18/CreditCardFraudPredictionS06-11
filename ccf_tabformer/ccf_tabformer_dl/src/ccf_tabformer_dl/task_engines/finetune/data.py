"""Train/val/test loaders from inner_table + imported folds."""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader

from ccf_tabformer_dl.data_contract.corrected_table import LABEL_MISSING
from ccf_tabformer_dl.data_contract.folds.collection import FoldsCollection
from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.data_contract.inner_table_dataset import InnerTableLabeledDataset
from ccf_tabformer_dl.tabformer_components.table_spec import FrozenTableSpec
from ccf_tabformer_dl.task_engines.finetune.config import FineTuneDataConfig


def load_finetune_collections(
    *, inner_table: str, folds: str
) -> tuple[InnerTableCollection, FoldsCollection]:
    table = InnerTableCollection.load(inner_table)
    fold_set = FoldsCollection.load(folds)
    parent_hash = fold_set.generating_process_signature["args"]["inner_table_hash"]
    if parent_hash != table.collection_hash:
        raise ValueError(
            "folds inner_table_hash does not match loaded inner_table: "
            f"{parent_hash} vs {table.collection_hash}"
        )
    return table, fold_set


def table_spec_from_inner_table(inner_table: str) -> FrozenTableSpec:
    return InnerTableCollection.load(inner_table).table_spec()


def build_indexed_loader(
    *,
    table: InnerTableCollection,
    row_index: np.ndarray,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    with_labels: bool,
) -> DataLoader:
    labels = None
    if with_labels:
        labels = table.labels()
        selected = labels[row_index]
        if int(selected.shape[0]) > 0 and bool((selected == LABEL_MISSING).any()):
            raise ValueError("train/val fold contains missing labels")
    dataset = InnerTableLabeledDataset(
        token_ids=table.token_ids(),
        labels=labels,
        row_ids=table.row_ids(),
        row_index=row_index,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def build_train_val_loaders(
    config: FineTuneDataConfig,
) -> tuple[DataLoader, DataLoader]:
    table, fold_set = load_finetune_collections(
        inner_table=config.inner_table, folds=config.folds
    )
    if config.fold >= fold_set.n_splits():
        raise ValueError(
            f"data.fold {config.fold} is outside n_splits={fold_set.n_splits()}"
        )
    train = build_indexed_loader(
        table=table,
        row_index=fold_set.train_row_index(held_out_fold=config.fold),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        with_labels=True,
    )
    val = build_indexed_loader(
        table=table,
        row_index=fold_set.val_row_index(held_out_fold=config.fold),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        with_labels=True,
    )
    return train, val


def build_eval_loaders(
    config: FineTuneDataConfig,
) -> tuple[DataLoader, DataLoader]:
    table, fold_set = load_finetune_collections(
        inner_table=config.inner_table, folds=config.folds
    )
    if config.fold >= fold_set.n_splits():
        raise ValueError(
            f"data.fold {config.fold} is outside n_splits={fold_set.n_splits()}"
        )
    val = build_indexed_loader(
        table=table,
        row_index=fold_set.val_row_index(held_out_fold=config.fold),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        with_labels=True,
    )
    test = build_indexed_loader(
        table=table,
        row_index=fold_set.test_row_index(),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        with_labels=False,
    )
    return val, test
