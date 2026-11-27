"""Torch dataset: index a frozen inner table by a list of row ids."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ccf_tabformer_dl.tabformer_components.reconstruction.steps import IDS_KEY

from ccf_tabformer_dl.data_contract.inner_table.collection import InnerTableCollection
from ccf_tabformer_dl.data_contract.row_subset.collection import RowSubsetCollection

LABEL_KEY = "label"
ROW_ID_KEY = "row_id"


class InnerTableIndexDataset(Dataset):
    """``token_ids[row_index[i]]`` as a long ``(S,)`` tensor under ``ids``."""

    def __init__(self, *, token_ids: np.ndarray, row_index: np.ndarray) -> None:
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be 2-D, got {token_ids.shape}")
        if row_index.ndim != 1:
            raise ValueError(f"row_index must be 1-D, got {row_index.shape}")
        if int(row_index.shape[0]) < 1:
            raise ValueError("row_index must be non-empty")
        n_rows = int(token_ids.shape[0])
        if int(row_index.min()) < 0 or int(row_index.max()) >= n_rows:
            raise ValueError(
                f"row_index values must be in [0, {n_rows}), "
                f"got min={int(row_index.min())} max={int(row_index.max())}"
            )
        self._token_ids = token_ids
        self._row_index = np.asarray(row_index, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._row_index.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = int(self._row_index[index])
        return {IDS_KEY: torch.tensor(self._token_ids[row], dtype=torch.long)}


class InnerTableLabeledDataset(Dataset):
    """``token_ids`` + ``labels`` + ``row_ids`` at ``row_index`` for fine-tuning."""

    def __init__(
        self,
        *,
        token_ids: np.ndarray,
        labels: np.ndarray | None,
        row_ids: np.ndarray,
        row_index: np.ndarray,
    ) -> None:
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be 2-D, got {token_ids.shape}")
        if row_ids.ndim != 1 or int(row_ids.shape[0]) != int(token_ids.shape[0]):
            raise ValueError(
                f"row_ids must be 1-D of length {int(token_ids.shape[0])}, "
                f"got {row_ids.shape}"
            )
        if row_index.ndim != 1:
            raise ValueError(f"row_index must be 1-D, got {row_index.shape}")
        if int(row_index.shape[0]) < 1:
            raise ValueError("row_index must be non-empty")
        n_rows = int(token_ids.shape[0])
        if int(row_index.min()) < 0 or int(row_index.max()) >= n_rows:
            raise ValueError(
                f"row_index values must be in [0, {n_rows}), "
                f"got min={int(row_index.min())} max={int(row_index.max())}"
            )
        if labels is not None:
            if labels.ndim != 1 or int(labels.shape[0]) != n_rows:
                raise ValueError(
                    f"labels must be 1-D of length {n_rows}, got {labels.shape}"
                )
        self._token_ids = token_ids
        self._labels = None if labels is None else np.asarray(labels)
        self._row_ids = np.asarray(row_ids)
        self._row_index = np.asarray(row_index, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._row_index.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        row = int(self._row_index[index])
        item = {
            IDS_KEY: torch.tensor(self._token_ids[row], dtype=torch.long),
            ROW_ID_KEY: torch.tensor(int(self._row_ids[row]), dtype=torch.long),
        }
        if self._labels is not None:
            item[LABEL_KEY] = torch.tensor(
                float(self._labels[row]), dtype=torch.float32
            )
        return item


def dataset_from_collections(
    *,
    inner_table: InnerTableCollection,
    subset: RowSubsetCollection,
) -> InnerTableIndexDataset:
    return InnerTableIndexDataset(
        token_ids=inner_table.token_ids(),
        row_index=np.asarray(subset.row_index()),
    )
