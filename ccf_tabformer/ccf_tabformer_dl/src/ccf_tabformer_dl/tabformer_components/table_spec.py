"""The structural spec the runtime modules read.

`TableSpec` is the row-independent layout produced (eventually) by the frozen
`vocabulary` + `inner_table` levels — see
`documentation/decisions/{vocabulary,inner_table}.md`. The `Embedder` and the
`Masker` are both built from a `TableSpec` and share nothing else, so they stay
decoupled from *how* the spec was produced. Any object exposing these members
satisfies the contract — including a thin adapter over the current numpy export,
which is why the modules do not import that (problematic) export at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class TableSpec(Protocol):
    @property
    def seq_len(self) -> int:
        """S — the fixed number of token positions in one row."""

    @property
    def n_features(self) -> int:
        """Number of distinct feature columns."""

    @property
    def max_slots(self) -> int:
        """Number of distinct place-value slots (B.4), size of the slot table."""

    @property
    def vocab_size(self) -> int:
        """Size of the value-embedding table, including the reserved `[MASK]` id."""

    @property
    def mask_id(self) -> int:
        """The reserved `[MASK]` value id, in `[0, vocab_size)`."""

    def support_additive(self) -> Tensor:
        """float `(seq_len, vocab_size)`: 0 where a value id is allowed at that
        position, -inf elsewhere. The reconstruction head adds it to its logits so
        it can only ever predict a value the generator writes at that position.
        """

    @property
    def feature_index(self) -> Tensor:
        """long `(S,)`: the owning feature of each position (ties a number's digits)."""

    @property
    def slot_index(self) -> Tensor:
        """long `(S,)`: the place-value slot of each position (0 for one-token columns)."""


class FrozenTableSpec:
    """A concrete `TableSpec` held as CPU tensors.

    Built by the frozen `inner_table` level (or by tests). Runtime modules copy
    the structural streams into buffers; they do not keep this object alive as
    the source of truth after construction.
    """

    def __init__(
        self,
        *,
        seq_len: int,
        n_features: int,
        max_slots: int,
        vocab_size: int,
        mask_id: int,
        feature_index: Tensor,
        slot_index: Tensor,
        support_additive: Tensor,
    ) -> None:
        _require_positive_int(seq_len, label="seq_len")
        _require_positive_int(n_features, label="n_features")
        _require_positive_int(max_slots, label="max_slots")
        _require_positive_int(vocab_size, label="vocab_size")
        if not isinstance(mask_id, int) or not 0 <= mask_id < vocab_size:
            raise ValueError(
                f"mask_id must be in [0, vocab_size), got mask_id={mask_id!r} "
                f"vocab_size={vocab_size!r}"
            )
        feature = torch.as_tensor(feature_index, dtype=torch.long).reshape(-1)
        slot = torch.as_tensor(slot_index, dtype=torch.long).reshape(-1)
        support = torch.as_tensor(support_additive, dtype=torch.float32)
        if tuple(feature.shape) != (seq_len,):
            raise ValueError(
                f"feature_index must have shape ({seq_len},), got {tuple(feature.shape)}"
            )
        if tuple(slot.shape) != (seq_len,):
            raise ValueError(
                f"slot_index must have shape ({seq_len},), got {tuple(slot.shape)}"
            )
        if tuple(support.shape) != (seq_len, vocab_size):
            raise ValueError(
                f"support_additive must have shape ({seq_len}, {vocab_size}), "
                f"got {tuple(support.shape)}"
            )
        if int(feature.min()) < 0 or int(feature.max()) >= n_features:
            raise ValueError(
                f"feature_index values must be in [0, {n_features}), "
                f"got min={int(feature.min())} max={int(feature.max())}"
            )
        if int(slot.min()) < 0 or int(slot.max()) >= max_slots:
            raise ValueError(
                f"slot_index values must be in [0, {max_slots}), "
                f"got min={int(slot.min())} max={int(slot.max())}"
            )
        self._seq_len = seq_len
        self._n_features = n_features
        self._max_slots = max_slots
        self._vocab_size = vocab_size
        self._mask_id = mask_id
        self._feature_index = feature.contiguous()
        self._slot_index = slot.contiguous()
        self._support_additive = support.contiguous()

    @property
    def seq_len(self) -> int:
        return self._seq_len

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def max_slots(self) -> int:
        return self._max_slots

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def mask_id(self) -> int:
        return self._mask_id

    def support_additive(self) -> Tensor:
        return self._support_additive

    @property
    def feature_index(self) -> Tensor:
        return self._feature_index

    @property
    def slot_index(self) -> Tensor:
        return self._slot_index


def support_additive_from_allowed(
    *,
    seq_len: int,
    vocab_size: int,
    allowed_ids: Sequence[Sequence[int]],
) -> Tensor:
    """Build the `(seq_len, vocab_size)` 0 / `-inf` support matrix."""
    _require_positive_int(seq_len, label="seq_len")
    _require_positive_int(vocab_size, label="vocab_size")
    n_allowed = len(allowed_ids)
    if n_allowed != seq_len:
        raise ValueError(
            f"allowed_ids length must equal seq_len {seq_len}, got {n_allowed}"
        )
    support = torch.full((seq_len, vocab_size), float("-inf"), dtype=torch.float32)
    for index in range(seq_len):
        raw = allowed_ids[index]
        n_ids = len(raw)
        if n_ids < 1:
            raise ValueError(f"allowed_ids[{index}] must be non-empty")
        ids = [0] * n_ids
        for j in range(n_ids):
            token_id = raw[j]
            if not isinstance(token_id, int) or not 0 <= token_id < vocab_size:
                raise ValueError(
                    f"allowed_ids[{index}][{j}] must be in [0, {vocab_size}), "
                    f"got {token_id!r}"
                )
            ids[j] = token_id
        support[index, ids] = 0.0
    return support


def _require_positive_int(value: object, *, label: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive int, got {value!r}")
