"""Global token vocabulary + per-position support mask.

The user's "single vocabulary, but predict only the valid indices per column"
is realised here. Every token position is mapped into ONE global value
vocabulary:

- digit slots share the ids ``0..9`` (so the digit ``3`` is the same embedding
  at every slot — the model learns "digit 3", and position is carried by the
  column/slot embeddings, not the value id);
- each non-digit column (categorical / small-int / flag) gets its own
  contiguous block of ids above ``10``, so its codes never collide with a
  digit or another column.

For each position we also record the set of *allowed* global ids (its support
over train+test). The reconstruction head masks every disallowed logit to
``-inf`` at that position, so the model can only ever predict a value the
generator actually wrote there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ccf_tabformer_dl.data_contract.tokens import TokenSpec

DIGIT_VOCAB_SIZE = 10  # digits 0..9, shared across all digit slots
NEG_INF = float("-inf")


@dataclass(frozen=True)
class GlobalVocab:
    """Mapping from per-token local codes into one global value vocabulary."""

    global_vocab_size: int
    offsets: np.ndarray  # int64 (n_tokens,) added to a local code to get the global id
    feature_index: np.ndarray  # int64 (n_tokens,) owning-feature index into feature_order
    slot_index: np.ndarray  # int64 (n_tokens,) digit slot within the number (0 for non-digit)
    feature_order: tuple[str, ...]
    n_features: int
    max_slots: int
    support_allow: np.ndarray  # bool (n_tokens, global_vocab_size) True where a value is allowed

    @property
    def n_tokens(self) -> int:
        return int(self.offsets.shape[0])

    def support_additive(self) -> np.ndarray:
        """(n_tokens, global_vocab_size) float: 0.0 where allowed, -inf elsewhere."""
        additive = np.where(self.support_allow, 0.0, NEG_INF).astype(np.float32)
        return additive


def build_global_vocab(specs: tuple[TokenSpec, ...]) -> GlobalVocab:
    feature_order: list[str] = []
    for spec in specs:
        if spec.feature not in feature_order:
            feature_order.append(spec.feature)
    feature_to_index = {name: i for i, name in enumerate(feature_order)}

    n_tokens = len(specs)
    offsets = np.zeros(n_tokens, dtype=np.int64)
    feature_index = np.zeros(n_tokens, dtype=np.int64)
    slot_index = np.zeros(n_tokens, dtype=np.int64)

    next_offset = DIGIT_VOCAB_SIZE
    for j, spec in enumerate(specs):
        feature_index[j] = feature_to_index[spec.feature]
        slot_index[j] = spec.slot
        if spec.kind == "digit":
            if spec.vocab_size != DIGIT_VOCAB_SIZE:
                raise ValueError(
                    f"digit token {spec.name!r} must have vocab {DIGIT_VOCAB_SIZE}, "
                    f"got {spec.vocab_size}"
                )
            offsets[j] = 0
        else:
            offsets[j] = next_offset
            next_offset += spec.vocab_size
    global_vocab_size = int(next_offset)

    support_allow = np.zeros((n_tokens, global_vocab_size), dtype=bool)
    for j, spec in enumerate(specs):
        for local_code in spec.allowed:
            support_allow[j, int(offsets[j]) + int(local_code)] = True

    return GlobalVocab(
        global_vocab_size=global_vocab_size,
        offsets=offsets,
        feature_index=feature_index,
        slot_index=slot_index,
        feature_order=tuple(feature_order),
        n_features=len(feature_order),
        max_slots=int(slot_index.max()) + 1,
        support_allow=support_allow,
    )


def to_global_ids(local_codes: np.ndarray, *, vocab: GlobalVocab) -> np.ndarray:
    """Map a (n_rows, n_tokens) local-code matrix to global ids (int64)."""
    if local_codes.ndim != 2 or local_codes.shape[1] != vocab.n_tokens:
        raise ValueError(
            f"local_codes shape {local_codes.shape} incompatible with "
            f"n_tokens {vocab.n_tokens}"
        )
    return local_codes.astype(np.int64) + vocab.offsets[None, :]
