"""Loader for the Phase-A digit-sequence token export.

The export (produced by ``experiments/tabformer/export_tokens.py`` in the
competition repo) is a directory holding:

- ``token_codes.npy``  int16 ``(n_total, n_tokens)`` — per-token value code.
  Digit tokens carry the digit ``0..9``; categorical / small-int / flag
  tokens carry that column's value code. Train rows come first, then test.
- ``meta.json``        per-token specs + feature metadata + split sizes.
- ``fold_ids.npy``     int8 ``(n_train,)`` — frozen alarm_profile fold id.
- ``labels.npy``       int8 ``(n_train,)`` — ``is_fraud``.
- ``row_ids.npy``      int64 ``(n_total,)`` — original ``id`` (train then test).

This module only reads and validates; it pulls in no torch, so it imports
without the DL stack. The model builds its embedding tables and per-token
reconstruction heads from :class:`TokenSpec`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOKEN_CODES_FILE = "token_codes.npy"
META_FILE = "meta.json"
FOLD_IDS_FILE = "fold_ids.npy"
LABELS_FILE = "labels.npy"
ROW_IDS_FILE = "row_ids.npy"

TOKEN_KINDS = ("digit", "smallint", "cat", "flag")


@dataclass(frozen=True)
class TokenSpec:
    """One token position in the fixed-length row sequence."""

    name: str
    feature: str
    kind: str
    slot: int
    vocab_size: int
    allowed: tuple[int, ...]


@dataclass(frozen=True)
class TokenizedTable:
    """The full tokenized corpus (train rows first, then test rows)."""

    token_codes: np.ndarray  # int16 (n_total, n_tokens)
    specs: tuple[TokenSpec, ...]
    feature_order: tuple[str, ...]
    fold_ids: np.ndarray  # int8 (n_train,)
    labels: np.ndarray  # int8 (n_train,)
    row_ids: np.ndarray  # int64 (n_total,)
    n_train: int
    n_test: int
    n_folds: int

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_test

    @property
    def n_tokens(self) -> int:
        return len(self.specs)

    @property
    def train_codes(self) -> np.ndarray:
        return self.token_codes[: self.n_train]

    @property
    def test_codes(self) -> np.ndarray:
        return self.token_codes[self.n_train :]


def _spec_from_mapping(raw: dict) -> TokenSpec:
    kind = raw["kind"]
    if kind not in TOKEN_KINDS:
        raise ValueError(f"unknown token kind {kind!r} (expected one of {TOKEN_KINDS})")
    return TokenSpec(
        name=str(raw["name"]),
        feature=str(raw["feature"]),
        kind=str(kind),
        slot=int(raw["slot"]),
        vocab_size=int(raw["vocab_size"]),
        allowed=tuple(int(a) for a in raw["allowed"]),
    )


def load_tokenized_table(*, tokens_dir: Path) -> TokenizedTable:
    """Read and validate the token export directory into a TokenizedTable."""
    tokens_dir = Path(tokens_dir)
    if not tokens_dir.is_dir():
        raise FileNotFoundError(f"token export directory not found: {tokens_dir}")

    meta = json.loads((tokens_dir / META_FILE).read_text(encoding="utf-8"))
    specs = tuple(_spec_from_mapping(m) for m in meta["tokens"])
    token_codes = np.load(tokens_dir / TOKEN_CODES_FILE)
    fold_ids = np.load(tokens_dir / FOLD_IDS_FILE)
    labels = np.load(tokens_dir / LABELS_FILE)
    row_ids = np.load(tokens_dir / ROW_IDS_FILE)

    n_train = int(meta["n_train"])
    n_test = int(meta["n_test"])
    n_folds = int(meta["n_folds"])

    _validate(
        token_codes=token_codes, specs=specs, fold_ids=fold_ids,
        labels=labels, row_ids=row_ids, n_train=n_train, n_test=n_test,
        n_folds=n_folds,
    )

    return TokenizedTable(
        token_codes=token_codes,
        specs=specs,
        feature_order=tuple(str(f) for f in meta["feature_order"]),
        fold_ids=fold_ids,
        labels=labels,
        row_ids=row_ids,
        n_train=n_train,
        n_test=n_test,
        n_folds=n_folds,
    )


def _validate(
    *,
    token_codes: np.ndarray,
    specs: tuple[TokenSpec, ...],
    fold_ids: np.ndarray,
    labels: np.ndarray,
    row_ids: np.ndarray,
    n_train: int,
    n_test: int,
    n_folds: int,
) -> None:
    n_total = n_train + n_test
    if token_codes.shape != (n_total, len(specs)):
        raise ValueError(
            f"token_codes shape {token_codes.shape} != ({n_total}, {len(specs)})"
        )
    if fold_ids.shape != (n_train,):
        raise ValueError(f"fold_ids shape {fold_ids.shape} != ({n_train},)")
    if labels.shape != (n_train,):
        raise ValueError(f"labels shape {labels.shape} != ({n_train},)")
    if row_ids.shape != (n_total,):
        raise ValueError(f"row_ids shape {row_ids.shape} != ({n_total},)")
    if int(fold_ids.max()) + 1 != n_folds:
        raise ValueError(f"fold_ids max+1 {int(fold_ids.max()) + 1} != n_folds {n_folds}")
    # every code must be a valid index into its token's vocab, and within support
    for j, spec in enumerate(specs):
        col = token_codes[:, j]
        lo, hi = int(col.min()), int(col.max())
        if lo < 0 or hi >= spec.vocab_size:
            raise ValueError(
                f"token {spec.name!r} codes [{lo},{hi}] out of vocab [0,{spec.vocab_size})"
            )
        if not set(np.unique(col)).issubset(set(spec.allowed)):
            raise ValueError(f"token {spec.name!r} has codes outside its allowed support")
