# embedding — ids to vectors, and how they combine

*Runtime module (not a frozen level — it carries learnable parameters, which the
data contract calls "later context"). Breakdown axes: B.3, B.4, B.5. See
[workflow.md](workflow.md).*

## In plain terms

`embedding` turns the `inner_table`'s id streams into the `(B, S, D)` tensor the
encoder reads. It holds the embedding tables and the function that fuses them.
It is the first place the model's weights touch the data, so the choices here
are the ones most likely to decide whether training descends at all.

Reads: `token_ids`, `feature_ids`, `slots` (from [inner_table.md](inner_table.md)).
Produces: one `D`-vector per position.

## What it owns

- the **token** table (shape set by the `vocabulary` strategy),
- the **feature** table (one vector per column),
- the **slot** table (one vector per place-value slot, B.4 — shared across
  columns is coherent *because* slots are place-value, not left-index),
- the **composition** that fuses the three.

## Coupling to `vocabulary` (B.3) — state it, don't bury it

- `universal_digits` ⇒ digits shared across columns ⇒ the **feature table is
  required** to tell the model which column a digit belongs to.
- `per_column_digits` ⇒ column identity is already in the token id ⇒ the feature
  table is **redundant** for numerics and may be dropped.

So "which tables exist" is not free here; it follows the vocabulary decision.

## The strategy axis (oracle): composition (B.5)

### `additive` (my lean — start here)

`vec = token + feature + slot`. The honest baseline. Note it is **identical** to
a `nn.Linear` over the concatenation (a linear map of learnable embeddings folds
back into the tables), so "concat + Linear" is not a different option. The
encoder's first layer supplies the downstream nonlinearity.

### `film` (the principled non-additive option)

The slot (and optionally feature) embedding produces a **scale + shift** applied
multiplicatively to the token embedding. This *is* the stated intuition — "a 0 in
the integer slot must not mean the same as a 0 in a decimal slot" is the slot
**modulating** the digit, which addition cannot express. Named ablation, not the
start.

### `mlp`

`concat(token, feature, slot) → Linear → activation → Linear`. Generic
non-additive fusion; more params, less interpretable than FiLM. Lower priority.

## As a composed object (the `Embedder`)

The embedding is a single class passed to the network, not a loose bag of
`nn.Embedding`s inside the encoder. `build_embedder(strategy, args, *, spec)` (the
oracle) returns it; `spec` is the `vocabulary`+`inner_table` product
(`global_vocab_size`, `n_features`, `max_slots`, `feature_index`, `slot_index`).

Because the structural streams are **row-independent** (every row shares the same
column→position layout), the `Embedder` holds `feature_index` / `slot_index` as
buffers and `forward(x)` takes only the `(B, S)` token ids. The network receives
an `Embedder`, not tables — this is the "tokens are a payload to read" boundary:
the Dataset reads the frozen level's payload, the `Embedder` is built from that
level's spec, and nothing downstream touches raw data.

**Decouple from masking via a `[MASK]` vocab id.** Rather than swap
`value → mask_emb` inside the encoder (the current code's murky interaction), let
`[MASK]` be a reserved id in the `vocabulary`. The `Embedder` then just embeds ids
(including `[MASK]`) and adds feature+slot; it never needs to know masking exists.
This also removes a redundancy: the current per-feature `mask_emb[f]` +
`column_emb[f]` is two per-feature vectors summed — a single generic `[MASK]` +
the existing column/slot embeddings carries identical information (and is what
B.2/B.9 decided).

## Deferred

- Whether to **tie** the token table to the reconstruction head
  (see [encoder_and_heads.md](encoder_and_heads.md)).
- Under `per_column_digits`, whether to drop the feature table entirely or keep a
  small one.
- FiLM source: slot-only vs slot+feature modulation.

Links: [[claude-breakdown-notes B.3]], [[claude-breakdown-notes B.4]],
[[claude-breakdown-notes B.5]]. Reads [inner_table.md](inner_table.md);
constrained by [vocabulary.md](vocabulary.md); feeds
[encoder_and_heads.md](encoder_and_heads.md).
