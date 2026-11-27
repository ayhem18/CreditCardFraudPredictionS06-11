# vocabulary — the global token inventory

*Frozen data-contract level. Breakdown axes: B.3, B.9, plus the B.2 mask-token
sub-decision. See [workflow.md](workflow.md).*

## In plain terms

`vocabulary` is the fixed list of token ids the model will embed. It turns the
`field_spec` into a closed inventory: every digit, every categorical value, every
boolean, and the special tokens. `inner_table` later maps each written cell to
ids *from this inventory*; nothing invents an id outside it.

It consumes `field_spec` and produces the id↔meaning table. Identity =
`field_spec` hash + the vocab strategy signature.

## What it owns

- the id of every token the model can see;
- which tokens are **shared across columns** and which are **column-specific**
  (the B.3 decision, below) — this is the vocabulary's defining choice;
- the **special tokens**, chiefly the mask token(s) (B.2 sub-decision).

It does **not** own how a value becomes ids per row (that is `inner_table`), nor
how ids become vectors (that is `embedding`).

## The strategy axis (oracle): B.3

The strategy is the hypothesis *"does a digit mean the same thing in every
column?"*

### `universal_digits` (my lean — start here)

Digits `0–9` are shared by all numeric columns. Categorical columns and
in_between_as_cat columns get contiguous global id blocks in `field_spec`
order; per-flag booleans follow those columns; specials last. Because digits are
shared, the **feature embedding is required downstream** (B.3) to tell the model
which column a digit belongs to. Cheaper, and encodes the "numbers are the
decimal system" prior.

### `per_column_digits` (the named ablation)

Each column's digits get their own id block, so column identity is baked into the
id and no feature embedding is needed for numerics. More ids, no cross-column
transfer, more flexibility. Kept as the falsifier for the universal prior.

## Sub-decisions living here

- **Booleans (B.9):** shared `False`/`True` vs per-flag. **Decided: per-flag** —
  6 flags × 2, negligible cost, and `used_vpn=True` ≠ `is_foreign=True`. This is
  the vocab's `boolean_policy`.
- **Mask token (B.2):** one generic `[MASK]`, or one per input type (digit vs
  categorical). **Lean: one generic**, because masked positions keep their
  feature + slot embeddings (see [masking.md](masking.md)), which already route
  "what to predict." Specials are part of this inventory so their ids are fixed
  and frozen with it.

## Deferred

- Whether in_between_as_num columns share the `0–9` block with true numerics, or
  get their own digit block even under `universal_digits` (the hour-vs-amount
  strain, B.6/B.3). Default: share; revisit if it hurts.

Links: [[claude-breakdown-notes B.3]], [[claude-breakdown-notes B.9]]. Consumes
[field_spec.md](field_spec.md); consumed by [inner_table.md](inner_table.md) and
constrains [embedding.md](embedding.md).
