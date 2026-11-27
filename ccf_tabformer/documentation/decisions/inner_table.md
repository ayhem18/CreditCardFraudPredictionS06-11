# inner_table — rows as fixed-length sequences

*Frozen data-contract level (decided Q2). Breakdown axes: B.1, B.3, B.4, and the
B.7 permutation note. See [workflow.md](workflow.md).*

## In plain terms

`inner_table` is the actual `(rows × S)` substrate the model reads. For each row
it applies `field_spec` + `vocabulary` to produce, per row, **four aligned
streams of length S**:

1. **token_ids** — the value's tokens (digits for numeric/in_between_as_num; one
   id for categorical/boolean/in_between_as_cat).
2. **feature_ids** — which column each position belongs to (ties a number's
   digit positions together, B.3).
3. **slots** — place-value index for digit positions (B.4), a single "not a
   digit" slot for one-token columns.
4. **real/pad mask** — which positions are genuine vs structural padding.

`S` is fixed: `sum over columns of (digit_slots if numeric/in_between_as_num else
1)`. The value→id mapping (Q1) is folded in here — this level *is* the tokenizer
applied.

Identity = `corrected_table` hash + `vocabulary` hash + layout signature. The
collection payload is the **full** encoded table as numpy (`token_ids`,
`row_ids`, `split`, `labels` sidecar). A row is an index into that array, not a
JSON entity (1M JSON members is the wrong grain). Feature/slot/support streams
are row-independent and live on the layout / `FrozenTableSpec`, not per row.

A **subset** is a later data level: a list of row indices into this array. It
does not re-tokenize.

## Two things this level fixes, by design

- **No row-level position embedding (B.7).** A row is a *set* of columns, not a
  sequence. Column identity rides in `feature_ids`; digit order rides in `slots`.
  Feature order along `S` is therefore arbitrary and carries no meaning — the
  model must not be given a positional embedding that implies one. (This is the
  likely culprit behind the vibe-coded loss going up; it is designed out here.)
- **Slots are place-value, not left-index (B.4).** "the tens digit" gets the
  same `slot` id in every column, so a shared slot table is coherent. Zero-pad is
  place-value aligned: `23.56` with `int_slots=4, decimal_slots=2` → positions
  for thousands(0) hundreds(0) tens(2) ones(3) tenths(5) hundredths(6).

## The label is NOT in the inner table

Because the chosen path is **unsupervised reconstruction + finetuning** (not
label-as-field), `is_fraud` is *not* a position in the sequence. The inner table
encodes **features only**; the label rides alongside as a separate per-row field
used only by the finetuning head. This settles the feature-side of B.8 for the
chosen path: no reserved label slot, no train/test sequence asymmetry.

## The strategy axis (oracle)

The strategy is the **layout** — how cells become the four streams.

### `fixed_width_padded` (the only strategy for now)

Zero-pad each numeric to its column's `digit_slots` (place-value aligned); one
position per categorical/boolean/in_between_as_cat; concatenate columns in
`field_spec` order into the fixed `S`. The one strategy we need; alternatives
(e.g. variable-length with separators) are not promoted (rule 10).

## Known property to carry forward

**Leading-zero dominance (B.10).** Fixed-width padding makes high slots almost
always `0`. That is useful magnitude signal (magnitude ≈ count of leading zeros)
but it is near-constant, so under masking those positions are trivial to predict
— a fact [masking.md](masking.md) must account for (it argues for field-level,
not digit-level, masking).

## Deferred

- A real-vs-pad stream on disk. `fixed_width_padded` has no structural pad
  (leading zeros are real digits), so every position is real; the stream is
  implied rather than stored.

Links: [[claude-breakdown-notes B.1]], [[claude-breakdown-notes B.4]],
[[claude-breakdown-notes B.7]]. Consumes [field_spec.md](field_spec.md) +
[vocabulary.md](vocabulary.md); read by [embedding.md](embedding.md) and
[masking.md](masking.md).
