# masking — what the model must reconstruct

*Runtime module (collate-time; random per epoch, so not a frozen level).
Breakdown axes: B.2 (decided: field-level), B.8. See [workflow.md](workflow.md).*

## In plain terms

For the **unsupervised reconstruction** (pretraining) phase, `masking` hides part
of each row and asks the model to rebuild it. The choice of *what unit* to hide is
the single most consequential decision in the whole approach (B.2), and it is
already decided: **field-level**.

Reads: the `inner_table` streams for a row. Produces: the masked `token_ids` (some
positions replaced by `[MASK]`) **plus** the reconstruction targets (the original
ids at the masked positions). At finetuning, masking is **off**.

## Why field-level (decided, B.2)

Masking a *whole field* at once (all digit positions of `amount_usd`, or the one
position of a categorical) forces the model to predict that field from the
**other fields** — i.e. to learn `P(field | other fields)`, which is the
generator's conditional structure we are after. Masking *individual digits* would
let the model rebuild a masked cents digit from the dollars digit of the *same
value* — intra-number correlation, mostly the trivial structure of
[leading-zero padding](inner_table.md) (B.10). Field-level closes that shortcut.

Masked positions **keep their feature + slot embeddings**; only the token id
becomes `[MASK]`. That is what lets a single generic `[MASK]` (from
[vocabulary.md](vocabulary.md)) suffice — the model still knows *which* field and
*which* slot it must predict.

## The label never enters masking (B.8, settled by the chosen path)

Because the path is unsupervised reconstruction with a **features-only**
`inner_table`, `is_fraud` is not a position in the sequence, so masking cannot
touch it. There is no "mask the label" case here; the label is the finetuning
head's target only. (The supervised-reconstruction alternative would make the
label a maskable field and force per-fold retraining — Part A2 — which is why we
are not taking it.)

## The strategy axis (oracle): the masking scheme

### `field_level_random` (the strategy to start with)

Per row, select a random subset of fields (by a per-field probability), mask all
positions of each selected field, and record their original ids as targets.

*Args (required, explicit):* the mask probability (or target fraction of fields),
and the per-field-type selection policy.

## As a composed object (the `Masker`)

Like the `Embedder`, the masker is a class built by its oracle from the shared
`vocabulary`+`inner_table` spec: `build_masker(strategy, args, *, spec)`. It takes
`(B, S)` ids and returns **masked ids** (the chosen positions overwritten with the
reserved `[MASK]` vocab id) plus a **boolean target mask**. Field-level grouping
uses `feature_index` from the spec. Because masking writes a vocab id rather than
reaching into the embedding tables, the `Masker` and `Embedder` never reference
each other — they only share the spec. The reconstruction loss reads the original
ids at the `True` positions of the target mask.

Train masking is fresh per step; the **val mask is fixed/seeded** so the val
reconstruction curve is comparable epoch-to-epoch (see
[experiment_and_cv.md](experiment_and_cv.md)).

## Open sub-decisions (params, not new strategies)

- **Mask rate.** What fraction of fields per row. Start near BERT's 15% of
  *fields*, tune.
- **Numeric vs categorical balance.** A numeric field spans many positions, a
  categorical one; decide whether selection is per-field (equal field weight) or
  per-position (favors numerics). Lean: per-field.
- **BERT 80/10/10 corruption** (replace-with-mask / random / keep). Decide whether
  to adopt; it mainly matters for finetuning–pretraining input match.

Links: [[claude-breakdown-notes B.2]], [[claude-breakdown-notes B.8]]. Reads
[inner_table.md](inner_table.md) + [vocabulary.md](vocabulary.md); targets are
consumed by the reconstruction head in
[encoder_and_heads.md](encoder_and_heads.md).
