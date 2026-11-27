# encoder & heads — the backbone and what the loss reads

*Runtime modules, grouped (Q3). The backbone is shared across phases; the heads
differ per phase. Breakdown axes: B.7 (encoder), B.2/B.8 (heads). See
[workflow.md](workflow.md).*

## Encoder — in plain terms

A transformer encoder that reads the `(B, S, D)` tensor from
[embedding.md](embedding.md) and returns a contextualized `(B, S, D)`. This is
where attention reassembles the multi-digit resolutions that `additive` embedding
left as separate positions (B.5) and where cross-field interactions are learned —
the one thing this approach can buy over the GLM (Part C of the breakdown notes).

**Design invariant (B.7): no row-level position embedding.** Column identity and
digit order already live in the `feature_ids` / `slots` streams. A standard
sequence-position embedding would inject a false order over columns; attention is
over the *set* of positions, order coming only from the structural streams. (The
current `model/network.py` already respects this — `value + column + slot` plus a
single learned CLS, no sequence-position table — so the loss-up cause is
elsewhere; see [coding_plan.md](coding_plan.md).)

*Config (required, explicit):* depth, width `D`, heads, FFN size, dropout, norm
placement (pre-norm). One backbone for now; an oracle is promoted only if a second
backbone earns it (rule 10).

## Heads — in plain terms

The heads are the only part that differs between the two runs.

### Pretraining: reconstruction heads (masked field modeling)

At each **masked** position, predict the original token id. Because token ids live
in different spaces per field type, the head is **per field type**, not one
softmax over the whole vocabulary:

- a **digit head** (10 classes) for numeric / in_between_as_num slots,
- a **per-categorical head** over each categorical column's own value set,
- a **boolean head**.

Loss: cross-entropy over masked positions only. This is the `mfm` step in the
existing tree.

*Open:* weight-tying the digit/categorical heads to the `embedding` token table
(fewer params, common in MLM); per-column categorical heads vs one shared.

### Finetuning: the fraud head

Pool the encoder output to one vector, then a linear layer to a single logit.
Loss: BCE; metric: **ROC-AUC** (the competition metric, rank-based, so no
calibration needed).

*Strategy axis (the one real choice here): pooling.* `cls_token` (a prepended
learned position) vs `mean_over_real_positions` vs `attention_pool`. Lean: start
with mean-pool over real (non-pad) positions; it needs no vocabulary change.

## Deferred

- Pooling strategy (above) — worth a small ablation.
- Head weight-tying.
- Whether the encoder is **frozen** or **fine-tuned** during the fraud phase —
  this is really an *experiment* axis and lives in
  [experiment_and_cv.md](experiment_and_cv.md) (the existing `run_recipes` already
  distinguish `frozen` / `pretrained` / `random`).

Links: [[claude-breakdown-notes B.7]], Part C. Reads [embedding.md](embedding.md);
reconstruction targets come from [masking.md](masking.md); driven by
[experiment_and_cv.md](experiment_and_cv.md).
