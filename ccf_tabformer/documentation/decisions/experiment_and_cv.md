# experiment & cv — the orchestration that composes everything

*Orchestration layer. Composes every component's `{strategy, args}` and runs the
two phases under CV discipline. Breakdown: Part A2 (decided). See
[workflow.md](workflow.md).*

## In plain terms

This is the top-level config — the one the experiment-visibility premise is about:
**every field required, defaults avoided**. Reading it tells you the entire
encoding and training setup without opening code. It names the three frozen-level
choices (by recipe/hash) and the four runtime-module strategies, plus the training
args, and it drives the two runs.

## Status: reconstruction implemented (full run training); fine-tuning now active

Run 1 (unsupervised reconstruction) is implemented and its full run is training.
Run 2 (fine-tuning + OOF) is now the active work — its plan is
[coding_plan_finetune.md](coding_plan_finetune.md). The two decisions below
governed the reconstruction run and still hold for it:

- **TrainVal, not TrainOnly.** The engine must report a validation metric so we
  can see whether the model *generalizes* its reconstruction, not just fits the
  train rows. TrainOnly (the vibe-coded choice) cannot answer that.
- **val = the test split.** Pretraining is label-free, so spending the unlabeled
  test rows on validation costs no label signal, and it measures reconstruction
  on the very rows we ultimately predict. The **val mask is fixed/seeded** (train
  masking stays fresh per step) so the val curve is comparable across epochs.

> **Deferred tension (Part A2).** Holding test out of *training* means the
> eventual submission encoder never adapts to test features — the transductive
> advantage. For development and the generalization read this is right; for the
> *final* submission model, fold test back into training once hyperparameters are
> fixed. Logged, not resolved now.

The **overfit wrapper** (find hyperparameters by overfitting a small subset) is a
later addition; it complements the "descending loss on a tiny subset" milestone.

## The two runs (Part A2)

### 1. Unsupervised reconstruction — pretraining, once, shared

One run over **all corrected rows** (train + test *features*; label-free), through
`embedding → masking(field_level) → encoder → reconstruction heads`. Produces a
single pretrained `embedding + encoder`. Because it is label-free, including every
row (test and all folds) leaks nothing and **matches the transductive test
condition** — so one shared pretraining is both correct and cheaper than
per-fold.

### 2. Finetuning — 5×, honest OOF

For each held-out fold: load the shared pretrained weights, attach the fraud head,
fine-tune on the other 4 folds, predict the held-out fold. Concatenating the 5
held-out predictions gives the **OOF**.

**The CV invariant (Part A2):** no held-out *label* may influence its own OOF
prediction. Features are not labels. This is satisfied automatically here because
pretraining is label-free and each fold's finetuning never sees its own fold's
labels. (The supervised-reconstruction path would violate this unless pretraining
were redone per fold — the reason it is not chosen.)

## Use the stack's fold definition (do not invent new folds)

The whole point of OOF is to compare — transformer variants against each other
**and against the TE / GLM / XGB stack at LB 0.95062**. That comparison is only
valid if the transformer's 5 folds are the **same partition** the existing stack
used. So the fold assignment is a frozen artifact **imported from the parent
`CreditCardFraud` project**, not a fresh `KFold` here. Same row→fold map →
comparable OOF AUC.

The concrete artifact is
`ccf_ml/src/ccf_ml/package_artifacts/cv_splits/alarm_profile_stratified_cv.json`
(`n_splits=5`, `id_column="id"`, `target="is_fraud"`, `folds[k].val_ids`). A new
`folds` data level imports it and aligns it to `inner_table.row_ids()` — see
[coding_plan_finetune.md](coding_plan_finetune.md).

## The strategy axis (oracle): the experiment recipes

The existing `run_recipes/` already encode the key axes as named configs:

- `supervised_reconstruction/pretrain/` — the shared reconstruction run
  (smoke_local, server).
- `classification_fine_tune/full_run/` — `pilot_fold0_random` (no pretraining,
  control), `pilot_fold0_frozen` (pretrained encoder frozen, train head only),
  `pilot_fold0_pretrained` (pretrained, full fine-tune), `fivefold` (the real OOF).

These three finetune variants are exactly the ablation that answers **"did
pretraining help?"**: random < frozen ≤ pretrained if the reconstruction learned
anything transferable. Keep them as first-class recipes.

## Priority order (Part C, reframed goal)

1. **A minimal run whose loss descends** on a tiny subset — the milestone that
   says the contract is sound. Debug the vibe-coded loss-up first; the suspects
   now visible in the code (see [coding_plan.md](coding_plan.md)) are the
   half-refactored `network.py` (a `NameError` in the masked branch) and the
   `mfm` masker's **digit-level** draw, which contradicts the field-level
   decision ([masking.md](masking.md)) and lets the model cheat — not a row
   position embedding, which the encoder already avoids.
2. **Honest OOF wiring** on the imported folds, so every later choice is measured.
3. Only then A/B the component strategies.

## Deferred

- Optimizer, LR schedule, epochs, batch size, warmup (training args — explicit in
  the recipe, tuned after step 1).
- Frozen vs full fine-tune as the default for `fivefold` (decide from the pilot
  ablation).
- Whether early-stopping uses the held-out fold (careful: that couples model
  selection to the OOF; prefer an inner split or fixed epochs).

Links: [[claude-breakdown-notes]] Part A2 and Part C. Composes all of
[field_spec.md](field_spec.md), [vocabulary.md](vocabulary.md),
[inner_table.md](inner_table.md), [embedding.md](embedding.md),
[masking.md](masking.md), [encoder_and_heads.md](encoder_and_heads.md).
