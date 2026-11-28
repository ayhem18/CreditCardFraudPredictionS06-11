# Coding plan — a new implementation that reflects the contract

*A fresh implementation of the unsupervised-reconstruction pipeline, reflecting
the decisions and contracts in this folder and mirroring the contract-driven
experimentation architecture of `digital_lab_engine` (`scripts/` + `task_engine`
+ data levels + recipes + the overfit wrapper). See [workflow.md](workflow.md).*

## Stance toward the current `ccf_tabformer_dl/` code

**Ignore and rewrite, do not refactor.** The vibe-coded implementation is not
written to reflect a mental model of the problem; building on it would carry its
unclear interactions forward. The new code is organized by the contracts
(`TableSpec`, `Embedder`, `Masker`, …) and by the svarka experimentation pattern.

What the old code is still useful for — as *reference*, not foundation:
- **Keep the ideas that were right:** additive `value + column + slot` embedding;
  an encoder with **no row-position embedding** (B.7); a reconstruction head that
  masks disallowed logits to `-inf` via a per-position support set.
- **Drop the anti-patterns:** the opaque numpy export as the data interface; the
  mixed digit+feature masker (contradicts field-level, B.2); the mask swap *inside*
  the encoder; using one TrainOnly engine for everything.

## The experimentation architecture we are copying

From `digital_lab_engine/scripts` + `task_engine`:
- **Thin CLIs**, one per mode: parse a recipe name → `resolve_<mode>_recipe` →
  `make_<mode>_engine(script, recipe, tracking_backend, force).train()`. No logic
  in the CLI.
- **`task_engines/`** holds the engines, each wrapping a `dl_toolbox.dl_engine`
  engine: overfit sanity-check (TrainOnly over a subset), full-run
  (`DefaultLayoutTrainValEngine`), optional lr-finder (`DefaultLayoutFindLrEngine`).
- **Data levels** frozen by thin `define_*_collection.py` scripts
  (`preparation.process(generating_process_name=...)`, `--force` →
  `cleanup_named_freeze`).
- **The overfit wrapper**: read a grid-search document → write one recipe per grid
  member → run each on a free GPU (`--min-free-gib`). This is the pruning tool.
- **Recipes** in `run_recipes/<mode>/`; MLflow logger; `human_readable_name` as the
  run/`generating_process_name`; `--force` overwrites that name only.

## The two tools, and why components come first

- **Overfit wrapper** — prune the million choices by seeing which combinations can
  even overfit a tiny subset (TrainOnly).
- **Full run** — commit the promising combinations (TrainVal, val = test).

Both need the experiment **components** built first. With the problem broken down
and the contracts nearly complete, that is now the tractable next step.

## Phases

### Phase 1 — implement the unsupervised-reconstruction components
Fill the contracts, all oracle-built from a `TableSpec`:
- `Embedder.forward` (`additive`); `Masker.mask` (`field_level` default;
  mixed draw as a *named ablation* strategy, not default).
- **`Encoder`** (pre-norm transformer, **no CLS**, no row-position embedding) and
  a **reconstruction `Head`** (`shared_support_masked`: one linear + support
  additive). Per-field-type heads stay a named ablation.
- A **model** object that composes `Embedder → Encoder → Head`. Masking is
  applied by the training step *before* `forward`.
- **loss + recon metric** over `target_mask` positions only.

### Phase 2 — freeze the table, then a row-index subset
Tokenization is the **full** CSV → numpy freeze (`field_spec` → `vocabulary` →
`inner_table`). A subset is **only row indices** into that array (stratified,
enough fraud rows that reconstruction is non-trivial). Named config
`alarm_label_1k_seed0`: 1k train rows from each of the four
`(is_alarm, is_fraud)` cells; `is_alarm` is the OR of the five EDA Cut-1
flags. Frozen by `scripts/data/bash_scripts/run_data_contract.sh` (or the
thin `scripts/data/define_*_collection.py` scripts). The overfit dataset is
`(tokenized_table, row_ids)`.

### Phase 3 — wire components + subset level to TrainOnly + the overfit wrapper
- A `task_engine` **overfit** engine: fit the subset level (as train, and as val
  = the same subset, per the svarka pattern) with the composed model; TrainOnly in
  spirit — no generalization claim.
- Thin CLI (`overfit_train.py`) + an overfit recipe; then the **grid-search
  wrapper** over the real choice space (embedding composition, mask rate, `d_model`
  / depth, lr, …) on free GPUs.
- **Milestone:** the model *overfits the subset* (recon loss → ~0, recon_acc →
  ~1) for at least some combinations. Combinations that cannot overfit are pruned.

### Phase 4 — full training (TrainVal)
- A `task_engine` **full-run** engine on `DefaultLayoutTrainValEngine`, over the
  full data level, **val = the test split**, **seeded val mask** (watch train recon
  down + val recon tracking, not diverging). Commit the combinations that survived
  Phase 3. (lr-finder optionally between 3 and 4.)

## Status of the contracts (what Phase 1 builds on)

- Done: `TableSpec` / `FrozenTableSpec`, additive embedder, field-level masker,
  pre-norm encoder, shared support-masked head, `MfmModel`, reconstruction
  `FitSteps`.
- Done: freeze chain `field_spec` → `vocabulary` → `inner_table` (full numpy) →
  `row_subset` (row indices), plus `InnerTableIndexDataset`.
- Phase 3: TrainOnly overfit engine over `alarm_label_1k_seed0`
  (`task_engines/overfit`, `scripts/unsupervised_reconstruction/overfit_train.py`, recipe
  `run_recipes/overfit_sanity_checks/mfm_overfit_alarm_label_1k.json`). Grid
  wrapper still later. Phase 4: TrainVal, val = test.

## Decisions — status

1. **Masking:** field-level default; mixed draw as a named ablation.
2. **Scope:** unsupervised reconstruction only; finetune / OOF out of scope.
3. **Engines:** TrainOnly for the overfit sanity check (Phase 3); TrainVal for the
   full run with val = test (Phase 4).
4. **Deferred (Part A2):** fold test back into *training* for the final submission
   model once hyperparameters are set (test is the val split during development).

Links: [workflow.md](workflow.md), [embedding.md](embedding.md),
[masking.md](masking.md), [encoder_and_heads.md](encoder_and_heads.md),
[experiment_and_cv.md](experiment_and_cv.md).
