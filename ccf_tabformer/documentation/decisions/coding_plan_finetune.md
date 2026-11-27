# Coding plan — fine-tuning (fraud classification from the pretrained backbone)

*Run 2 of [experiment_and_cv.md](experiment_and_cv.md), now active: fine-tune the
pretrained MFM backbone into a fraud classifier under honest 5-fold OOF. Grounded
in the current `ccf_tabformer_dl/` tree. See [workflow.md](workflow.md).*

## Stance toward the existing `task_engines/finetune/`

**Replace it.** The current `task_engines/finetune/` (and `tabformer_components/
model/{classifier,network,vocab}.py`, `data_contract/{tokens,dataset}.py`) are
from the vibe-coded era: they read the **dead opaque numpy export**
(`load_tokenized_table`, `table.fold_ids`) and a separate `FineTuneModel`, none of
which touch the frozen `inner_table` levels or the oracle-built components the full
run uses. Fine-tuning is rebuilt on the new stack; the old files are retired
(cleanup, not deleted without your ok).

## What fine-tuning reuses, and the one new idea

- **Reuses by name:** the same `embedder` + `encoder` contracts as `MfmModel`, so
  the pretrained full-run checkpoint's backbone transfers by key. The new
  `ClassifierModel` names its submodules `embedder` / `encoder` identically; the
  reconstruction head is dropped.
- **One new component:** a **pooling `ClassifierHead`** (the encoder has no CLS),
  `(B, S, d) → (B,)` fraud logit. Default `mean_pool`; `cls_token` /
  `attention_pool` are named ablations.
- **No masking** at fine-tuning: `ids` are the raw row ids.

## The fold artifact (Part A2 — comparability is the whole point)

Folds are **imported from the stack**, not re-invented:
`ccf_ml/src/ccf_ml/package_artifacts/cv_splits/alarm_profile_stratified_cv.json`
— `n_splits=5`, `id_column="id"`, `target="is_fraud"`, 699,960 rows,
`folds[k] = {fold_id, n_train, n_val, val_ids}` where `val_ids` are original `id`
values. Reusing this exact partition makes the transformer OOF AUC directly
comparable to the stack's LB 0.95062.

**Decided:** `alarm_profile_stratified_cv` is the split the 0.95062 stack's OOF was
computed on (confirmed) — the `folds` level imports that file, not
`flag_stratified_cv` / `profile_stratified_cv`.

## Contracts set up now (stubs, compute deferred)

- `tabformer_components/classifier_head/` — `ClassifierHead` ABC + `mean_pool` +
  oracle (`build_classifier_head(strategy, args)`).
- `tabformer_components/classifier_model.py` — `ClassifierModel`
  (embedder→encoder→classifier_head; `set_encoder_frozen`), `build_classifier_model`.
- `tabformer_components/backbone.py` — `load_backbone_from_checkpoint` (filter
  `embedder.*`/`encoder.*` from a full-run checkpoint into the ClassifierModel;
  the LightningModule key prefix is the one detail to confirm from a real
  checkpoint).

## Phases

### Phase 1 — fill the component contracts
`ClassifierHead.forward` (mean over S → MLP → squeeze), `ClassifierModel` already
wired, `load_backbone_from_checkpoint` (confirm the checkpoint key prefix, assert
the backbone matched fully and only `classifier_head.*` is missing).

### Phase 2 — a `folds` data level
A frozen data-contract level mirroring `row_subset`:
- process + oracle, strategy `imported`; args name the source CV file and scheme;
- reads `alarm_profile_stratified_cv.json`, builds `id → fold_id` from the five
  `val_ids` lists, aligns to `inner_table.row_ids()` (train rows), writes a
  `fold_id (n_rows,)` sidecar with a sentinel for test rows;
- identity = `inner_table` hash + a digest of the source CV document;
- thin `scripts/data/define_folds_collection.py`, named config e.g.
  `alarm_profile_5fold_v1`.
Verify: `n_val` per fold and total train count match the JSON, and every train
`row_id` gets exactly one fold.

### Phase 3 — labeled dataset + the fine-tune task engine (rewrite)
- A labeled dataset yielding `{ids: (S,) long, label}` by indexing
  `inner_table.token_ids()` + `.labels()` at a row index (extend
  `InnerTableIndexDataset` or a sibling).
- `task_engines/finetune/` rebuilt: per fold `k`, train index = train rows with
  `fold_id != k`, val index = train rows with `fold_id == k`; build
  `ClassifierModel` from the `FrozenTableSpec` via the oracles; if
  `init == "pretrained"`, `load_backbone_from_checkpoint`; `set_encoder_frozen`
  per the regime; BCE loss; the **ROC-AUC metric is reusable as-is** from the old
  `finetune/steps.py` (concat per-item `[score, label]` → `roc_auc_score`), rewired
  to the new `ids`/label dataset and `ClassifierModel`. `DefaultLayoutTrainValEngine`,
  monitor `val auc`.
- Regimes as recipes (the "did pretraining help?" ablation, experiment_and_cv.md):
  `random` (no backbone load), `frozen` (backbone frozen, head only), `pretrained`
  (backbone loaded, full fine-tune).

### Phase 4 — OOF assembly + submission
A finetune **wrapper** (like the overfit wrapper): run the 5 fold recipes, collect
each fold's **held-out** predictions → OOF `(n_train,)` keyed by `row_id`; report
**OOF AUC** (compare to 0.95062). For the submission, predict the **test split**
with each fold model and **average** the 5 (simple, leakage-free); write
`row_id, prediction`.

## Decisions — status

1. **Folds:** reuse `alarm_profile_stratified_cv` (confirmed); a new `folds`
   level, not the dead export.
2. **Pooling:** `mean_pool` default; `cls_token` / `attention_pool` ablations.
3. **Regimes:** `random` / `frozen` / `pretrained` recipes.
4. **Early stopping must not watch the held-out fold** (it would couple model
   selection to the OOF). Use fixed epochs, or an inner split carved from the 4
   training folds. Decide in Phase 3.
5. **Submission test prediction:** average the 5 fold models (default) vs refit on
   all train rows. Default: average.
6. **Deferred (Part A2):** the pretrained backbone used here was trained with
   test in the val split, not in training. For a final submission model, a backbone
   pretrained with test folded into training is the transductive-correct choice.

Links: [experiment_and_cv.md](experiment_and_cv.md),
[encoder_and_heads.md](encoder_and_heads.md), [coding_plan.md](coding_plan.md)
(the reconstruction pipeline this builds on).
