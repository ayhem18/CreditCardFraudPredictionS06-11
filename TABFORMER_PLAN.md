# Tabular BERT for S6E11 — pretrain (masked features) + fine-tune plan

Status: design agreed 2026-11-20; code home not yet chosen (see open
question at the bottom). This doc is self-contained so the work can be
picked up from any folder/session.

## 1. Source write-up (Chris Deotte, "Q-FT", S6E10 discussion 750009)

Each row → one vector per feature ("token") → transformer → CLS logit.

- **Tokenizer**: per column, embedding = column embedding + value-ID
  embedding (learned lookup over the column's codes) + periodic numeric
  embedding (standardized x through sin/cos at 32 learned frequencies →
  MLP 64→1024), then RMSNorm × gain to token scale. Vocabularies from
  train + test features only (label-free). High-cardinality numerics get
  BOTH an ID and a number (Flight Distance as "route id": every distance
  seen ≥3× gets its own code).
- **Backbone**: Qwen3-0.6B-Base via `inputs_embeds`, causal attention,
  learned CLS at last position, Linear→1 head. LoRA r16 α32 on all 7
  projections (10.1M trainable). Feature order re-shuffled every row,
  every step (fixed order loses 0.0069 AUC). Prediction = mean logit over
  5 orders.
- **Results (S6E10, ROC AUC)**: OOF 0.959795 ≈ tuned XGBoost 0.959852 >
  RealMLP 0.959311; TabPFN-3.5 0.961248 (best). **Random-init backbone
  0.960436 vs pretrained Qwen 0.960444 — language pretraining adds
  nothing.** Frozen backbone (tokenizer+head only) 0.959788 — nearly as
  good. Their stated next step: *tabular* pretraining + in-context
  learning, TFM-style (Part 2: discussion 751334).

So: our experiment = their untested next step, on our data, on our frozen
CV. Honest bar: on S6E10 the whole rig tied XGBoost. On S6E11 the bar is
`xgbU_bagged_v2` OOF 0.94671. Realistic payoff is a **decorrelated
ensemble member** (hgb-for-xgb effect), not a solo win.

## 2. Adaptation to S6E11 (22 features; vocab from train+test)

| column(s) | value-ID path | numeric path |
|---|---|---|
| merchant_category (12), card_type (4), auth_method (3), channel (3), device_type (8), is_new_merchant (2) | category code | — |
| 5 alarm flags (bool) | code (0/1) | — |
| cvv_retry_count (0-4), txn_count_last_24h, prior_disputes, time_of_day_hour (0-23), day_of_week (0-6) | one code per level | standardized value |
| amount_usd, hours_since_last_txn, distance_from_home_km, card_age_months, customer_age, account_balance_usd, velocity_score, merchant_risk_score | quantile-bin ID (256 bins, train+test edges; "rare" fallback) | standardized value, periodic sin/cos @ 32 learned freqs → MLP |

**Bidirectional encoder (BERT-style), not causal** — this also tests the
write-up's open question (remove causal mask ⇒ no order shuffling
needed). CLS token; column embeddings learned from scratch (their
ablation: name-initialization and names don't matter).

## 3. Pretraining task (the "BERT style")

- Corpus: train + test features = 999,944 rows, no labels touched.
- Mask ~20% of the 22 tokens per row (per-column learned [MASK]
  embedding; column embedding retained).
- Targets: masked token reconstruction — CE over the column's vocab for
  the ID path (for binned numerics, CE over bins ≈ soft distribution
  learning); smooth-L1 on the standardized value for the numeric path.
- Budget: 3–10 epochs (local pilot 3, server more), batch 512–2048,
  AdamW, cosine or linear decay, bf16 autocast, grad clip 1.0,
  `dl_toolbox` seed_everything + logger contract (mlflow/tensorboard).

## 4. Fine-tune protocol (competition integration)

- Frozen `alarm_profile_stratified` 5-fold (same folds as every `*U*`
  exp). Per fold: train on the 4 outer folds' rows, predict the held-out
  fold → OOF table in `training_io` format (`oof.csv`: id, fold_id,
  is_fraud, y_prob) + `config.json` with `cv.scheme` recorded.
- Then everything existing works unchanged: `eval_oof.py`,
  `compare_regimes.py` (direct tabular-transformer vs XGB line),
  `build_bag.py` (as an 11th member; needs `submission.csv` = mean of the
  5 fold models on test).
- Head ablation, fold-0 pilot first (their protocol): pretrained-init vs
  random-init vs frozen-encoder+head. Decision gate: proceed to 5-fold
  only if pretrained ≥ random-init or AUC ≥ ~0.943.

## 5. Hardware plan

- **Local (RTX 3070 laptop, 8 GB, 16 cores, 25 GB RAM)**: pilot tier —
  from-scratch encoder ~8 layers × width 256 (~15M params), batch 512.
  Rough estimate 1–2 h per pretrain + ~30–60 min per fine-tune fold.
- **Server (multi-GPU)**: scale tier — width 512–768, 12–16 layers, bigger
  mask budget/epochs; optionally 5 folds in parallel on 5 GPUs (their
  layout). Needed: GPU models, CUDA/driver, python env, and how this repo
  or the data gets there (git clone + copy `storage/raw_data`).

## 6. Code home — OPEN QUESTION

Torch is not in the comp venv (and shouldn't be). Options:
(a) new sibling project under `DL_projects` built on `dl_toolbox`
(editable-path dependency per its README) — fits the toolbox's
"promote-after-use" philosophy; (b) `experiments/tabformer/` inside this
repo with a separate torch venv — tightest competition integration;
(c) open a fresh session in the chosen folder and implement there, using
this doc as the spec. Data export (codes/vocab, see §2) can be produced
by the existing pandas venv either way.
