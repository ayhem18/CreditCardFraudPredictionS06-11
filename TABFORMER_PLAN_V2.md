# Tabular BERT for S6E11 — revised plan (v2)

Status: proposal for review, 2026-11-24. Supersedes `TABFORMER_PLAN.md` (v1)
as the working design; v1 is kept for the Chris-Deotte source notes and the
S6E10 evidence. **No code is written until this is approved.**

Defaults below are the agreed recommendations; alternatives we deliberately
keep open are marked **[option]** so we can flip them without a redesign.

---

## 0. What changed from v1, and why

1. **Server-trained, not local.** Target hardware is the server: **2 GPUs ×
   60 GB**. The architecture is sized by the problem, not by the laptop. All
   v1 "8 GB / 15M param / batch 512" ceilings are dropped. Multi-GPU (DDP)
   comes from the `dl_toolbox` engine, not hand-rolled.
2. **Digit-sequence tokenizer replaces quantile bins.** v1's 256-bin value-ID
   path throws away the one thing this dataset rewards — the *exact digits*
   (the generator's "handwriting"; see `JOURNEY.md` §4 and
   `experiments/target_encoding/README.md` §5). We instead write each number
   as a fixed-width digit string and let the transformer attend over the
   digits. This is the general form of the digit-key TE features that already
   work, and of the top solutions' "GPT-2 token keys."
3. **Built on `dl_toolbox`.** The training loop, DDP/occupancy, logging
   (mlflow/tensorboard), checkpointing, LR finder, warmup-cosine, AMP, and
   representation-eval (linear probe) all come from the toolbox's
   `dl_engine` — mirroring the existing FCMAE masked-pretraining task engine.
   We do not duplicate any of it.

---

## 1. Core idea — numbers as fixed-width digit sequences

### 1.1 The data supports it cleanly (verified on the raw CSV)

Int/decimal widths per continuous column (from the raw strings, both splits):

| column | int width | dec width | digit slots |
|---|---|---|---|
| amount_usd | 4 | 2 | 6 |
| account_balance_usd | 5 | 2 | 7 |
| hours_since_last_txn | 2 | 2 | 4 |
| distance_from_home_km | 2 | 2 | 4 |
| velocity_score | 2 | 1 | 3 |
| merchant_risk_score | 3 | 1 | 4 |

**Zero-padding to each column's max decimals is lossless.** The CSV drops
trailing zeros (`6.3`, `3233.0`), but a dropped trailing zero is exactly a
zero digit, and no value in either split has more than 2 decimals. So
`61.16 → 0061.16` and `3233.0 → 03233.00` recover the canonical value with no
information lost. (The "number of decimals written" is itself recoverable as
"is the last digit 0", so we lose no fingerprint by padding.)

### 1.2 Token layout per row (47 tokens)

```
[CLS]
  + 28 digit tokens   (the 6 continuous columns above, one token per slot)
  +  7 small-int tokens (txn_count_24h, cvv_retry, hour, day_of_week,
                         card_age_months, customer_age, prior_disputes) — one
                         whole-value token each (ranges are tiny: ≤64)
  +  5 categorical tokens (merchant_category, card_type, auth_method,
                           channel, device_type)
  +  6 flag tokens    (the 6 booleans)
= 47 tokens, identical length for every row.
```

Bidirectional (no causal mask) → **no feature-order shuffling needed** (this
is v1's open question, resolved by using an encoder, not a causal LM).

### 1.3 Token embedding

Each token embedding is a **sum of three learned parts**, so a bare digit
carries its context:

```
emb(token) = value_emb  +  column_emb[c]  +  slot_emb[c, slot]
```

- `value_emb`: for a digit slot, a lookup over `{0..9}` (shared across all
  digit slots); for a categorical/flag/small-int token, a lookup over that
  column's own value vocabulary.
- `column_emb[c]`: which of the 24 columns this token belongs to.
- `slot_emb[c, slot]`: which digit position within that number (so "tens
  digit of amount" ≠ "units digit of balance"). Trivial (one slot) for
  non-digit tokens.

A masked token replaces only `value_emb` with a learned per-column `[MASK]`
embedding; `column_emb` and `slot_emb` are retained (same as v1/FCMAE).

**[option] magnitude channel** (default OFF): additionally feed each
continuous column's standardized value through a periodic sin/cos → MLP and
add it to that column's tokens, so the model has both a digit view and a
smooth-magnitude view. Turn on only if the pilot underperforms. Default off
keeps the test of "are digits enough?" clean.

### 1.4 Prediction restricted to valid indices per column (your idea)

Reconstruction uses a **per-column (and per-slot for digits) softmax**:

- digit slot → softmax over `{0..9}`, with a **hard support mask**
  (default): logits for digits that never occur at that `(column, slot)` in
  train+test are set to `-inf`. This *is* "force the model to predict only
  the correct indices," applied at the finest grain and exactly.
- categorical/flag/small-int token → softmax over that column's value
  vocabulary.

Why the hard mask instead of your two-loss design: pretraining runs on
train+test = **every row that exists**, so the observed support per slot is
the *true, complete* support. A learned "viable-set" loss can only
rediscover what the mask already states exactly, and (by your own prediction)
saturates in the first steps — so it is redundant with the mask.

**[option] viability loss** (default OFF): if you want it as a learning
exercise or a soft prior (e.g. to study the "flatten-then-sharpen"
dynamics), add a second head with a multi-label "which tokens are possible
here" BCE, alongside the reconstruction CE, and do *not* hard-mask. Kept as a
toggle, not the default.

---

## 2. Model (server-sized, from scratch)

- Bidirectional Transformer encoder, **from scratch** (language pretraining
  added nothing on S6E10 — v1 §1 — and we have no pretrained tabular LM).
- `d_model = 512` (sweepable to 768), **12 layers** (sweepable to 16),
  8–12 heads, FFN 4×, pre-norm (RMSNorm), dropout 0.1, GELU. Learned `[CLS]`.
- Seq len 47 → attention is cheap; width/depth and batch size are the only
  real knobs, and 2×60 GB removes the ceiling. Concrete sizes get fixed by a
  short LR-finder + a width/depth sweep on the pretrain objective.

---

## 3. Pretraining — masked feature modeling (`train_only` engine)

- **Corpus:** train + test features = 999,944 rows, **no labels touched** →
  transductive use of test features is legitimate (same discipline as
  full-train target-encoding of the test set).
- **Mask ~20% of tokens**, a mix of:
  - *whole-feature masks* (all slots of one feature) → learn cross-feature
    structure;
  - *single-slot masks* (one digit) → learn intra-number digit structure
    (your stated goal — "relation between the digits themselves").
- **Loss:** masked cross-entropy, per-column/per-slot, over the masked
  positions only (+ smooth-L1 on the numeric channel **[option]**, only if
  the magnitude channel is on).
- **Metrics:** per-slot reconstruction accuracy and perplexity on a held-out
  slice of masked positions; logged each epoch.
- **Output:** one encoder checkpoint, consumed by every fine-tune run.

---

## 4. Fine-tuning — fraud classification (`train_val` engine, 5 folds)

- **Folds:** the frozen `alarm_profile` 5-fold from
  `storage/artifacts/training/icU_base/oof.csv` (row-aligned with every
  existing blend member).
- **Head (what you asked for):** final `[CLS]` hidden state → a 2–3 layer MLP
  → 1 logit, BCE loss, **val ROC-AUC** as the monitored metric, checkpoint
  the best epoch.
- **One engine run per fold** (5 names in one run-group). Each run trains on
  the 4 training folds, predicts the held-out fold → that fold's OOF rows.
  Test prediction = mean of the 5 fold models' test scores.
- **Regimes compared at fold-0 (the gate):**
  1. pretrained-init encoder, fine-tune end-to-end;
  2. random-init encoder (does pretraining help at all?);
  3. frozen encoder + head only (cheap; also the linear-probe baseline).
- **Quick pretrain-quality gate:** `dl_toolbox` representation-eval
  (`linear_probe`) — freeze the encoder, fit a logistic head on `[CLS]`
  features, read val AUC. Fast signal on whether the representation carries
  fraud signal before spending on full fine-tunes.

---

## 5. How it wires into `dl_toolbox` (concrete)

- **New project `ccf_tabformer`** scaffolded with the toolbox
  (`setup-project`), living under `kaggle_comps/CreditCardFraud/ccf_tabformer/`
  with its own project tree (`dl_toolbox_project.json`, `storage/`,
  `scripts/`, `experiments/`, `run_recipes/`) and its **own venv**
  (`torch`, `lightning`, `dl-toolbox` as an editable path dep, `numpy`,
  `pandas`). This keeps the comp venv clean and makes the project portable to
  the server. Mirrors the existing consumer projects
  (`Solar_Filament_Seg.../solar_fil_src`, `ICM_powder_segmenter/icm_dl`).
- **Task code built project-local first** (the toolbox's "promote-after-use"
  rule): a `tabular_mfm` package providing the pieces FCMAE provides for
  images — `model/` (tokenizer + encoder + reconstruction heads + CLS head),
  `mask/` (token-mask sampler), `loss/` (masked CE + optional smooth-L1),
  `metrics/` (reconstruction acc/perplexity, AUC). The **project owns** the
  `FitSteps` assembly, batch keys, and the loss-name switch (exactly as the
  FCMAE note prescribes). If it proves out, propose promoting the reusable
  bits into `dl_toolbox/dl_engine/task_engines/tabular_pretraining/`.
- **Engines used:** `train_only_engine` (pretrain), `train_val_engine`
  (fine-tune), `find_lr_engine` (LR). DDP across the 2 GPUs via the engine's
  `ProcessRank`/`RunOccupancy`; mlflow or tensorboard logging via the toolbox
  logger contract; checkpointing via `InspectableCheckpointer`.
- **Contracts respected:** no default parameters (large signatures →
  config objects), `Y_dim`/`X_dim` terminology where spatial, and new
  contracts only drafted under `standards/contract/proposals/` — per the
  `python_preferences` / contract-driven-development rules.

### 5.1 Data bridge (keeps competition integration trivial)

1. **Export step** (runs in the comp `.venv`, pandas only): build the
   label-free vocabs + per-(column,slot) support masks + the integer
   code matrix + fold ids, and save them (npz/json) under the comp repo's
   storage. This is the only piece that needs the comp repo; it ships to the
   server with the raw data.
2. **Train on the server** (the `ccf_tabformer` project + its venv).
3. **Export back:** a small script writes
   `storage/artifacts/training/icU_tabformer/{oof.csv,submission.csv,config.json}`
   in the comp repo's `training_io` format, so the member blends through the
   existing `build_te_stack.py` rank-mean harness with zero new glue.

---

## 6. Leakage discipline (unchanged, explicit)

- Vocabs, support masks, digit widths, standardization stats: from train+test
  **features only**, label-free.
- Pretraining touches **no labels** → using test features is fine.
- Fine-tuning per fold uses labels from the 4 training folds only; the
  held-out fold's *features* were seen in pretraining but never its *label*,
  so the OOF stays label-honest and directly comparable to the other members.

---

## 7. Phases, each a gate (pause points marked)

- **A — Export & sanity.** Tokenizer + support masks + code matrix; check
  widths, no NaNs, splits aligned, round-trip decode == original value.
- **B — Pretrain + gates.** MFM pretrain on train+test. Gate: reconstruction
  loss falls, held-out slot accuracy sane; then the **linear-probe** AUC as a
  first read on representation quality.
- **C — Fold-0 pilot. ⏸ PAUSE for your go-ahead.** The 3 regimes
  (pretrained / random / frozen). Decision: proceed to full 5-fold only if
  fold-0 val AUC is competitive (≈ ≥ 0.943, near RealMLP's per-fold) and the
  member looks useful (strong or decorrelated). You review before the hours
  go into 5 folds.
- **D — Full 5-fold.** OOF + submission in `training_io` format.
- **E — Integrate.** OOF AUC, correlation vs existing members, blend search.
  Log to `EXPERIMENTS.md` + a folder README. **Kaggle submit only on your
  explicit OK.**

---

## 8. Honest expectation

On S6E10 this whole family *tied* XGBoost and pretraining added ~nothing over
random-init (v1 §1). The realistic payoff here is a **decorrelated ensemble
member** (a transformer reads the digits unlike GBM/GLM/RealMLP), not a
solo-best score; current best single ≈ 0.949, blend LB 0.95062. The fold-0
gate (Phase C) exists precisely so we stop early if the member is both weak
*and* redundant, before spending server time on 5 folds. The digit-sequence +
MFM angle is also a genuine method to learn and carry to the next synthetic
competition, which is part of the point.

---

## 9. Open items for you

1. **Server data transfer:** git clone + copy `storage/raw_data` and the
   token export, or another path? (Sets the export's output location.)
2. **Tracking backend:** mlflow or tensorboard for this project? (Scaffold
   writes one UI script accordingly.)
3. Confirm the kept-optional toggles stay off by default: magnitude channel
   (§1.3), viability loss (§1.4), smooth-L1 numeric loss (§3).
