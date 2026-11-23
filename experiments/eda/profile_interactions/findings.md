# Findings — profile_interactions

# ═══════════════════════════════════════════════════════════════════════
# ENGINEERED FEATURE REFERENCE — intersection_conditional family
# (exact definitions for independent verification, 2026-11-23)
# ═══════════════════════════════════════════════════════════════════════

Everything below is for you to interpret. Each feature is given as: the
**cell** (categorical profile it is aggregated on), the **exact formula**,
and how **fold-safety** is handled. Then every **experiment** with its
OOF number. Code lives in `ccf_ml/src/ccf_ml/dataset_versions/`
(`constants.py`, `transforms.py`, `specs.py`); training in
`experiments/baselines/train_xgb.py`; build via
`scripts/build_dataset_versions.py --version <name>`.

## 0. Conventions that apply to every experiment

- **Model (fixed, = the champion's config):** XGBoost, `max_depth=5`,
  `learning_rate=0.05`, `n_estimators=400`, `subsample=0.8`,
  `colsample_bytree=0.8`, `min_child_weight=1`, `reg_lambda=1`,
  `random_state=42`, `tree_method=hist`, `enable_categorical=True`.
  Nothing is tuned per feature — a feature's effect is measured at the
  champion's operating point.
- **CV (fixed):** the frozen `alarm_profile` 5-fold scheme
  (`--cv alarm_profile`). Every OOF number below is on these folds, so all
  are directly comparable.
- **Reference to beat:** `intersection_conditional_base` OOF **0.94819**
  (not the champion's 0.94806 — see §1). A feature's "Δ" is
  `base+feature − base`, same folds, same seed. "folds up" counts the 5
  folds where the feature's per-fold AUC exceeds the base's.
- **Adoption bar (user, 2026-11-23):** Δ ≥ **+0.0001** AND ≥ 3/5 folds up.
- **Two fold-safety regimes** (which one each feature uses is stated):
  - *Rate features use the label* → must be fold-safe. For a training row
    the rate is computed from the **other four folds only**
    (`_fold_of_row` assigns each row its validation fold; the table is
    rebuilt on the complement). The test set uses one **full-train** table.
    Shrinkage `k = IC_RATE_SHRINKAGE = 20`.
  - *Placement features (z, percentile, deviation) are target-free* — they
    use only the numeric's own values and the categorical columns, never
    the label — so there is no leakage and they are computed once on the
    **full train** frame and reused unchanged for test (same treatment as
    the anchor frequencies). Cell stats need `≥ IC_CELLZ_MIN_ROWS = 200`
    rows or they fall back to the global stat.
- **Key encoding** (for any cell / rate group): each key column is turned
  into a string part and parts are joined with `"|"`. Numeric key modes:
  `value = round(x)` (`np.rint`), `floor = ⌊x⌋`, `tenth = round(10x)`;
  categorical mode `raw` = the category string. Example cvv cell key for a
  POS/iPhone row with cvv 1 → `"1|POS|iPhone"`.

## 1. `intersection_conditional_base` — the foundation (attempt 1)

- **What it is:** the full `anchor_v2` feature set **plus `channel` and
  `card_type` added back**; only the time columns (`time_of_day_hour`,
  `day_of_week`) are dropped. 41 columns.
- **Inherited anchor_v2 columns** (unchanged, see `EXPERIMENTS.md` §4/§7
  and `specs.py`): the 20 raw columns, `afreq_<col>` (6, target-free log
  train frequency of the integer anchor), `arate_<col>` (6 continuous, at
  integer anchors), `arate10_velocity_score` / `arate10_merchant_risk_score`
  (0.1-grid rates), and `arate_customer_age` / `arate_cvv_retry_count` /
  `arate_txn_count_last_24h` (per-value rates). **Note for interpretation:
  the base already carries a GLOBAL per-value cvv and txn-count rate**, so
  the cvv feature in §4 only adds the channel×device *interaction*.
- **Why re-add channel/card_type:** the conditional features use `channel`
  (and `device_type`); the raw categoricals are kept so the tree can
  combine them with the conditional columns. They were "free drops" for
  anchor_v2, so the base is expected ≈ champion.
- **Result:** OOF **0.94819** vs champion 0.94806 (+0.00013, within noise).
  Confirms the re-add is neutral; it becomes the reference for §2–§6.

## 2. `icrate_prior_disputes` — marginal dispute rate (attempt 2)

- **Cell (profile):** NONE — global, keyed on the value of
  `prior_disputes` alone (`key_cols=("prior_disputes",)`, mode `value`).
- **Exact formula (per value v):**
  `rate(v) = (frauds_with_disputes=v + 20·p̄) / (rows_with_disputes=v + 20)`,
  where `p̄` is the global fraud rate. Each row gets `rate(round(its
  prior_disputes))`. Fold-safe (rate regime).
- **Rationale:** `prior_disputes` is the one small-int column with **no**
  rate in anchor_v2, so even the marginal rate is new.
- **Observed table (full-train):** v=0 → 0.0155, v=1 → 0.0236, v=2 → 0.0300,
  v=3 → 0.0342 (monotone increasing).
- **Dataset version:** `intersection_conditional_disputes` (42 cols).

## 3. `iczs_merchant_risk_score` / `icza_merchant_risk_score` — coarse risk robust-z (attempt 3)

- **Cell (profile):** `merchant_category × is_new_merchant`
  (`key_cols=("merchant_category","is_new_merchant")`, both mode `raw`;
  ~24 cells). Chosen because risk's distribution shift and the new-merchant
  anti-informativeness both live here.
- **Exact formula (per row, risk x in its cell c):**
  - `scale_c = 1.4826 · median(|x − median_c|)` over cell c's rows
    (robust σ); if cell c has `< 200` rows or `scale_c = 0`, use the global
    `scale`; same global fallback for `median_c`.
  - `iczs = (x − median_c) / scale_c`  (signed; how far above/below the
    cell's typical risk, in robust SDs)
  - `icza = |iczs|`  (magnitude; **non-monotone** outlierness — the shape no
    raw-risk threshold can reproduce)
- **Fold-safety:** target-free (median/MAD use no label) → computed on full
  train, reused for test.
- **Dataset version:** `intersection_conditional_riskz` (43 cols).

## 4. `icrate_cvv_ctx` — cvv × context rate (attempt 4)

- **Cell (profile):** `cvv_retry_count (value) × channel × device_type`
  (`key_cols=("cvv_retry_count","channel","device_type")`, modes
  `("value","raw","raw")`; ~135 observed cells).
- **Exact formula (per cell c = (round(cvv), channel, device)):**
  `rate(c) = (frauds_in_c + 20·p̄) / (rows_in_c + 20)`; each row gets its
  cell's rate. Fold-safe (rate regime), unseen cells → global `p̄`.
- **What's new vs base:** base already has the *global* cvv value rate, so
  this is purely the **interaction** — it only helps if the cvv lift varies
  by channel/device. It does in the data: cvv=1 fraud rate ranges
  **0.042–0.103** across cells (vs cvv=0 0.006–0.013).
- **Dataset version:** `intersection_conditional_cvv` (42 cols).

## 5. `iczs_risk_fine` / `icza_risk_fine` — fine escalated risk robust-z (attempt 5)

- **Cell (profile):** the **full 7-atom behavioural profile** with
  **drop-one escalation**. Atoms = `(channel, device_type,
  is_new_merchant, merchant_category, is_alarm, card_type, auth_method)`.
  Per row the effective cell is: the full-7 cell if it has ≥ 200 rows; else
  the **largest** single-atom-dropped (6-atom) cell with ≥ 200 rows; else
  the global stat. (`is_alarm` is the OR of the five alarm flags.)
- **Exact formula:** identical robust-z to §3
  (`(x − median_cell)/scale_cell`, signed + absolute), but `median`/`scale`
  come from the escalated effective cell. Target-free.
- **Purpose:** encode a **local** risk norm a depth-5/6 tree cannot
  assemble (reach a 7-way cell AND rank within it). This is the "real B1".
- **Dataset version:** `intersection_conditional_riskz_fine` (43 cols).

## 6. Experiments run (all on the fixed config + alarm_profile CV)

Per-fold AUCs are fold 0 / 1 / 2 / 3 / 4 (fold 0 easiest, fold 4 hardest).

| # | exp_name / version | feature(s) added to base | OOF | Δ vs base | folds up | per-fold AUC | verdict |
|---|---|---|---|---|---|---|---|
| 1 | `icU_base` / intersection_conditional_base | (re-add channel, card_type) | 0.94819 | — (ref) | — | 0.95165 / 0.94743 / 0.94868 / 0.94813 / 0.94519 | reference |
| 2 | `icU_disputes` / …_disputes | icrate_prior_disputes | 0.94818 | −0.00001 | 2/5 | 0.95157 / 0.94740 / 0.94881 / 0.94804 / 0.94521 | rejected (flat; monotone rate the tree already holds) |
| 3 | `icU_riskz` / …_riskz | iczs_ + icza_merchant_risk_score (merchant×is_new) | **0.94832** | **+0.00013** | 3/5 | 0.95165 / 0.94764 / 0.94884 / 0.94807 / 0.94556 | best single OOF; **LB 0.94855 vs champion 0.94858 (−0.00003, tied/noise)** |
| 4 | `icU_cvv` / …_cvv | icrate_cvv_ctx (cvv×channel×device) | 0.94829 | +0.00011 | 3/5 | 0.95126 / 0.94754 / 0.94909 / 0.94807 / 0.94563 | keeper; big on hard folds 2/4, loses easy fold 0 |
| 5 | `icU_cvv_riskz` / …_cvv_riskz | icrate_cvv_ctx + risk z (both §3,§4) | 0.94821 | +0.00002 | 1/5 | 0.95152 / 0.94742 / 0.94861 / 0.94796 / 0.94569 | rejected — **combine WORSE than either single** (they interfere) |
| 6 | `icU_riskz_fine` / …_riskz_fine | iczs_ + icza_risk_fine (7-atom escalated) | 0.94826 | +0.00008 | 3/5 | 0.95155 / 0.94732 / 0.94882 / 0.94837 / 0.94545 | below bar; worse than coarse §3 despite higher standalone AUC |

**LB confirmation (2026-11-23):** `icU_riskz` scored **LB 0.94855** vs the
champion's **0.94858** — −0.00003, i.e. tied within public-set noise. The
best OOF feature (+0.00026 over the champion's OOF, counting the base
re-add) did **not** improve the LB. First break in the previously-perfect
OOF→LB tracking, and exactly what the buried-fraud screen (dated section
below) predicted: the conditional features carry no real signal the model
lacks. Reading: the intersection_conditional direction is confirmed
marginal/null on the leaderboard, not just on OOF.

Single-feature leaky label-AUCs (full-train, for context only, NOT the
model deltas): icrate_prior_disputes 0.529; iczs_risk(coarse) 0.694 /
icza 0.573; icrate_cvv_ctx 0.714; iczs_risk_fine 0.737 / icza 0.620; raw
merchant_risk_score 0.749. (Higher standalone AUC did **not** predict a
bigger model Δ — see §6 rows 3 vs 6.)

## 7. The buried-fraud battery (screen only — NOT trained as model features)

`experiments/eda/profile_interactions/buried_battery_screen.py` builds, for
each of amount/risk/velocity/customer_age, a battery of **target-free**
placement features at two cell granularities (top-1 and top-2 interacting
cats from the eta² map) and scores each by within-k6-cell AUC on the 1,941
buried frauds vs their clean confusers. Battery per (numeric, cell):
`signed_z`, `abs_z`, `within_pct` (within-cell rank percentile), `raw_dev`
(x − cell median), `cell_median` (the cell's median as a stand-alone
column). Exact cell definitions and the verdict are in the dated section
immediately below.

# ───────────────────────────────────────────────────────────────────────

## 2026-11-23 — buried_battery_screen.py: distribution-shift → feature battery → buried-fraud screen

Script: `experiments/eda/profile_interactions/buried_battery_screen.py`
(run: `./.venv/Scripts/python.exe -I experiments/eda/profile_interactions/buried_battery_screen.py`).
Full result table: `storage/artifacts/profile_interactions/buried_battery_screen/tables/battery_screen_results.csv`.
This section documents the **mechanics** so the numbers can be checked
independently; readings are at the end and are labelled as readings.

### Inputs (exact)

- **Numerics screened:** `amount_usd`, `merchant_risk_score`,
  `velocity_score`, `customer_age`.
- **Candidate cats (7 atoms):** `channel`, `device_type`,
  `is_new_merchant`, `merchant_category`, `is_alarm` (= OR of the five
  alarm flags, via `alarm_flag_mask`), `card_type`, `auth_method`.
- **Buried frauds:** the **1,941** rows with `is_fraud=1` and champion OOF
  probability **< 0.02** (threshold set in the mistakes_v2 build;
  `mistakes_plan.md` M1). Loaded from
  `…/mistakes_v2/tables/map1/buried_ids.npy` as **positional row indices**
  into `train.csv` (verified: `train.iloc[buried].is_fraud.mean() == 1.0`).
- **Confuser sets (k6):** for each buried fraud, its clean neighbours at
  overlap level **k=6** — clean rows matching the fraud on **6 of the 7
  atoms** — from the prebuilt index
  (`map1/k6_clean_ids.i32` + `map1/k6_offsets.npy`). Per fraud: median
  **2,709** cleans, range **3–23,744**; **8,535,686** fraud–clean pairs
  total (`map1_overlap_summary.csv`, k=6 row). These are the cleans the
  champion sits next to in categorical space — the rows it confuses each
  buried fraud with.

### A. Distribution-shift metric (exact)

For a (numeric, cat): rank-transform the numeric **once over all 699,960
rows** (`pd.Series(num).rank()`), then
`eta² = Σ_levels n_level·(mean_rank_level − grand_mean_rank)² / Σ_rows (rank − grand_mean)²`
— the fraction of the rank's total variance that lies **between** the cat's
levels. 0 = numeric distribution identical across levels; →1 = level
determines the numeric. `eta2_rank` in the script. Full 4×7 table:

| numeric | channel | device_type | is_new_merchant | merchant_category | is_alarm | card_type | auth_method |
|---|---|---|---|---|---|---|---|
| amount_usd | 0.0002 | 0.0 | 0.0000 | **0.3552** | 0.0000 | 0.0001 | 0.0001 |
| merchant_risk_score | 0.0000 | 0.0 | 0.0001 | **0.2331** | 0.0002 | 0.0001 | 0.0003 |
| velocity_score | 0.0000 | 0.0 | 0.0000 | 0.0003 | 0.0002 | 0.0000 | 0.0001 |
| customer_age | 0.0007 | 0.0 | 0.0000 | 0.0006 | 0.0001 | 0.0000 | 0.0003 |

The script then sets each numeric's **sub-profile** to its top cats by
eta². For amount/risk that is a real signal (`merchant_category`); for
velocity/age the top cat is ranked off eta²≈0.0003–0.0007, i.e. noise, so
those two have no genuine interacting cat (stated for transparency — the
screen still builds their features at the nominal top cat).

### B. The feature battery (exact formulas)

For a numeric `x` and a **cell** = `groupby` over one or two cat columns
(`within_cell_features` in the script). Let `m_c` = median of `x` in the
row's cell, `MAD_c = median(|x − m_c|)` in the cell, and the global
fallbacks `m_*`, `scale_* = 1.4826·MAD_*` (or std, or 1). A cell with
**< 200 rows** uses the global `m_*`, `scale_*`. Then:

| feature | formula | monotone in x? |
|---|---|---|
| `signed_z` | `(x − m_c) / scale_c`, `scale_c = 1.4826·MAD_c` (global if MAD_c=0) | yes (within a cell) |
| `abs_z` | `|signed_z|` | **no** (both tails) |
| `within_pct` | rank of `x` within the cell, `pct=True` (0–1) | yes |
| `raw_dev` | `x − m_c` | yes |
| `cell_median` | `m_c` (the cell's own median, as a column — no `x`) | — |

All five are **target-free** (use only `x` and the cat columns, never the
label), computed once over all train rows. Each numeric is screened at
**two granularities**: its top-1 cat cell and its top-1+top-2 cat cell.

### C. The screen metric (exact)

For a feature vector `f` over all train rows, the **within-k6 AUC** is the
pooled rank statistic over all 8,535,686 (buried fraud `i`, confuser `j`)
pairs:
`AUC(f) = [ Σ_pairs 1{f[i] > f[j]} + 0.5·1{f[i] == f[j]} ] / (#pairs)`
(`screen(f)` — `np.repeat(f[buried], neighbour_counts)` compared to
`f[clean_ids]`). It answers: **within the cells where the champion fails,
does `f` rank the buried fraud above the cleans it was confused with?**
0.5 = no better than coin flip.
`raw_auc` = `screen(x)` on the raw numeric (the baseline the model already
has). `added = screen(f) − raw_auc`. **Leaky-direction note:** the pairs
are formed using the labels, so this is an upper-bound-flavoured screen —
but every `f` here is target-free, so its AUC reflects the feature's own
discrimination, and the raw numeric is the matched baseline, so `added`
isolates what conditioning contributes over the raw value.

### Full results (every row of battery_screen_results.csv)

`screen_auc` = within-k6 AUC of the feature; `raw` = the numeric's own
within-k6 AUC; `added` = screen_auc − raw.

| numeric | cell (sub-profile) | feature | screen_auc | raw | added |
|---|---|---|---|---|---|
| amount_usd | — | **RAW** | 0.5187 | 0.5187 | 0.0000 |
| amount_usd | merchant_category | abs_z | 0.5156 | 0.5187 | −0.0031 |
| amount_usd | merchant_category+channel | abs_z | 0.5147 | 0.5187 | −0.0041 |
| amount_usd | merchant_category | signed_z | 0.5125 | 0.5187 | −0.0063 |
| amount_usd | merchant_category | raw_dev | 0.5120 | 0.5187 | −0.0068 |
| amount_usd | merchant_category+channel | signed_z | 0.5118 | 0.5187 | −0.0070 |
| amount_usd | merchant_category+channel | raw_dev | 0.5116 | 0.5187 | −0.0071 |
| amount_usd | merchant_category+channel | within_pct | 0.5103 | 0.5187 | −0.0085 |
| amount_usd | merchant_category | within_pct | 0.5101 | 0.5187 | −0.0087 |
| amount_usd | merchant_category | cell_median | 0.5074 | 0.5187 | −0.0113 |
| amount_usd | merchant_category+channel | cell_median | 0.4997 | 0.5187 | −0.0191 |
| customer_age | channel | signed_z | 0.6030 | 0.6030 | 0.0000 |
| customer_age | channel | within_pct | 0.6030 | 0.6030 | 0.0000 |
| customer_age | channel | raw_dev | 0.6030 | 0.6030 | 0.0000 |
| customer_age | — | **RAW** | 0.6030 | 0.6030 | 0.0000 |
| customer_age | channel+merchant_category | within_pct | 0.6027 | 0.6030 | −0.0003 |
| customer_age | channel+merchant_category | signed_z | 0.6026 | 0.6030 | −0.0004 |
| customer_age | channel+merchant_category | raw_dev | 0.6025 | 0.6030 | −0.0005 |
| customer_age | channel+merchant_category | cell_median | 0.5107 | 0.6030 | −0.0923 |
| customer_age | channel | cell_median | 0.4999 | 0.6030 | −0.1032 |
| customer_age | channel+merchant_category | abs_z | 0.4619 | 0.6030 | −0.1411 |
| customer_age | channel | abs_z | 0.4588 | 0.6030 | −0.1442 |
| merchant_risk_score | — | **RAW** | 0.6718 | 0.6718 | 0.0000 |
| merchant_risk_score | merchant_category | raw_dev | 0.6604 | 0.6718 | −0.0114 |
| merchant_risk_score | merchant_category | within_pct | 0.6602 | 0.6718 | −0.0117 |
| merchant_risk_score | merchant_category | signed_z | 0.6601 | 0.6718 | −0.0117 |
| merchant_risk_score | merchant_category+auth_method | raw_dev | 0.6600 | 0.6718 | −0.0119 |
| merchant_risk_score | merchant_category+auth_method | within_pct | 0.6598 | 0.6718 | −0.0120 |
| merchant_risk_score | merchant_category+auth_method | signed_z | 0.6597 | 0.6718 | −0.0122 |
| merchant_risk_score | merchant_category+auth_method | cell_median | 0.5336 | 0.6718 | −0.1383 |
| merchant_risk_score | merchant_category | cell_median | 0.5263 | 0.6718 | −0.1456 |
| merchant_risk_score | merchant_category+auth_method | abs_z | 0.5197 | 0.6718 | −0.1521 |
| merchant_risk_score | merchant_category | abs_z | 0.5192 | 0.6718 | −0.1526 |
| velocity_score | — | **RAW** | 0.6490 | 0.6490 | 0.0000 |
| velocity_score | merchant_category | within_pct | 0.6484 | 0.6490 | −0.0006 |
| velocity_score | merchant_category | raw_dev | 0.6484 | 0.6490 | −0.0006 |
| velocity_score | merchant_category+is_alarm | within_pct | 0.6484 | 0.6490 | −0.0005 |
| velocity_score | merchant_category+is_alarm | raw_dev | 0.6484 | 0.6490 | −0.0006 |
| velocity_score | merchant_category | signed_z | 0.6483 | 0.6490 | −0.0007 |
| velocity_score | merchant_category+is_alarm | signed_z | 0.6483 | 0.6490 | −0.0007 |
| velocity_score | merchant_category | cell_median | 0.5204 | 0.6490 | −0.1285 |
| velocity_score | merchant_category+is_alarm | cell_median | 0.5200 | 0.6490 | −0.1290 |
| velocity_score | merchant_category+is_alarm | abs_z | 0.4984 | 0.6490 | −0.1506 |
| velocity_score | merchant_category | abs_z | 0.4976 | 0.6490 | −0.1514 |

### Caveats for interpretation (so you can judge the readings)

1. **Leaky direction.** Pairs are formed with the labels; absolute AUCs are
   optimistic. Only *relative* comparisons (engineered vs its `raw`
   baseline) are meant to be trusted.
2. **Cell ≠ k6 cell.** A feature's cell (e.g. `merchant_category`) is NOT
   the k6 neighbourhood the screen pairs within (6-of-7 atoms, which atom
   varies per fraud). The screen asks only "does this column, however
   built, rank buried frauds above their k6 confusers." It does not require
   feature-cell = screen-cell.
3. **What is NOT tested here.** Only *target-free placement* of a *single*
   numeric. Not tested: label-based value×cat **rates** (would score ≈1 by
   leakage under this screen — need a fold-safe model test instead),
   **numeric×numeric** interactions, or any multi-column construction.
4. **k6 definition** is taken from the prebuilt mistakes_v2 map (6-of-7
   atom match); the exact matching predicate lives in that builder and was
   not re-derived here.

### Readings (tied to the numbers above — verify against the table)

- **Every `added` ≤ 0.** No engineered placement of any of the 4 numerics
  beats the raw value at ranking buried frauds vs their k6 confusers. The
  raw numeric's within-k6 AUC (risk 0.672, velocity 0.649, age 0.603,
  amount 0.519) is the ceiling; `signed_z`/`within_pct`/`raw_dev` land
  just below it (risk −0.011, velocity −0.001, age 0.000).
- **`abs_z` (outlierness) is far below chance-adjacent** (risk 0.519,
  velocity 0.498, age 0.459) — reading: buried frauds are *more central*
  within their cells than the confuser cleans, i.e. they are typical, not
  outliers. If correct, this refutes the "conditional outlier" (risk-55)
  hypothesis on the actual miss set.
- **`cell_median` alone is ~0.50–0.53** (risk 0.526, velocity 0.520) —
  reading: the cell's typical level barely separates buried frauds, so the
  separation that exists lives in each row's deviation-from-cell, which is
  what the raw value already encodes.
- **amount:** biggest eta² (0.355) yet raw within-k6 AUC only 0.519 —
  reading: distribution shift does not imply buried-fraud signal.
- These are the direct, failure-set reading of the irreducible-overlap
  result (joint-space NN ratio ~1.05). They are consistent with the
  model deltas (§6) and the riskz LB tie, but note caveats 1–3 bound their
  scope: this screens placement features, not rates or interactions.

## 2026-11-23 — profile_quantiles_v1.py: within-profile numerical placement screen

Design: premise (user) — once a categorical profile is set, only the
numerics decide; the screen asks whether a numeric's WITHIN-PROFILE
percentile ranks frauds above the cleans of the same profile
(stratified Mann-Whitney over same-profile fraud-clean pairs), compared
to the same transformed variable used globally. Label-side screen (leaky
direction, no folds); features get built fold-safe only for winners.
Profile7 = user's 7 features with escalation (full7 -> largest drop-one
>= 200 rows -> merchant+auth+is_alarm); level mix: full7 301k rows,
drop:merchant 138k, drop:device 81k, base 54k, others smaller.

Results (`storage/artifacts/profile_interactions/profile_quantiles_v1/tables/within_profile_auc.csv`):

- **cvv_retry_count is the standout**: within-profile AUC 0.706-0.709
  across every family while the pooled percentile variable manages only
  0.60-0.64 — a within-minus-marginal gap of +0.10 to +0.11, ten times
  anyone else's. cvv's ordering power lives almost entirely INSIDE
  profiles; its absolute value means little. This quantifies the user's
  cvv x device/channel intuition.
- merchant_risk_score / velocity_score percentiles: highest absolute
  within-AUCs (0.71-0.75) but tiny within-minus-marginal gaps (+/-0.005
  to +0.018) — mostly the same signal the raw columns already carry.
  Real but modest: risk within merchant+is_new (+0.018) and within
  profile7 (+0.016) — supports the "risk aggregated per
  (is_new_merchant, merchant_category)" idea at small size.
- Reading: the remaining space is not "profile rates" (always low, and
  the model has them) but profile-relative numeric placement, and the
  one numeric whose placement is disproportionately informative is cvv.

Caveats: leaky-direction screen (upper-bound flavor); the earlier
`profile_context` dataset version (2026-11-17, HistGB, 90-profile
hierarchy, within-profile percentiles of 9 numerics) was flat — but that
was pre-champion, a model result not a screen, and did not target cvv.
Next step if confirmed: fold-safe per-profile cvv percentile (plus risk
within merchant+is_new) on top of the current tuned champion.

## 2026-11-23 — numeric_cat_interactions_v1.py: conditional-CDF maps (label-free)

User's intuition-algorithm: integer-ize each numeric, take its top-10%
most frequent integer values, show P(numeric <= v | category) vs the
marginal CDF. 77 two-panel maps (CDF + deviation) in
`storage/artifacts/profile_interactions/numeric_cat_interactions_v1/plots/cdf_maps/`,
summary in `tables/conditional_cdf_deviation_summary.csv`. Note: for
integer thresholds raw and anchor views coincide. (First run had a
denominator bug — cum.sum() instead of tab.sum() — marginals were ~0;
fixed and rerun. The earlier stats-battery version of this script was
killed and replaced by this visual form.)

- **Two numerics cannot be read in isolation from merchant_category:
  amount_usd (max deviation 0.853: P(amount <= 72 | Crypto Exchange) =
  0.043 vs marginal 0.895 — Crypto lives almost entirely ABOVE $72
  while ~90% of all rows live below) and merchant_risk_score (0.676:
  P(risk <= 39 | Crypto) = 0.004 vs 0.679 — Crypto sits at the high end
  of risk). Their median map deviations (0.085 / 0.074) are an order of
  magnitude above everyone else's.
- Everything else is distributionally context-free: balance, age,
  velocity, hours, distance, card_age, txn_count, prior_disputes all
  max out at |dev| <= 0.04, usually on Crypto/Streaming/Utilities.
- **cvv structural-zero hypothesis REFUTED** (tested as stated):
  P(cvv = 0 | channel x device_type) spans only 0.902-0.935 across all
  40 cells (auth_method: 0.897-0.923). No context where CVV "isn't
  used" — cvv is a uniformly ~91.5%-zero column everywhere; consistent
  with the generator drawing it iid. Its fraud signal remains
  within-profile placement (profile_quantiles_v1), not distribution
  shift.
- Chain implication: unusualness sub-profiles should be
  merchant_category-centric for amount and risk; for cvv the
  distribution gives no sub-profile guidance — use the k=6 profile
  from the mistake analysis instead.

## 2026-11-23 — numeric_cat_interactions_v2.py: user-specified bubble plots

v1's long heat maps replaced (user design): per (numeric, cat) two
scatter plots — diameter = P(num <= v | cat) (cdf__*.png, grouping read)
and diameter = |P(cond) - P(marg)| with red/blue sign (dev__*.png) —
plus hist__*.png (N histograms together) for categoricals with <= 5
levels. 254 plots in
`storage/artifacts/profile_interactions/numeric_cat_interactions_v2/plots/plots/`.
Same computation as v1 (tables unchanged).
