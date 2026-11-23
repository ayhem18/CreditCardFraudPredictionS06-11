# Experiment log — CreditCardFraud (playground-series-s6e11)

One row per training experiment, grouped by line, ranked within each group.
All OOF numbers come from the frozen fold artifacts under
`storage/artifacts/training/<exp_name>/` (config.json, oof.csv, fold models).
Keep this file updated when a new experiment lands.

## How to read the numbers

- **Three CV schemes exist.** (1) Global models are scored on the frozen
  *flag-stratified* 5-fold over all 699,960 rows. (2) Region models
  (`rgn_*`) are scored on the frozen *profile-stratified* 5-fold over the
  646,938 no-alarm rows only. (3) The frozen *alarm-profile-stratified*
  5-fold over all rows (`alarm_profile_stratified_cv.json`, 2026-11-19)
  stratifies on alarm × profile (90 cells) × target — 360 theoretical
  cells, 282 strata after the 25-row small-cell collapse. It is the
  **shared scheme**: every row has one fold id, the global trainers score
  on all rows (`--cv alarm_profile`), the region trainers score on the
  same folds restricted to no-alarm rows
  (`train_region_variants.py --cv alarm_profile --exp-name-prefix rgnU`),
  so OOF numbers from the two approaches are directly comparable and
  per-row OOF blends are fold-coherent. Aggregated to alarm × target it
  reproduces scheme (1)'s strata; restricted to the region it reproduces
  scheme (2)'s 153 strata exactly. Numbers from different schemes are
  **never comparable** — a global model's AUC on its no-alarm rows is the
  nearest fair reference for a region model under schemes (1)/(2).
- **OOF is the compass; LB is the check.** Across all five submissions of
  2026-11-18 the OOF ordering predicted the public-LB ordering exactly.
- Baseline for global deltas: `hgb_baseline` OOF **0.94155** (LB 0.94164).

## 1. Global single models (flag-stratified CV, all rows)

| rank | exp | setup | OOF AUC | Δ vs hgb baseline |
|---|---|---|---|---|
| 1 | `xgb_seed_13` | XGB, subsample/colsample 0.8, seed 13 | 0.94505 | +0.0035 |
| 2 | `xgb_tune_learningrate0.05nestimators400` = `xgb_tuned_v1` | XGB, lr 0.05 × 400 rounds | 0.94543* | +0.0039 |
| 3 | `xgb_seed_2026` | stochastic XGB, seed 2026 | 0.94487 | +0.0033 |
| 4 | `xgb_seed_21` | stochastic XGB, seed 21 | 0.94478 | +0.0002 |
| 5 | `xgb_baseline` | XGB, defaults (d6, lr .1, n200) | 0.94454 | +0.0030 |
| 6 | `xgb_tune_maxdepth5` | XGB depth 5 | 0.94530* | +0.0038 |
| 7 | `hgb_baseline` | HistGB, 200 × 31 leaves | 0.94155 | — |

\* tuning grid, full results: lr.05×400 **0.94543** > depth 5 **0.94530** >
min_child_weight 5 **0.94497** > baseline **0.94454** > subsample 0.8
**0.94443** > depth 7 **0.94352** > depth 8 **0.94160**. Direction: slower
and shallower helps; deeper hurts.

Stochastic bag members (singles): seed 7 **0.94438**, seed 13 **0.94505**,
seed 21 **0.94478**, seed 2026 **0.94487**.

## 2. Ensembles and blends (components from the lines above)

| rank | exp | recipe | OOF AUC | public LB |
|---|---|---|---|---|
| 1 | `xgb_bagged_v1` | probability mean of 5 XGBs (full-sample + 4 × subsample/colsample 0.8) | **0.94618** | **0.94659** |
| 2 | `blend_weighted_v1` | alarm 0.25·hgb + 0.75·xgb; no-alarm 0.75·xgb + 0.25·region | 0.94518 | 0.94560 |
| 3 | `two_stage_ensemble_v1` | equal means: alarm hgb+xgb; no-alarm hgb+xgb+region | 0.94486 | 0.94464 |
| 4 | — | global mean hgb+xgb (reference) | 0.94442 | — |
| 5 | — | xgb + region only (reference) | 0.94489 | — |

Weight-search references (OOF): xgb alone 0.94454, equal two-stage 0.94486,
xgb+region 0.94489, best simple 0.25-grid 0.94518, best 0.1-grid 0.94521
(chosen config used the simple weights — robustness rule: grid gain under
0.0003 is selection noise).

## 3. Region models — no-alarm only (profile-stratified CV, 646,938 rows)

Reference: global models restricted to their no-alarm rows —
`hgb_baseline` 0.92673, `xgb_baseline` 0.92993 (global folds, same rows).

| rank | exp | setup | OOF AUC |
|---|---|---|---|
| 1 | `rgn_v4_quantiles_xgb` | drops t/c/card + within-profile quantiles (8 numerics) | **0.92705** |
| 2 | `rgn_no_time_no_card_type_xgb` | drops time + card_type | 0.92683 |
| 3 | `rgn_no_time_no_channel_no_card_type_xgb` (v4) | drops time/channel/card | 0.92672 |
| 4 | `rgn_no_time_no_channel_xgb` | drops time + channel | 0.92645 |
| 5 | `rgn_no_time_xgb` | drops time | 0.92641 |
| 6 | `rgn_full_xgb` | all features minus alarm flags | 0.92625 |
| 7 | `rgn_no_time_no_card_type_hgb` | HGB | 0.92084 |
| 8 | `rgn_no_time_no_channel_no_card_type_hgb` (v4) | HGB | 0.92090 |
| 9 | `rgn_no_time_hgb` | HGB | 0.92043 |
| 10 | `rgn_v4_quantiles_hgb` | HGB + quantiles | 0.92016 |
| 11 | `rgn_no_time_no_channel_hgb` | HGB | 0.91983 |
| 12 | `rgn_full_hgb` | HGB | 0.91925 |
| — | `rgn_v4_profile_id_*` (both families) | + profile_id (90 levels) | **harmful**: hgb 0.91713, xgb 0.92001 |
| — | `rgn_v4_profile_id_quantiles_*` | + id + quantiles | hgb 0.91822, xgb 0.91958 |

Within-family readings: XGB ≫ HGB in the region (+0.006). The quantile
features are a **small positive for XGB** (+0.0003 over v4, positive in
3/5 folds) and neutral-to-negative for HGB. `profile_id` hurts both
families (~−0.004 to −0.007).

## 4. Unified-CV regime comparison (2026-11-19, `*U*` exps)

First numbers on the shared alarm-profile-stratified folds
(`compare_regimes.py`): every row carries the same fold id in both lines,
so these are directly comparable — unlike everything above, which mixes
two schemes. All models retrained unchanged, only the CV differs.

| exp | scope | OOF AUC |
|---|---|---|
| `xgbU_bagged_v2` (10 members, d5+d6) | all rows | **0.94671** |
| `xgbU_bagged_v1` (5 × tuned) | all rows | 0.94650 |
| `xgbU_bagged_v1` | alarm rows | **0.93397** |
| `xgbU_bagged_v1` | no-alarm rows | **0.93187** |
| `xgbU_tuned_v1` (lr .05 × 400) | all rows | 0.94526 |
| `xgbU_baseline` | all rows | 0.94449 |
| `hgbU_baseline` | all rows | 0.94103 |
| `xgbU_tuned_v1` | no-alarm rows | 0.93065 |
| `xgbU_baseline` | no-alarm rows | 0.92964 |
| `rgnU_v4_quantiles_xgb` | no-alarm rows | 0.92592 |
| `hgbU_baseline` | no-alarm rows | 0.92583 |
| `xgbU_tuned_v1` | alarm rows | 0.93117 |
| `xgbU_baseline` | alarm rows | 0.92972 |
| `hgbU_baseline` | alarm rows | 0.92623 |

Readings:

1. **The global model beats the region specialist on its own ground.**
   Tuned XGB on the no-alarm rows: 0.93065 vs the specialist's 0.92592
   (+0.0047, clean sweep 5/5 folds). Specialization does not pay back the
   lost rows and features.
2. **The old diversity finding replicates — and is dominated.** hgb +
   region rank-mean on no-alarm rows = 0.92856, exactly +0.0027 over hgb
   alone (the old cross-scheme headline, now fold-coherent) — but plain
   `xgbU_baseline` alone (0.92964) beats that blend.
3. **Same-family blending is empty.** Best tuned-xgb + region blend
   (0.75/0.25): 0.930668 vs 0.930653 alone (+0.00002). Assembled
   two-stage overall: 0.94529 vs 0.94526 (+0.00003). With the tuned XGB
   as the global component, the region model adds nothing.
4. Scheme-draw effect: the same configs score ~0.0003–0.0005 lower than
   on the flag-stratified draw (hgb 0.94103 vs 0.94155; xgb 0.94449 vs
   0.94454; tuned 0.94526 vs 0.94543). fold 0 is consistently the
   easiest and fold 4 the hardest for every model, so that is fold
   composition, not model behavior.
5. **Bagging the tuned config replicates and keeps paying (2026-11-19/20).**
   `xgbU_bagged_v1` = probability mean of the full-sample tuned + 4 ×
   stochastic members (lr .05 × 400, subsample/colsample 0.8, seeds
   7/13/21/2026): **OOF 0.94650**, +0.0012 over the full-sample single,
   +0.0004 over the best member. Every stochastic member individually
   beats the full-sample tuned: at lr .05 × 400, row/col subsampling
   works as regularization, not just diversity. **LB 0.94668.**
   `xgbU_bagged_v2` doubles the bag to 10 members (+ 2 depth-6 seeds
   3/99, + 3 depth-5 seeds 7/13/2026): **OOF 0.94671, LB 0.94690 —
   competition best**; the OOF delta (+0.0002) matched the LB delta
   almost exactly. The depth-5 stochastic members are the three best
   individuals (0.94643–0.94655) — depth 5 + subsampling is the
   strongest single config found so far. Built with `build_bag.py`.

Verdict: under a fair single scheme, per-region modeling is closed as a
score lever (standalone and as a blend component). The paying direction
stays the global XGB line — bagging the tuned config delivered +0.0012
(`xgbU_bagged_v1` 0.94650, best OOF on this scheme).

6. **Grid-decomposition probe (2026-11-20): explicit anchor/slot IDs
   actively harm tuned XGB.** After the verification battery
   (`experiments/eda/data_analysis/jitter_skeleton_check.py`) showed the
   numerics live on exact decimal grids with conditional fraud signal in
   the fine digits (within-anchor slot AUC up to 0.74 on
   merchant_risk_score), the probe fed that structure to XGB explicitly
   (`grid_slots` / `grid_slots_exact` dataset versions: floor-anchor ID +
   fractional-slot ID as categoricals, + exact-value IDs for the 0.1-grid
   columns). Tuned config, unified CV, free-drop control:

   | exp | setup | OOF AUC | Δ vs `xgbU_tuned_v1` |
   |---|---|---|---|
   | `xgbU_tuned_v1` | control (all columns) | 0.94526 | — |
   | `xgbU_tuned_free_drops` | − time/channel/card | 0.94526 | ±0.0000 |
   | `xgbU_tuned_grid_slots` | + 12 grid IDs | **0.92359** | **−0.0217** |
   | `xgbU_tuned_grid_exact` | + 12 IDs + 2 exact | 0.91718 | −0.0281 |

   Every fold down ~0.02 — not noise. Reading: the high-cardinality
   anchor IDs (1.4k / 5.6k levels) hand the greedy splitter cheap
   memorization splits that crowd out generalizable numeric thresholds;
   adding more IDs (exact) makes it worse. **Verdict: the raw-float
   representation is not just sufficient — the explicit grid encoding
   poisons tree learning. The verified grid signal is real but local,
   and its pooled value is far below the encoding's cost.** Unseen-level
   handling (mapped to missing, not nearest — the raw numeric column
   already carries the position exactly) affects ≤0.08% of test rows,
   immaterial. Possible follow-up: slots-only (no anchors) tests whether
   the low-cardinality slots avoid the poisoning.

7. **Rate-feature campaign (2026-11-20, ≤10 attempts, stopped at 6):
   +0.0018 OOF from anchor statistics — the biggest feature win of the
   competition.** After the ID poisoning, the slot/grid structure was
   re-encoded as numeric reference features (user's proposal). Attempts
   (tuned XGB, unified CV, control 0.94526):

   | # | attempt | features | OOF | Δ |
   |---|---|---|---|---|
   | 1 | `slot_z` | (slot − mean_a)/std_a, target-free | 0.94525 | −0.0000 |
   | 2 | `slot_refs` | fold-safe fraud/clean slot centroids | 0.94520 | −0.0001 |
   | 3 | `anchor_freq` | log1p train count of floor anchor | 0.94554 | +0.0003 |
   | 4 | `anchor_full` | + fold-safe anchor fraud rate (k=20) | 0.94654 | **+0.0013** |
   | 5 | `anchor_v2` | + 0.1-grid rates (velocity/risk) + small-int rates | **0.94703** | **+0.0018** |
   | 6 | `anchor_v3` | + (value × alarm) interaction rates | 0.94688 | +0.0016 (rejected) |
   | 7 | `anchor_v2t` | + common decimal tilt (low blocks, fold-safe) | 0.94709 | +0.0018 (+0.00006 vs av2, flat — not adopted) |
   | 8 | `anchor_v2c` | + combined va×ra anchor rate (logit-add of marginals) | 0.94704 | +0.0018 (+0.00001 vs av2, flat — not adopted) |
   | 9 | `anchor_v2_is_alarm` | + `is_alarm` (OR of the five flags; flags kept) | 0.94679 | +0.0015 (−0.00024 vs av2, 3/5 folds down — not adopted) |

   Attempt 7 (2026-11-22) came out of the hypothesis-first grid EDA
   (`num_grid_eda.md` Q1→Q3→stability, notebooks under
   `experiments/eda/numerical_grid_structure/`): one shared within-anchor
   decimal tilt g(slot) per column in the resolved low blocks (velocity
   ≤ 40, risk ≤ 55; risk permutation p = 0.010, velocity p = 0.075),
   replicating across folds and common to every anchor (per-anchor
   deviations from the shared tilt fully noise-sized, obs/sim 0.82-0.86).
   Encoded fold-safe as `atilt_<col>` (10 levels, neutral 1.0 above the
   block) on top of anchor_v2: +0.00006 — flat.
   Attempt 8 (2026-11-22, user's call) made the joint-anchor finding
   explicit: `acomb_velocity_risk` = sigmoid(logit(p_va) + logit(p_ra) −
   logit(global)) from the two shrunk marginals (q1_joint_anchor_structure
   showed the cell field IS this product, independence p = 0.796).
   +0.00001 vs av2 — flat, as the joint analysis predicted: the product
   adds nothing the trees can't assemble from the two marginals they
   already have. The tilt is real but its
   ±10% within-anchor rate modulation adds nothing beyond arate10's
   value-level rates. anchor_v2 stays champion.
   Attempt 9 was re-checked with a paired hyperparameter sweep before
   rejection (2026-11-23, seed 42, lr 0.05, 400 trees, unified CV): depth
   5, depth 5 + subsample/colsample 0.8, depth 6 + subsample/colsample
   0.8, and min_child_weight 5, each on `anchor_v2` and on
   `anchor_v2_is_alarm`. Deltas for `is_alarm` were −0.00024, +0.00013,
   −0.00026, +0.00030, +0.00001. The only clear positive (+0.00030, depth
   6 subsample) is the size of the old selection-noise rule and is not
   the best config. The best single in the sweep is `anchor_v2` depth 5
   + subsample/colsample 0.8 at **0.94806**; the same config with
   `is_alarm` scores 0.94780 and is down in 5/5 folds.
   `n_alarm_flags` on that same depth-5 subsample config
   (`xgbU_tuned_av2_nflags`, 2026-11-23) scores 0.94780 versus 0.94806
   (−0.00026, 0/5 folds). Not adopted.

   Why it works: rarity and rate are **non-monotone in value** (popular
   100, rare 101), so numeric thresholds cannot recover them and the IDs
   poisoned the splitter — one shrunk scalar per row is the representation
   trees can use. Slot-only signals stayed flat: the decimals matter only
   through their value-level rate. All rates fold-safe (fold-train stats,
   k=20 shrinkage; test rows use full-train payloads). Screens (label-side
   AUC) picked the candidates before spending attempts; native-grid rate
   leaky screens: risk 0.76, risk×alarm 0.83.

   **Submissions:** `xgbU_tuned_av2` (single) OOF 0.94703 → **LB 0.94738**
   (then-best). `xgbU_bagged_v3` (8 members on anchor_v2: full + 4 × d6
   seeds + 3 × d5 seeds; d5 members again best individuals 0.9480-0.9481)
   OOF **0.94824** → **LB 0.94829, competition best** (+0.0014 over the
   pre-feature bag). OOF→LB now 9/9.

## 5. Feature-engineered dataset versions (global, HistGB, drop-column + transforms)

All from `ccf_ml.dataset_versions`, trained as `hgb_<version>`:

| rank | version | change | OOF AUC | Δ |
|---|---|---|---|---|
| 1 | `no_time_no_channel_no_card_type` | drop t+c+card | 0.94149 | −0.0001 (free) |
| 2 | `no_time` | drop time | 0.94133 | −0.0002 |
| 3 | `no_time_no_channel` | drop time+channel | 0.94124 | −0.0003 |
| 4 | `no_time_no_card_type` | drop time+card | 0.94105 | −0.0005 |
| 5 | `risk_aggregates_full` | triple drop + (merchant×auth) risk aggregates | 0.94111 | −0.0005 |
| 6 | `profile_context` | triple drop + within-profile percentiles (global) | 0.94052 | −0.0010 |
| 7 | `risk_context` | + conditional risk replacement for new merchants | 0.93469 | −0.0069 |
| 8 | `risk_aggregates` | as above but drops device + is_new_merchant | 0.93228 | −0.0093 |

Key attributions: `is_new_merchant` alone is worth ~+0.009 (never drop);
the raw row-level risk score is informative even for new merchants
(replacing it cost −0.017 AUC on exactly those rows); group aggregates add
nothing to the global trees; profile percentiles added nothing globally.

## 6. Leaderboard history (public scores)

| date | submission | LB |
|---|---|---|
| 11-16 | HGB baseline | 0.94164 |
| 11-17 | hgb_no_time | 0.94466 |
| 11-17 | hgb_no_time_no_channel | 0.94304 |
| 11-17 | hgb_no_time_no_card_type | 0.94154 |
| 11-17 | hgb_no_time_no_channel_no_card_type | 0.94175 |
| 11-18 | two_stage_ensemble_v1 | 0.94464 |
| 11-18 | xgb_baseline | 0.94564 |
| 11-18 | xgb_bagged_v1 | 0.94659 |
| 11-18 | xgb_tuned_v1 | 0.94637 |
| 11-18 | blend_weighted_v1 | 0.94560 |
| 11-19 | xgbU_bagged_v1 (unified CV, tuned members) | 0.94668 |
| 11-20 | xgbU_bagged_v2 (10 members, depth-mixed) | 0.94690 |
| 11-20 | xgbU_tuned_av2 (anchor_v2 rate features, single) | 0.94738 |
| 11-20 | xgbU_bagged_v3 (8 members on anchor_v2) | 0.94829 |
| 11-21 | (ref 58687518, no description — submitted outside the ledger's sessions; provenance unknown) | 0.94850 |
| 11-22 | xgbU_tuned_av2t (anchor_v2t tilt features, single) | 0.94757 |
| 11-23 | **xgbU_tuned_av2_d5s** (anchor_v2, depth 5, subsample/colsample 0.8; OOF 0.94806) | **0.94858** |
| 11-23 | icU_riskz (intersection_conditional base + merchant×is_new risk robust-z; OOF 0.94832) | 0.94855 |
| 11-23 | **icU_blend4** (rank-mean of av2_d5s + ic_base + ic_riskz + ic_cvv; OOF 0.94862) | **0.94872** |

Note on the intersection_conditional line (2026-11-23, see
`experiments/eda/profile_interactions/`): per-feature conditional columns
(risk/cvv/velocity robust-z, value×context rates) are each only ~+0.0001
OOF and null-to-tied on the LB (icU_riskz 0.94855 ≈ champion), and a
buried-fraud screen shows they add no discriminative power on the frauds
the champion misses (those sit at typical within-cell positions). Their
value is purely as **blend diversity**: a rank-mean of four near-champion
models (0.99-correlated) reached LB **0.94872**, +0.00014 over the best
single — the paying use is ensembling them, not stacking features into one
model (the in-model cvv+riskz combine was worse than either single).

Note: LB differences under ~±0.001 between near-identical models are
public-set noise (the 11-17 drop-ablation spread proves it); OOF ordering
has tracked LB ordering for genuinely different models.

## 7. What we believe (current state of knowledge)

1. Column surgery is exhausted: time/channel/card_type are free drops;
   every hand-built feature variant was null or negative in the global
   model. The remaining signal the trees miss (per the ROI mining) is not
   pocket-position-shaped.
2. XGBoost > HistGB everywhere on this data (+0.003 global, +0.006 region).
3. Ensembling is the paying direction: bagging (+0.0014) and blending
   (+0.0006 over best single) both held up on the LB.
4. The region line's quantile features are the only hand-built features
   with a positive sign anywhere (region XGB, +0.0003) — honest verdict:
   real but marginal.
5. Next queued: ~~bag the tuned config~~ done 2026-11-19 (`xgbU_bagged_v1`
   OOF 0.94650, §4); the numerics-expressiveness probes (poly-LR deg 2/3,
   EBM/GAM, single-tree depth sweep) are designed but not run; a
   depth-mixed bag (5+6) is the obvious bag follow-up.
6. Unified CV frozen (2026-11-19): retrain the global xgb/hgb baselines
   with `--cv alarm_profile` and the region ladder with
   `--cv alarm_profile --exp-name-prefix rgnU` to get the first
   directly-comparable per-region vs global numbers (and a fold-coherent
   re-run of the blend search). **Done same day — see §4.** Global beats
   the region specialist on its own ground (+0.0047, 5/5 folds); the
   region model adds nothing to the tuned XGB, standalone or blended.
7. Per-region modeling is closed as a score lever (2026-11-19, §4). The
   only surviving use of the region line is as a diversity component for
   a *different-family* global (hgb + region = +0.0027), and even that is
   dominated by any plain XGB.
