# Time-feature interaction analysis — plan

Hypothesis: the data generator changes across pockets of the sample space, and those pockets are mostly combinations of categorical features. The time features (`time_of_day_hour`, `day_of_week`) either help define those pockets or are noise. This analysis decides which one it is. The outcome is one of:

- **drop**: time is near-independent of everything that matters → remove the raw columns from the model;
- **distill**: a real intersection exists → replace raw hour/day with bucket features (e.g. night/day) or interaction features;
- **keep**: the raw values themselves carry signal.

Scope: full train (699,960 rows). Every key test is repeated inside the alarm region and the no-alarm region, because a global test can average away a pocket effect. The earlier ROI-only result (no-alarm, `y_prob <= 0.02`: fraud independent of the 7x24 cell, merchant mix independent of time) is referenced, not repeated.

## Modules

**M1 — Are transaction counts uniform over the clock?**
Counts per hour, per day, and per 7x24 cell against a uniform expectation, plus an hour x day independence test (does the joint cell add anything beyond the two marginals?). Descriptive only: an uneven clock describes the count process, not fraud, and triggers no action by itself.

**M2 — Does fraud depend on time?**
Fraud rate by hour with confidence bands, chi-square and MI for hour / day / joint cell, separately for all, alarm, no-alarm.

**M3 — Do categorical features depend on time?**
Each categorical column (merchant, card, auth, channel, device, new-merchant, flags) against the joint time cell: MI + chi-square. Merchant also per region, plus a lift heatmap.

**M4 — Do numeric features depend on time?**
Each numeric feature (time columns excluded) bucketed into deciles against the joint time cell (MI), plus a rank test across the 24 hours (Kruskal-Wallis, effect size epsilon-squared). The named ones — `amount_usd`, `velocity_score`, `merchant_risk_score` — also get per-hour mean-deviation plots and per-region MI. Money uses the log scale for means.

**M5 — Age special case.**
Does customer age depend on time? If yes, the candidate feature "customer's age minus the average age at that time" gets sized (share of age variance explained by hour). If no, the idea is dropped instead of built.

**M6 — Does the baseline model actually use time?**
Permutation check on each fold's validation rows: shuffle hour, shuffle day, shuffle both, and shuffle `merchant_risk_score` as a reference scale. AUC drop per shuffle, averaged over folds. If the time drops are ~0 while the reference drop is large, the model itself agrees the columns are dead weight.

**M7 — Decision table.**
One row per test: statistic, effect size, verdict, suggested action. Saved as `decision_table.csv`.

## How we decide (fixed before looking at the results)

An interaction is called real only if **both** hold:

1. **Beats the shuffle null.** Recompute MI 10 times with the time labels randomly permuted. The observed MI must beat the best shuffled value by at least 10%.
2. **Effect size clears a floor.** Cramer's V >= 0.02 for categorical pairs and the label; epsilon-squared or eta-squared >= 0.005 for numerics.

Verdict bands: `independent` (fails the null) → drop candidate; `detectable_not_exploitable` (passes the null, effect under the floor) → still a drop candidate, the pattern is real but too small to use; `weak` (V 0.02–0.05) → bucket / interaction candidate; `strong` (V >= 0.05) → distill properly.

Why both gates: with ~700k rows a chi-square rejects deviations nobody could use, and with 168 cells a raw MI is biased upward. The shuffle null calibrates the statistic; the effect size caps the enthusiasm.

## M4b: per-cell distribution profiles (added after review)

For each continuous numeric feature and each of the 168 time cells, compare the cell's distribution against the pooled one through a fixed set of statistics: mean, sd, quantiles q1 / q5 / q10 / q25 / q50 / q75 / q90 / q95 / q99, IQR, and the robust range q99 − q1. If none of these move beyond sampling noise, the cell distributions are the same for practical purposes.

Each cell statistic is standardized by its sampling error under independence (the cells are disjoint samples, so under the null the 168 z values behave like standard normal draws):

- quantile: `SE(q_p) = sqrt(p(1-p)/n_cell) / f(q_p)`, density estimated from the pooled quantile curve;
- mean: `sd / sqrt(n_cell)`; sd: `sd / sqrt(2 n_cell)`;
- IQR and robust range: from the quantile variances and covariances of their two endpoints.

Pre-registered flags, per (feature, statistic): variance ratio `mean(z^2)` outside **1 ± 0.33** (its own noise is ±0.11 at 168 cells), or a single cell with **|z| > 4.5** (the expected max of 168 standard-normal draws is ~3.3).

Boundaries declared up front: discrete features (≤ 30 unique values) are excluded here because their quantiles are step functions — their level counts are already covered by M4's decile test; money columns are tested on the log scale (monotone transform, same null); at ~4.2k rows per cell only q99 (~42 rows) is thin, accepted with that caveat; the whole module runs on the full train only (regions would have ~300 rows per cell, too thin for tails).

Output: `m4b_quantile_moment_tests.csv` (one row per feature x statistic: variance ratio, max |z|, flag) plus heatmaps of z for the largest deviations. The M7 `decision_table.csv` predates M4b and is not rewritten.

**Repair (after the first run).** The first run failed its own sanity check (pooled z sd ≈ 1e13): integer-valued features and clamped piles make the pooled quantile curve locally flat, the density estimate collapses, the quantile SE goes to ~0, and z explodes. Three changes, design otherwise unchanged:

- integral-valued features (all values whole numbers) leave M4b entirely and move to M4c value counting;
- a quantile level is usable only when the pooled window `q(p-1%) .. q(p+1%)` contains at least 10 distinct values (locally continuous); unusable levels are dropped, and IQR / robust range need both endpoints usable;
- flags now require *excess* that also beats a shuffled-time pass run through the same machinery: `VR > max(1.33, VR_null + 0.33)`, or `max |z| > 4.5` while the null pass stays at or below 4.5.

## M4c: value counting for integer-valued features

Integral-valued features are counting features: the honest discretization is the raw levels, not deciles and not quantiles. Two layers:

1. **Omnibus net.** MI(raw levels; 7x24 time cell) against the same 10-shuffle null as M2/M3, plus Cramer's V, same verdict bands. This also closes a blind spot of M4, which decile-bucketed these features and blurred the piles where their information lives.
2. **Magnifying glass, per value.** The K=10 most and K=10 least common values, selected on pooled share only (time-blind), testable only when the expected count per cell is >= 20 (the binomial z needs it). Per cell the value's share becomes `z = (p_cell - p) / sqrt(p(1-p)/n_cell)`; the 168 z values per value give a variance ratio, calibrated by one shuffled-time pass. Flags identical to the repaired M4b.

Known limits: top/bottom-K watches head and tail while the omnibus MI covers the middle; per-value z's are correlated (shares sum to 1 within a cell), so they are never summed into one chi-square.

## Follow-up: money x time experiment

M4b found the one real time association in the data: a mild `amount_usd` distribution wiggle across the 168 cells (log scale; mean VR 1.82 vs null 0.81, median 2.08 vs 1.28, shifts of roughly 1.2-1.4 sampling errors per cell). The raw time columns are still drop candidates, but this leaves room for one cheap experiment: money x time-bucket features, e.g. the amount's percentile inside its hour-bucket, or the log-amount deviation from the time-cell mean. Expected value is low (fraud is independent of time overall), so this is a single variant to test against the same OOF machinery, not a default feature set.

## Not covered here

- numeric x numeric pockets (left to the model for now);
- train-vs-test time drift (separate quick check before deleting anything globally);
- the circular nature of the hour (0:00 and 23:00 are neighbors; treated as plain labels here).

## Files

- plan: `time_feature_interaction_plan.md` (this file)
- notebook: `time_feature_interactions.ipynb`
- tables and plots: `storage/artifacts/time_feature_interactions/`
- final decision output: `decision_table.csv` in the tables folder
