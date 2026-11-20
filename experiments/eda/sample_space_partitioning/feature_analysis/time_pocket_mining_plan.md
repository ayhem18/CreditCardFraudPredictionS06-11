# No-alarm time-pocket mining — plan

Hypothesis: even if fraud is independent of the raw hour label, a **contiguous circular window** of the clock (for example 22:00–04:00) can still change what a numeric looks like inside a categorical slice. That would change what “unusual” means (a $200 utilities purchase at night) without showing up as a 24-hour MI.

This is not a redo of M3/M4 in `time_feature_interaction_plan.md`. Those tests asked whether the 24 hour labels (or the 168 hour×day cells) associate with X globally. This pass asks a different, more readable question: inside one category, is there a wrap-around window that shifts the numeric’s scale or mix enough to change the unusualness definition?

Outcome is one of:

- **drop**: P0 looks flat and nothing survives the shuffle gate;
- **generator curiosity, still drop raw time**: real mix/scale pockets exist, but the unusual tail is not more fraudulent than the same cutoff outside the window;
- **distill candidate**: a pocket survives the shuffle gate and its unusual tail (or highlighted age band) is more fraudulent inside the window than the same cutoff outside it. The named follow-up (not this pass) is the percentile of that numeric within `(window × cat)`.

No training. No CMI / AUC-heterogeneity battery. `day_of_week` and the alarm region are parked.

## Scope

- **Rows:** no-alarm train only, same `alarm_flag_mask` as `time_feature_interactions.ipynb` (~646,938 rows, 6,931 frauds).
- **Dropped:** the five alarm flags (`REGION_ALWAYS_DROP`); they are constant here.
- **Clock:** `time_of_day_hour` only, treated as circular (hour 0 neighbors 23).
- **Categorical slices:** one column at a time, plus a no-slice baseline (all no-alarm rows):
  - `merchant_category`
  - `auth_method`
  - `device_type`
  - `is_new_merchant`
  - `card_type`
  - `channel`
- **Numerics:** every baseline numeric except the two time columns. Featured in plots: `amount_usd` (unusualness) and `customer_age` (mix). The rest appear in the ranked table; only keepers get pictures.

## Two story types

**Mix** — who is transacting. Window × Crypto Exchange → share of `customer_age` in [18, 25] versus the rest of that merchant’s day.

**Unusualness** — what counts as weird. Window × Utilities → amount p90 is $25 versus $80 in the rest of Utilities, so $200 is a pocket outlier even if it is not globally extreme.

Each keeper row is written as a sentence of that form, with `n_in` and `n_out` attached.

## Time grain (pre-registered)

Windows: every circular interval of length **4, 6, 8** hours starting at each of the 24 hours (**72** windows). Start `s`, length `L` covers hours `{s, s+1, …, s+L−1} mod 24`, labeled `[s, s+L mod 24)` with the end exclusive. Example: start 22, length 6 is `[22, 04)` (10pm–4am).

Named windows are drawn on P0 only; they are not a separate search:

- night `[22, 04)` (length 6)
- morning `[06, 10)` (length 4)
- business `[09, 17)` (length 8)
- evening `[18, 22)` (length 4)

Occupancy floor, both sides, **inside the slice**: `n_in >= 2000` and `n_out >= 2000`. A window that fails the floor is skipped, not tested.

## How we decide (fixed before looking)

Overlapping windows will always produce a “best” window. The only statistical gate is a **clock shuffle**, the same idea as repaired M4b:

1. For each `(slice, numeric, readout)`, take the most extreme of the 72 windows on the real clock.
2. Scramble `time_of_day_hour` once (numeric values and categoricals stay put).
3. Run the same 72-window search on the scrambled clock. Take that scramble’s most extreme window for the same `(slice, numeric, readout)`.
4. Keep the real story only if its extremity beats `1.1 ×` the scramble’s max.

Readouts:

- unusualness: `|log(p90_in / p90_out)|`, defined only when both p90 are strictly positive;
- mix (only `customer_age`): `|young_share_in − young_share_out|` where young is age in `[18, 25]`.

Headline columns of every table are counts, quantiles, and shares — not p-values. Optional Mann-Whitney (numeric in vs out) and two-proportion / 2×2 chi-square (fraud overlay) sit as footnotes on keepers.

P4 fraud overlay runs **only on keepers**. Unusualness:

- pocket = slice ∩ window;
- cut = pocket p90 of the numeric;
- compare fraud rate of `numeric > cut` vs `numeric <= cut` inside the pocket;
- apply the **same cut** to the complement of the window in that slice;
- require at least 10 frauds above the cut inside the pocket, otherwise `too_thin`.

Mix keepers get the analogous check on the `[18, 25]` band inside the pocket versus the rest of the pocket (and the same band’s rate outside the window, as a reading aid).

A keeper is a **distill candidate** only if the unusual side is more fraudulent *inside the window* than the same cutoff (or band) outside it, and more fraudulent than the non-unusual side of the pocket. Pre-register the “more” as a 1.1× rate ratio on both comparisons.

## Modules

**P0 — pictures before any search.** Heatmaps: hour (0–23) × `merchant_category` of amount p50, amount p90 (USD, not z-scores), and mean `customer_age`. Companion heatmap: each merchant’s hourly p90 divided by that merchant’s overall p90, so a night dip is visible across different ticket sizes. Line plots: 24-hour amount p90 and mean age per merchant, with the four named windows shaded. Mix bars: age-band shares `[18,25] / [26,40] / [41,60] / [61+]` by hour for Crypto Exchange, plus any merchant whose hourly mean-age range exceeds 2 years or whose max/min hourly p90 ratio exceeds 1.15.

If these plots are visually flat, P1 is unlikely to find a trustworthy pocket — that is itself the answer.

**P1 — systematic window search.** For every (slice, numeric, window) that clears the occupancy floor, compare inside the window vs the rest of that slice: `n_in`, `n_out`, p50 / p90 / p95, p90 ratio, and for `customer_age` the `[18, 25]` share on both sides. Output: `window_search.csv`. Amount stays in USD; log1p is only the ECDF x-axis in P3.

**P2 — shuffle gate.** One scramble of the hour labels. `keepers.csv`: at most one row per `(slice, numeric, readout)`, only if it beats `1.1 ×` the scramble max. Each row carries the English sentence.

**P3 — gallery of keepers.** ECDF of the numeric inside vs outside the window (same slice), p50/p90 marked, sentence and counts in the title. Cap at the top 20 keepers by `observed_extremity / null_max`. Amount ECDFs use log1p on the axis, tick labels in USD.

**P4 — fraud overlay.** `fraud_overlay.csv`, one row per keeper. Verdict per row: `too_thin` / `distill_candidate` / `curiosity`.

**P5 — English takeaway.** One of the three outcomes above. If distill: name the candidate feature (percentile of that numeric within the kept window × cat). No OOF run here.

## Files

- plan: `time_pocket_mining_plan.md` (this file)
- notebook: `time_pocket_mining.ipynb`
- tables and plots: `storage/artifacts/time_pocket_mining/`
- decision output: `takeaway.txt` plus `keepers.csv` / `fraud_overlay.csv` in the tables folder
