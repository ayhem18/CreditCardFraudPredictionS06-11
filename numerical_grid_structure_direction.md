# The numerical grid structure — hypotheses, evidence, and what we never checked

Written 2026-11-21, after the rate-feature campaign (+0.0018 OOF, LB
0.94690 → 0.94829). Purpose: step back from modeling and lay out what we
actually know about the numeric columns, what we assumed, and — most
importantly — which steps of the scientific chain we skipped. That is
where the remaining performance probably hides.

## How to read this document

Every experiment is described in six slots:

1. **Initial observation** — what we saw in the data.
2. **Initial hypothesis** — the story that would explain it.
3. **Predicted data shape** — how the data would look *if the hypothesis
   were true*. This is the slot we most often skipped, and skipping it is
   why several conclusions feel like leaps.
4. **Evidence** — what was actually measured, for or against.
5. **Feature engineering** — what we built from it.
6. **Modeling results** — what the model said.

A missing slot is marked explicitly. Missing slots are not failures to
hide; they are the open work. "The model did not improve" is recorded as
what it is — one experiment's outcome — never as proof a direction is
dead.

## Vocabulary (plain definitions, used consistently below)

- **Grid**: the set of exact values a column actually takes. Every
  continuous column sits on an exact 2-decimal grid (values are exact
  multiples of 0.01; two columns — velocity, risk — on a coarser 0.1
  grid). "On the grid" means: no value ever falls between grid points.
- **Anchor**: the integer part of a value (floor). The value 62.30 has
  anchor 62. Amount has 1,360 distinct anchors, balance 5,582, hours 53,
  distance 59, velocity 74, risk 100.
- **Slot**: the decimal part, expressed in cents above the anchor
  (0–99). The value 62.30 has slot 30. Velocity and risk only use slots
  that are multiples of 10.
- **Value frequency**: how many rows (train) share an anchor. "Rarity"
  features use log of this count.
- **Value rate**: the fraction of fraud rows among all rows sharing an
  anchor (or an exact value). "Shrunk" rate mixes the anchor's own rate
  with the global rate, weight k=20, so small groups don't produce
  extreme numbers.
- **Fold-safe**: a per-group statistic computed only from the parts of
  training data the model does not validate on, so the number a row
  carries never includes that row's own label.
- **Leaky screen**: a quick label-side check that deliberately uses
  full-train statistics (including the row's own label). It gives an
  optimistic upper bound on a feature's possible value, used only to
  decide what is worth a real experiment — never as final evidence.
- **(Terminology note)** earlier notes used "ID poisoning" for the result
  of experiment 3. Retired: it names an untested mechanism, not a
  measurement. This document says only what was measured.

## Background fact (measured, reproducible)

Pooled train+test (~1.0M rows), checked 2026-11-20
(`experiments/eda/data_analysis/jitter_skeleton_check.py`):

- All six continuous columns are exactly on their grids (max deviation
  from the 0.01 grid: 0 for five columns, 2×10⁻³ for amount — a float
  storage artifact).
- The decimal part is **not uniform noise** (details in experiment 2).
- The decimal part carries **conditional** fraud signal: within specific
  anchors, fraud and clean rows sit at different decimal positions
  (strongest: risk anchor 62, where the decimal part alone separates
  fraud from clean with AUC 0.74 on that anchor's 2,035 rows).

---

## Experiment 1 — "The values live on an exact decimal grid"

1. **Initial observation.** `amount_usd` has 14,800 distinct values, but
   only 1,360 distinct integer parts (you found ~1.2k first). The same
   collapse appears in every continuous column.
2. **Initial hypothesis.** *Not stated at the time.* The hypothesis was
   articulated only after later experiments: the data generator drew
   values from a fixed decimal grid (cents, or tenths for velocity and
   risk) and stored them without measurement noise.
3. **Predicted data shape.** If true: every value is an exact multiple
   of 0.01 (or 0.1); distinct-value counts equal the number of grid
   points actually used; no value ever sits between grid points.
4. **Evidence.** Confirmed: five columns have zero deviation from the
   0.01 grid; velocity and risk sit exactly on a 0.1 grid (only ~11
   possible decimal values each); distinct counts match grid occupancy.
5. **Feature engineering.** None standalone — this fact underpins every
   later per-value feature.
6. **Modeling results.** None standalone.

Status: solid. The grid is real and exact.

---

## Experiment 2 — "The decimal part is noise" (REFUTED — and the cautionary tale)

1. **Initial observation.** The decimal part *looks* harmless: pooled
   distribution approximately uniform, and its pooled AUC against fraud
   is 0.50–0.51 for every column. Your earlier finding that converting
   to integers "works without problems" seemed consistent with this.
2. **Initial hypothesis.** A value = integer anchor + random decimal
   noise, with the noise carrying no information. If true, dropping the
   decimals loses nothing (and re-drawing them would be a free data
   augmentation).
3. **Predicted data shape.** If true: (a) within every anchor, the
   decimal part is uniform over its possible values; (b) within every
   anchor, fraud and clean rows have the same decimal distribution;
   (c) the decimal part is uncorrelated with other columns.
4. **Evidence.**
   - (a) **Refuted.** Per-anchor chi-square tests reject uniformity in
     100 of the 100 most frequent anchors for amount (and similarly for
     every column). Concretely: velocity's ".0" position holds 160k rows
     where ~91k are expected if decimals were uniform — some decimal
     positions are systematically over- or under-filled. The average
     decimal position also drifts from anchor to anchor (by up to ±0.4
     of a full unit), so the lumpiness is not one fixed pattern.
   - (b) **Refuted.** Within anchors, fraud rows' decimal positions
     differ from clean rows': per-anchor mean-difference z-scores reach
     10.3 (risk anchor 62); about 25 anchors show significant shifts
     where ~1 would be expected by chance. Local decimal-only AUCs:
     0.74 (risk anchor 62), 0.68 (anchor 64), 0.80 (amount anchor 59,
     only 34 frauds — weak evidence), 0.55–0.60 on larger amount and
     velocity anchors.
   - (c) **Weakly supported.** Spearman correlations between decimal
     parts and other columns' integer parts are all ≤ 0.07.
5. **Feature engineering.** None — the "drop the decimals" feature was
   never built (the refutation came first, which is the point of doing
   3 before 5).
6. **Modeling results.** None.

Status: refuted. The lesson that shaped this document: the pooled AUC of
0.50 was a *marginal* test; the signal was *conditional* on the anchor,
and marginal tests cannot see conditional structure.

---

## Experiment 3 — Handing the grid to the trees as category IDs

1. **Initial observation.** From experiment 2: specific anchors have
   fraud-vs-clean decimal separation the trees might be missing.
2. **Initial hypothesis.** Half-stated at best: "if the grid structure
   is given explicitly (anchor and decimal as categorical IDs), trees
   can use patterns they cannot carve out of raw floats." Your stated
   prior — IDs are usually bad for tree models — was on record, and the
   probe ran anyway.
3. **Predicted data shape.** **Never defined.** This was a hypothesis
   about model behavior, not about the data, and no data-level
   prediction was written down. (What should have been predicted, in
   hindsight: if the trees were missing the structure, explicit IDs
   should help; if IDs mislead the splitter, they should hurt — and the
   trees' split statistics should show ID splits displacing numeric
   splits.)
4. **Evidence.** None at the data level.
5. **Feature engineering.** `grid_anchor_<col>` (anchor as category) +
   `grid_slot_<col>` (decimal as category); a second variant added
   exact-value IDs for velocity and risk (674 / 959 categories).
6. **Modeling results.** Strongly negative: −0.022 (anchor+slot IDs) and
   −0.028 (with exact-value IDs) vs the 0.94526 control — worse in every
   fold. The earlier write-up attributed this to "IDs poison the
   splitter by tempting it to memorize." That mechanism was **never
   tested** (nobody inspected the trees' actual split usage). What is
   known, strictly: this encoding made this model much worse.

Status: this encoding, with this model, hurts. Mechanism unknown.
Open test: inspect feature usage and gain distribution of the trained
fold models (do anchor-ID splits dominate? do numeric splits shrink?).

---

## Experiment 4 — Decimal "unusualness" (slot_z)

1. **Initial observation.** Decimal distributions differ across anchors
   (experiment 2, point a).
2. **Initial hypothesis.** "Rows whose decimal position is *unusual*
   for their anchor are more likely fraudulent" — an unusualness
   (magnitude) story.
3. **Predicted data shape.** If true: within anchors, fraud rate rises
   with the distance between a row's decimal position and its anchor's
   average decimal position; pooled, |z| correlates with fraud.
4. **Evidence.** **Never checked.** Note a design mismatch discovered
   while writing this document: experiment 2's evidence showed *signed*
   shifts (fraud sits at different positions), which supports a
   *directional* story — the unusualness feature may not have matched
   the evidence it was built from.
5. **Feature engineering.** `slotz_<col>` = (slot − anchor mean) /
   anchor std, computed without labels.
6. **Modeling results.** Flat (−0.00001 vs control).

Status: unresolved. A flat model is not a refuted hypothesis,
especially when the hypothesis's own data-level check (slot 3) was never
run. The cheap open test: plot fraud rate vs decimal z within anchors.

---

## Experiment 5 — Decimal "direction" (slot_lda)

1. **Initial observation.** Within ~25 anchors, fraud and clean rows sit
   at different decimal positions (experiment 2, point b).
2. **Initial hypothesis.** "A row's fraud risk is higher when its
   decimal position is closer to its anchor's typical *fraud* position
   than to its typical *clean* position" — a directional story matching
   the evidence.
3. **Predicted data shape.** If true: in anchors with enough frauds,
   fraud rows satisfy |slot − fraud mean| < |slot − clean mean| more
   often than chance; a single signed distance-difference separates the
   classes locally.
4. **Evidence.** Partial and indirect: the per-anchor mean-shift
   z-scores from experiment 2 are evidence for a *mean shift*, but the
   centroid-distance formulation itself was never validated at the data
   level, and only ~25 of ~350 anchors showed shifts.
5. **Feature engineering.** `slotlda_<col>` = |slot − clean centroid| −
   |slot − fraud centroid|, centroids fold-safe, shrunk k=50.
6. **Modeling results.** Flat (−0.00006 vs control).

Status: unresolved. Three untested explanations: (i) k=50 shrinkage
flattened exactly the ~25 anchors that carried the effect; (ii) the
effect is real but too local to move pooled AUC; (iii) trees already
exploit decimals within anchors through ordinary numeric splits (the
raw float is still a feature). Open test: for the significant anchors,
compare the champion model's per-anchor fraud ranking against the
decimal-only ranking — if the model already ranks them identically,
explanation (iii) holds and the direction is genuinely closed.

---

## Experiment 6 — Value rarity (anchor frequency)

1. **Initial observation.** A cheap screen: the log train-frequency of
   an anchor, as a single feature, has marginal fraud AUC 0.63
   (velocity), 0.63 (risk), 0.56 (hours), 0.55 (amount), 0.51
   (balance), 0.50 (distance).
2. **Initial hypothesis.** "How *often* a value appears carries fraud
   information beyond the value itself" — rare (or common) values are
   fraud-enriched.
3. **Predicted data shape.** If true: fraud rate varies systematically
   with value popularity (monotone in frequency, or at least separable
   by a frequency threshold), and the relation is stable across folds.
4. **Evidence.** Only the marginal screen AUCs. Two basic facts were
   never established: **which direction** (rare vs common → fraud — the
   screen recorded only max(AUC, 1−AUC)), and **what shape** the
   frequency→fraud relation has (a plot was never made).
5. **Feature engineering.** `afreq_<col>` = log(1 + train count of the
   anchor). Target-free; unseen test anchors get count 0.
6. **Modeling results.** +0.0003 vs control (0.94554), positive in all
   five folds. Small but consistent.

Status: weakly positive effect; poorly understood cause.

---

## Experiment 7 — Value fraud-rate (the big win)

1. **Initial observation.** Leaky screen: full-train per-anchor shrunk
   fraud rate as a single feature reaches AUC 0.67 (balance), 0.61
   (amount), 0.57 (hours).
2. **Initial hypothesis.** "Fraud probability depends on the *exact
   value* in ways that go beyond any smooth trend — specific values or
   tight value neighborhoods carry elevated or reduced risk." The deeper
   question — *why* would a generator produce value-specific risk — was
   never articulated (candidates in the open-questions section).
3. **Predicted data shape.** If true: (a) per-anchor fraud rates deviate
   strongly from any smooth curve fitted to rate-vs-value (spiky
   differences between neighboring anchors); (b) the fraud value
   distribution differs from the clean value distribution, column by
   column; (c) rate estimates are stable across folds.
4. **Evidence.** Only the leaky screen. **None** of (a)–(c) was ever
   tested — the screen's number went straight into feature building.
5. **Feature engineering.** `arate_<col>` = (frauds + 20·global rate) /
   (rows + 20) per anchor, fold-safe; test rows use full-train rates.
6. **Modeling results.** **+0.0013 vs control** (0.94654), positive in
   all five folds — the largest single jump of the campaign.

Status: the strongest result we have, sitting on the least
understanding. This asymmetry is the main motivation for this document.

---

## Experiment 8 — Finer grids and small integers (anchor_v2)

1. **Initial observation.** Leaky screens: fraud rate at the native 0.1
   grid of velocity/risk reaches AUC 0.76 / 0.72 (vs 0.63 / 0.63 at
   integer anchors); per-value rates for small-integer columns: cvv
   0.71, customer_age 0.65, txn_count 0.62.
2. **Initial hypothesis.** "Rate specificity matters: velocity/risk
   carry value-level risk at their native 0.1 resolution (their ~700
   rows per value make estimates reliable), and small-integer columns
   carry per-value risk that trees under-use when it must be re-derived
   in every interaction context."
3. **Predicted data shape.** If true: within a single integer anchor of
   velocity/risk, the ten 0.1-values have *different* fraud rates beyond
   binomial noise; small-int columns' per-value rates deviate from the
   global rate beyond noise.
4. **Evidence.** Screens only. The natural direct test — compare the
   spread of rates across 0.1-values inside each integer against
   binomial noise — **was never run**.
5. **Feature engineering.** `arate10_<col>` for velocity/risk (rate per
   exact 0.1 value); `arate_<col>` per value for customer_age,
   cvv_retry_count, txn_count_last_24h.
6. **Modeling results.** +0.0005 over experiment 7 (cumulative +0.0018;
   0.94703, positive in all five folds). Champion configuration; its
   bag later set the competition best (LB 0.94829).

Status: best-performing feature set; the within-integer rate-spread test
remains the cleanest way to confirm its premise.

---

## Experiment 9 — Rate split by alarm region (anchor_v3)

1. **Initial observation.** Leaky screen: rate of (0.1-value × alarm
   flag) for risk reaches AUC 0.83 (vs 0.76 value-only).
2. **Initial hypothesis.** "Value-level risk differs inside vs outside
   the alarm region — the same value means something different
   depending on the alarm context."
3. **Predicted data shape.** If true: within the same value, alarm and
   no-alarm rows show different fraud rates beyond noise.
4. **Evidence.** Screen only; the direct within-value alarm/no-alarm
   rate comparison was never run.
5. **Feature engineering.** `arate10a_<col>` = fold-safe rate per
   (0.1-value, alarm) group.
6. **Modeling results.** −0.0002 vs experiment 8 → not adopted. The
   write-up's explanation — "trees can interact the rate with the alarm
   flag themselves" — is plausible, untested, and should be treated as
   a hypothesis, not a finding.

Status: one negative model result; direction not closed.

---

## The open questions (ranked by how much understanding they unlock)

These are the hypotheses we never articulated. Each comes with the
predicted data shape that was missing the first time around.

**Q1. Why do specific values carry specific fraud rates?** Three
candidate generator stories, with distinguishable predictions:

- *H1 — different value distributions.* The generator drew fraud rows
  from a different value distribution than clean rows (e.g., a partially
  overlapping value list, or shifted concentrations). Predicted shape:
  fraud and clean value histograms differ column by column; some values
  over-represented among fraud by large factors; the difference is
  *distributional* (both rarity and rate features are shadows of it).
  Test: two-sample comparison (e.g., KS) of fraud vs clean values per
  column, plus per-value odds ratios with confidence intervals.
- *H2 — value-conditional labeling.* The generator drew values from one
  distribution, then flipped the label with a value-dependent (but
  smooth-ish) probability. Predicted shape: fraud and clean value
  distributions share the same support and rough shape; rate-vs-value
  curves are smooth, not spiky.
- *H3 — a mixture.* Both mechanisms, or different regimes per column.
  The spiky-vs-smooth shape of rate-vs-value curves is the direct
  discriminator between H1 and H2, and **nobody has looked at those
  curves.**

**Q2. Why are decimal distributions lumpy and anchor-dependent?**
Candidate: the generator mixes "exact price-point" draws with smeared
draws. Predicted shape: within anchors, a few specific decimal positions
(like .00, .50, or particular cents) spike consistently, and the spike
pattern is either global or depends on the anchor's magnitude/type.
Test: decompose per-anchor decimal histograms; check whether the same
slots spike across anchors.

**Q3. Is the within-anchor decimal signal (experiments 4–5) recoverable
with proper statistical treatment?** The ~25 significant anchors were
found with plain z-tests; a hierarchical (partial-pooling) estimate
would say whether the effect is a few strong anchors or a broad weak
gradient — which decides whether any encoding can carry it.

**Q4. What did the trees actually do with the IDs (experiment 3's
mechanism)?** Inspect the trained fold models: split counts and gain by
feature. Predicted shapes: if the memorization story is right, anchor-ID
splits proliferate and gain concentrates there; if something else went
wrong (e.g., unseen-level handling), the pattern will differ.

**Q5. Which direction does rarity point (experiment 6)?** One line,
never recorded: are rare values fraud-enriched or fraud-depleted, per
column? The frequency→fraud curve answers this and Q1 simultaneously.

**Q6. Does the rate signal drift between train and test?** Labels are
unavailable for test, but the *distribution of the rate features* on
test rows vs train folds is checkable. Predicted shape if the generator
is stable: nearly identical feature distributions.

## What we would do differently, stated once

For every experiment above that lacks slots 3–4, the missing step was
the same: write down how the data would look if the story were true,
then look. The two most expensive skips were experiment 3 (no data-level
prediction at all) and experiment 7 (a screen number went straight into
a feature). Both produced large metric outcomes — one very negative, one
very positive — and in both cases we moved on without understanding
why. The metric told us *that*; only the data can tell us *why*.
