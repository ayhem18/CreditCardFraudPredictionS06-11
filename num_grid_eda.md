# Numerical grid EDA — initial questions

Written 2026-11-21, before any analysis is run. This document contains
questions, not answers. Each question follows the agreed form: context →
question → what the data would look like under each possible answer →
how to check. No feature is designed until the questions it rests on are
answered.

Established facts carried in (from `jitter_skeleton_check.py` and the
direction document):

- Every continuous column sits on an exact decimal grid (0.01 steps;
  velocity and risk on 0.1 steps).
- The decimals are not uniform noise, and they are not independent of
  the anchor (per-anchor decimal distributions differ).
- Within some anchors, fraud and clean rows differ in decimal position.
  The specific counts from the old battery ("~25 significant anchors")
  are **provisional**: that test assumed uniform-variance noise, which
  the battery itself refuted. Everything below re-derives structure
  without that assumption.

Two standing guardrails for everything below:

- **Noise before structure.** Anchor fraud counts are small (dozens to
  a few hundred). Any "special anchor" claim must be measured against
  binomial noise around a baseline — ideally with shrinkage or a
  hierarchical treatment — before it is called structure.
- **Stability before features.** Whatever pattern is found must hold
  across CV folds (and, where labels don't exist, its feature
  distribution must match between train and test) before anything is
  engineered from it.

---

## Q1. Is there a global value → fraud structure? (the baseline)

**Context.** Fraud rate as a function of value. This is the oldest
question in the book, and the model already profits from whatever smooth
trend exists (the raw float is a feature; trees absorb monotone and
piecewise-smooth effects). The purpose here is not to find a new feature
but to **establish the smooth baseline** that Q2's test of "specialness"
must deviate from.

**Question.** For each numeric column, how does fraud rate vary with the
anchor value — and is that variation a *trend*?

**Possible answers and their data shapes.**

- *Monotone trend* (your "larger anchors ⇒ larger fraud probability"):
  fraud rate rises (or falls) steadily with value; a binned rate-vs-value
  plot with binomial confidence bands looks like a staircase that an
  isotonic fit explains well.
- *Flat*: no relation; all bins compatible with the global rate.
- *Smooth but non-monotone*: rate rises and falls (one or more humps) —
  smooth, but not a trend. Important middle case: it is still "the
  generator throwing fraud points with smoothly varying intensity,"
  just not monotone.
- *Rough*: binned rates jump between neighboring bins by more than
  binomial noise allows — that is Q2's territory, not a trend.

**How to check.** Per column: fraud rate per anchor (or value-bin) with
binomial confidence intervals; fit monotone (isotonic) and smooth
(lowess / splines) baselines; record how much variation each explains.
Read the curve by eye — one plot per column, before any test statistic.

---

## Q2. Are frauds thrown indiscriminately, or are certain anchors special?

**Context.** This is the central question, and it is about the
generator's *mechanism*, not about any single feature. Your distinction,
restated as two candidate worlds:

- **World A — smooth intensity.** The generator draws fraud points with
  a fraud *intensity* that varies smoothly over the value axis. Large
  anchors (or value regions) get more fraud simply because the intensity
  is higher around them; no individual value is special. An observer
  standing on any anchor would see a fraud rate predictable from its
  neighbors.
- **World B — special values.** Some specific values carry elevated (or
  reduced) fraud probability beyond anything their neighborhood
  explains — the generator treats them individually (a fraud-enriched
  value list, per-value parameters, or value-conditional labeling).

The distinction matters because World B is an **inter-row relation**: a
row's risk depends on which *other rows* share its exact value — the
mechanism our (still unexamined) rate features accidentally exploited.

**Question.** After removing the best smooth baseline from Q1, do
individual anchors deviate from their predicted rate by more than
binomial noise?

**Possible answers and their data shapes.**

- *World A holds*: per-anchor deviations from the smooth fit are
  binomial-sized; a histogram of standardized deviations looks standard
  normal; no anchor stands out repeatedly across folds.
- *World B holds*: deviations are overdispersed (spread wider than
  binomial); a stable set of anchors deviates in every fold; deviations
  do not shrink when neighbors are pooled (a special anchor's excess
  fraud does not bleed into adjacent values).
- *Mixed*: most anchors explained by smooth intensity; a handful of
  outliers. The interesting sub-question: are the outliers concentrated
  (e.g., only large values, only round values, only values popular in
  some profile)?

**How to check.** Fit the Q1 baseline; compute per-anchor standardized
residuals; test overdispersion (variance ratio vs binomial; permutation
of labels within value-neighborhoods as the robust alternative); list
the deviating anchors and check their fold-stability and their
characteristics (value size, popularity, column). Deliverable: a table
of "special" anchors per column with honest uncertainty, replacing the
provisional old count.

---

## Q3. Within a value's neighborhood, does grid position matter?

**Context.** Zoom below the anchor: the decimal positions (slots) inside
one anchor's neighborhood [anchor, anchor+1). We know slots are not
uniformly filled and that their fill pattern varies by anchor. Unknown:
whether the slot pattern relates to fraud in its own right — and if so,
whether the pattern is positional (distance to the anchor), specific
(certain slots spike), or only a shadow of Q2 (the whole anchor is
special, slots inside it indifferent).

**Question.** Conditioned on an anchor, is a row's fraud risk related to
its slot — its distance to the anchor, its absolute decimal value, or
membership in specific slots?

**Possible answers and their data shapes.**

- *Indifferent slots*: within an anchor, fraud rows' slot distribution
  matches clean rows' slot distribution. The decimals' lumpiness is
  then generator plumbing (how it fills values) with no fraud meaning.
- *Distance pattern*: fraud concentrates near the integer (slots close
  to .00) or near the middle/far end — a position-within-neighborhood
  gradient, consistent across anchors.
- *Specific slots*: certain decimals (.00? .50? particular cents?) are
  fraud-enriched or depleted, either globally or within particular
  anchors.
- *Shadow of Q2*: slot differences appear only inside the anchors Q2
  flags as special.

**How to check.** Within anchors that have enough fraud (say ≥ 50):
compare fraud vs clean slot distributions directly (per-anchor chi-square
on slot counts — no uniformity assumption, empirical reference; KS or
permutation as robustness). Pool across anchors with a hierarchical
model to see whether any slot effect is common. Compare the slot
profiles of special vs ordinary anchors from Q2 to test the "shadow"
case explicitly.

---

## Q4. Is whatever we find stable? (the gate before any feature work)

**Context.** Every pattern above is estimated from finite fraud counts;
each must survive resampling before it is real to us.

**Question.** Do Q1's baseline shape, Q2's deviating anchors, and Q3's
slot effects reproduce across independent splits of the data?

**Data shapes.** Stable: same conclusions in each CV fold; feature
distributions (e.g., value rates) on test rows match train folds.
Unstable: the deviating-anchor list churns between folds — then the
"special anchors" were mostly noise, and any feature built on them was
fitting noise (which the fold-safe training would have partially
masked, making flat model results uninterpretable).

**How to check.** Recompute Q1–Q3 per fold; measure overlap of the
special-anchor sets; compare rate-feature distributions train vs test.

---

## What each outcome would mean (no features designed yet)

- Q1 smooth + Q2 World A + Q3 indifferent → the value axis carries only
  smooth risk; the raw floats nearly exhaust it; the rate features'
  +0.0013 would need re-examination (it would contradict this world —
  which is itself informative).
- Q2 World B confirmed → value-level rates are a justified, principled
  feature family (and the right tokenizer treatment for the transformer
  direction), with the special-anchor table telling us exactly where
  the signal lives.
- Q3 patterned slots → decimal-aware encodings return with a precise
  target, instead of the blind `slot_z` / `slot_lda` attempts.
