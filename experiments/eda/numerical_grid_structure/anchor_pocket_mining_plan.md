# Anchor × category interactions — v1

Chain: observation → hypothesis → data → statistical tests → feature
engineering → model. v1 stops at the tests; FE/model only if something
passes. (This replaces the earlier verbose port of the time-pocket plans —
the idea imported from them is one sentence: *explain the variance inside
anchors by profiling categories*.)

## Feature taxonomy (the framing)

Features are categorical, numerical, or **in-between**. In-between
features are analyzed through **bands** — either systematic (every band
of a fixed grid, like the 72 clock windows) or **data-driven** (bands
built from the data itself). The anchor revelation moved
merchant_risk_score and velocity_score into the in-between class: their
value space is a modest set of discrete levels, so they can be banded and
profiled like a categorical, without decile blurring.

## Initial observations

1. Within-anchor cell rates are noisy (Q3 small multiples); the per-anchor
   fraud rates themselves are consistent (Q1).
2. The other score column explains within-anchor variance only through
   its own marginal — the joint field is the product of marginals
   (q1_joint_anchor_structure, p = 0.796). The untested partner axis is
   **categorical**.
3. `anchor_v3` (value × alarm rates) was flat as a model result, but its
   data-level premise was never checked.

## Hypothesis

Some categorical columns change what a merchant_risk anchor means:
P(fraud | anchor, category value) deviates from P(fraud | anchor) beyond
sampling noise, consistently across anchors (a value-level lift, not
one-off cells).

## Data

Full train. merchant_risk_score anchors (integer). Partners: the eleven
baseline categorical columns, one at a time. Rates pooled (descriptive
scan; any feature built later will be fold-safe). Occupancy: cells with
>= 500 rows; rate ratios shown only with >= 10 frauds behind a cell.

## Statistical tests (fixed before looking)

1. **Cell scan**: per (anchor, value) cell, two-proportion z of the cell
   rate against its anchor's pooled rate; count cells with |z| >= 3
   (noise expectation ~0.27% of tested cells).
2. **Within-anchor shuffle**: per partner, permute the category labels
   among each anchor's rows (20 draws), recount |z| >= 3 cells. A partner
   is **meaningful** only if the observed count exceeds every shuffle.
3. **Value-level lift**: for values of meaningful partners, frauds
   observed at that value divided by frauds expected if the value were
   indifferent inside its anchors (the tilt-style ratio, bootstrap CI).
   A lift consistently off 1 across anchors is the interaction made
   visible; the anchor × value heat map shows where it lives.

## Feature engineering (only if tests pass)

Conditional anchor rate — shrunk rate per (anchor, value) of the passing
partner(s), fold-safe, on top of anchor_v2.

## Model

Only if the FE step is justified by the tests.

## v2 (named fallback, not built here)

If no partner passes: group the anchors by their own fraud rate
(data-driven bands) and repeat the scan on (rate-band × category) cells —
bigger samples, and the banding itself is the taxonomy idea applied to
the anchor axis.

## Files

- plan: this file
- notebook: `anchor_category_interactions_v1.ipynb`
- plots: `storage/artifacts/numerical_grid_structure/anchor_category_interactions_v1/`
