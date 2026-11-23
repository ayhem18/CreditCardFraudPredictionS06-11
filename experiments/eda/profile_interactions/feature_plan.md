# Feature-engineering plan — `intersection_conditional` family

Goal: beat the current champion by giving the model a few **conditional**
numbers it cannot compute for itself. New dataset versions are named with
the **`intersection_conditional_`** prefix (not `anchor_`). Plain-language
plan to read and review before we spend any attempts.

## The one rule we score against

- **Champion to beat:** `xgbU_tuned_av2_d5s` — the `anchor_v2` feature set,
  XGB depth 5, subsample/colsample 0.8, lr 0.05, 400 trees, unified
  `alarm_profile` CV. **OOF 0.94806, LB 0.94858.**
- **OOF is free, LB is the budget.** We can run as many OOF experiments
  locally as we like; we only have **10 LB submissions per day**. So we
  screen everything on OOF first and **only submit a config that clears
  the OOF bar.**
- **OOF bar before a submission:** paired delta **≥ +0.0001** AND up in
  **≥ 3 of 5 folds** (bar lowered from +0.0003 by the user on 11-23:
  keep near-champion models around until one clear champion emerges; the
  current champion itself came from fine-tuning, so these untuned feature
  bases may gain more after tuning). Measured by adding the feature to the
  base and retraining the *same* config with the *same* seed.
- **Near-champion keepers so far:** riskz (0.94832) and cvv (0.94829) both
  clear the +0.0001 bar; the combine (0.94821) does not.
- **Paired, not standalone.** A feature can score 0.71 on its own and add
  nothing, because the trees may already reconstruct it. Only "base +
  feature vs base" counts.

## The base dataset version (build and establish this first)

**`intersection_conditional_base`** = all of `anchor_v2`'s features
(afreq, integer-anchor rates, the 0.1-grid velocity/risk rates, and the
per-value rates for customer_age / cvv_retry_count / txn_count_last_24h)
**PLUS the two categoricals we had dropped — `channel` and `card_type` —
added back.** Only the time columns stay dropped.

- **Why add them back:** the new features condition on channel (and
  device). The model should also hold the raw `channel` and `card_type`
  columns so it can combine the conditional rate with the plain category.
- **Caution:** channel and card_type were *free drops* before (removing
  them cost ~0.0001). So establishing the base is itself attempt 1 — if
  `intersection_conditional_base` differs from the champion by more than
  noise, that is the re-added cats, and we must know that number so any
  later gain is credited to the conditional features, not the raw cats.

Every feature version below is `intersection_conditional_base` + one (or a
combination) of the features.

## What "conditional" means and why it can help

Once the categorical profile of a row is fixed (channel, device, card,
auth, merchant, alarm/new flags), only the numbers decide fraud. The trees
see every raw number and category. What they **cannot** do, at depth 5–6,
is walk to a narrow cell AND then judge how unusual a number is *inside
that cell* — that costs too many splits. So we pre-compute that number.

- **Discrete columns** (cvv_retry_count, prior_disputes, txn_count): the
  meaning of a value changes with context → **conditional rate**.
- **Continuous columns** (merchant_risk_score, velocity_score): a value
  can be an **outlier inside its cell** even when normal overall →
  **conditional outlier score**.

Running example (mistake analysis): cell **Online · Android · Visa · 3D
Secure · not-new · no-alarm** (10,034 rows, 23 frauds). Inside it,
`cvv_retry_count = 1` has fraud rate 0.0090 vs 0.0019 for `cvv = 0`
(4.7×), and risk 55 sits far above the cell's ~34 median though 55 is
ordinary across the whole dataset.

## What the base already contains (from the P0 audit)

This changes what counts as "new". `anchor_v2` already holds **global
per-value fraud rates** for `cvv_retry_count`, `txn_count_last_24h`,
`customer_age` (and the 0.1-grid rates for velocity/risk). So:

- The marginal "cvv=1 is 4.7× cvv=0" lift is **already in the model.**
- Conditioning a rate on the **alarm flag was already tried and rejected**
  (anchor_v2c / the old anchor_v3, both flat). So new conditioning must be
  on **channel × device**, which is untested.
- The 2-key interaction-rate machinery already exists
  (`_interaction_rate_table`), so these features are cheap to build.

---

## Family A — discrete value × context, conditional rate

**What the number is:** the fold-safe, shrunk fraud rate of a discrete
value seen *in this context*.

### `intersection_conditional_cvv` — rate of `cvv × channel × device_type`
- **Means:** "Of transactions with this CVV-retry count, on this channel
  and device, what share were fraud?"
- **Example:** cvv=1 on Online/Android → ~0.009; cvv=1 on POS/iPhone →
  maybe ~0.002. One number carries the difference.
- **New content = the interaction only.** The base already has the global
  cvv rate, so this wins only if the lift genuinely *varies by
  channel×device*. That is the whole bet.

### `intersection_conditional_disputes` — rate of `prior_disputes` (then × context)
- **Means:** "How fraud-prone is this dispute count?" — and then by context.
- **Fully new:** `prior_disputes` has **no** rate in the base at all, so
  even the marginal version is new. Cheapest genuine win on the board —
  try the marginal first, then the context version.

### `intersection_conditional_txn` — rate of `txn_count_last_24h × context`
- Marginal already in the base → **interaction-only**, same status as cvv.

---

## Family B — continuous value, conditional outlier score

**What the number is:** how far a value sits from the *typical* value for
otherwise-identical transactions, in robust units. **Non-monotone** — a
shape the base does not have in any form.

### `intersection_conditional_riskz` — robust-z of risk inside the k=6 cell
- **Formula (plain):** `|risk − cell median| / cell spread`, spread = MAD,
  small cells borrow the broader profile's spread.
- **Means:** "how unusual is this risk vs the normal risk for look-alike
  transactions."
- **Example:** risk 55 in a cell with median 34 → large score, though 55
  is common globally. This targets the frauds we watched the model miss.
- **Why new:** conditional AND non-monotone (`|·|`) → no raw-risk
  threshold reproduces it, and it is not the (flat) percentile idea.
- **Variant `..._riskz_signed`:** `(risk − cell median) / spread`, so the
  tree can also use direction.

### `intersection_conditional_velz` — robust-z of `velocity_score`
- Same recipe for velocity.

---

## Family C — conditional tail-depth (low priority)

### `intersection_conditional_taildepth` — `P(risk ≤ x | cell)` (and velocity)
- **Means:** "98% of look-alike transactions have a lower risk than this."
- **Demoted:** this is a within-cell percentile — **monotone** — which is
  essentially the old `profile_context` (within-profile percentiles,
  2026-11-17) that scored **−0.0010**. Treat as a late probe only, not a
  day-1 lever. Feed per feature as separate columns; **never multiply them
  together** (the `acomb` lesson: the trees assemble the joint, +0.00001).

---

## Family D — the stability knob (sweep on the A/B winners)

Not new features — a setting we sweep on whatever clears the bar:

- **broad:** only the categories the number interacts with (cvv →
  channel+device; risk → merchant_category). More rows, steadier.
- **narrow:** the full k=6 cell (most specific, noisiest).
- **shrunk:** narrow pulled toward broad by cell size.

cvv was flat across widths → start **broad**. risk → test narrow vs broad
vs shrunk and keep the best.

## How everything is kept honest (one paragraph)

Every rate, median and spread is computed **using only the other folds'
rows**, so a row is never scored with a number that saw its own label; test
rows use the full-training value. Small cells are **shrunk toward the
broader profile** so a 10-fraud cell can't manufacture a pattern. Same two
safeguards the winning `anchor_v2` rates already use.

---

## Daily schedule (fits the 10-attempt budget)

Each "attempt" is one OOF experiment. The **LB** column is only spent when
the OOF bar is cleared. Priority reflects the audit: fully-new features
first, interaction-only ones next, the monotone probe last.

### Day 1 — base, then each lever alone
| # | version | status vs base | submit if OOF bar cleared |
|---|---|---|---|
| 1 | `intersection_conditional_base` (re-add channel, card_type) | baseline check | only if it moves |
| 2 | `..._disputes` (marginal prior_disputes rate) | **fully new** | yes |
| 3 | `..._riskz` (risk robust-z, shrunk) | **fully new (non-monotone)** | yes |
| 4 | `..._cvv` (cvv × channel × device) | interaction-only | yes |
| 5 | `..._riskz_signed` | fully new | only if beats riskz |
| 6 | `..._disputes` × context | fully new | yes |
| 7 | `..._velz` (velocity robust-z) | fully new | yes |
| 8 | `..._txn` (txn_count × context) | interaction-only | yes |
| 9 | `..._taildepth` (risk + velocity) | probe (likely flat) | only if bar cleared |

**Day-2 combine result (11-23):** `cvv + riskz` = OOF **0.94821**, +0.00002
vs base, **1/5 folds** — *worse than either single* (riskz 0.94832, cvv
0.94829). The two weak positives **interfere, not stack**: they tap
overlapping hard-fold signal, so together they add noise. Stacking
strategy refuted. Best lever remains **riskz alone at +0.00013** — still
below the +0.0003 submission bar. Consistent with every prior hand-built
feature (profile_context flat; only non-monotone anchor rates ever paid)
and the irreducible-overlap ceiling (buried frauds sit in the clean crowd,
NN ratio ~1.05).

**Fine-cell riskz result (11-23):** riskz within the full 7-atom profile
(drop-one escalation) = OOF **0.94826**, +0.00008 vs base, 3/5 — *below*
the coarse merchant×is_new riskz (+0.00013) and below the +0.0001 bar,
despite a higher standalone AUC (0.7365 vs 0.6942). Reading: the finer
signed-z is closer to raw risk (more redundant) and noisier, so it adds
less. **Coarse riskz stays the best single feature.** Risk conditioning is
caught between redundancy (fine → ≈ raw risk) and signal-loss (coarse-with-
merchant → weak residual); the merchant×is_new rung is the sweet spot and
it is still only +0.00013.

### Day 2 — combine the winners
- Stack the day-1 features that each cleared the bar into one version; test
  the **combined** paired delta (combinations can add or cancel).
- Family D sweep on the best continuous feature: narrow vs broad vs shrunk.
- Drop any feature that doesn't earn its place (drop-one discipline).

### Day 3 — lock and bag
- Freeze the best set as **`intersection_conditional_v1`**.
- Rebuild the 8-member bag (full + 4×depth-6 seeds + 3×depth-5 seeds) on
  it; submit the bag — bagging added +0.0014 last time and the depth-5
  members are the best individuals.

---

## Tracking table (fill in as we go)

| date | attempt | version | OOF | Δ vs 0.94806 | folds up | LB | verdict |
|---|---|---|---|---|---|---|---|
| 11-23 | 1 | intersection_conditional_base | **0.94819** | +0.00013 vs champ | — | | within noise; **features now measured against 0.94819** |
| 11-23 | 2 | ..._disputes (marginal) | 0.94818 | −0.00001 vs base | 2/5 | — | **flat** — monotone rate the trees already hold; rejected |
| 11-23 | 3 | ..._riskz (within merchant×is_new) | 0.94832 | +0.00013 vs base | 3/5 | — | weak positive (non-monotone); below bar, **keep for combine** |
| 11-23 | 4 | ..._cvv (cvv×channel×device) | 0.94829 | +0.00011 vs base | 3/5 | — | weak positive; big on hard folds 2/4, loses easy fold 0; **keep for combine** |
| | 5 | ..._riskz_signed | | | /5 | | |
| | 6 | ..._disputes × context | | | /5 | | |
| | 7 | ..._velz | | | /5 | | |
| | 8 | ..._txn | | | /5 | | |
| | 9 | ..._taildepth | | | /5 | | |

---

## Honest expectation

The frauds the champion misses mostly sit **inside the clean crowd** — in
the running cell every fraud had a clean row within one percentile on every
number (nearest-neighbour ratio ~1.05). Those are irreducible at this
resolution. These features recover the **minority** of missed frauds that
are genuine conditional outliers (the risk-55 type). The marginal cvv and
txn rates are already in the model, so the realistic prize sits in the
**interaction** and the **non-monotone** shapes — another **+0.0005 to
+0.001**, same regime as the whole rate campaign. Best-supported single
bets: `..._riskz` (aimed at the misses we watched) and `..._disputes` (the
one column with no rate yet). `..._cvv` is promising but is an interaction
test, not a fresh signal.
