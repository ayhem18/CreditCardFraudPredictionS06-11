# S6E11 — the journey, and the pivot to statistical inference

A summary of how we got here, written to learn from. The last section maps
the remaining work onto the statistics worth studying, because that is the
real reason to take this road.

---

## 1. The setting: the data and the imbalance

- **Task / metric.** Kaggle Playground Series S6E11, "Predicting Credit Card
  Fraud." Binary target `is_fraud`. Scored by **ROC-AUC** — a *ranking*
  metric: only the order of predicted scores matters, not their calibration.
- **Size.** 699,960 training rows, 299,984 test rows, 24 feature columns
  (+ `id`, `is_fraud`).
- **Columns.** ~13 numerics (amount, balance, hours-since-last-txn, distance,
  velocity_score, merchant_risk_score, customer_age, card_age, cvv_retry,
  txn_count_24h, prior_disputes, hour, weekday), 5 categoricals
  (merchant_category, card_type, auth_method, channel, device_type), 6 binary
  flags (foreign, new_merchant, vpn, ip_mismatch, billing/shipping_mismatch,
  ai_scam).
- **The imbalance.** Fraud rate ≈ **1.7%** (`P(fraud) ≈ 0.0167`). This is the
  single most important statistical fact, because *every* quantity we
  estimate is a proportion of a rare event:
  - The variance of an estimated rate `p̂` in a cell of `n` rows is
    `p(1−p)/n`. With `p ≈ 0.017`, a cell needs hundreds of rows before `p̂`
    is stable — so small cells give wild estimates and **must be shrunk**.
  - AUC being rank-based means we never need calibrated probabilities, only a
    monotone score. That is why log-odds / rate features work without any
    calibration step.
- **No leakage.** We checked: `id` carries no order signal (AUC 0.4985), there
  are **zero** duplicate feature-rows and **zero** exact train↔test overlaps.
  So nothing is won by a cheap data trick; the data is clean synthetic.

---

## 2. The features that actually moved the score

The winning features are all variants of one idea — **replace a value by an
estimate of the fraud rate among rows that share it** — at increasing
resolution:

1. **Anchor rates (`arate`).** Fraud rate at the integer floor of each
   numeric; plus the 0.1-grid rate for velocity/risk; plus per-value rates for
   the small-integer columns (cvv, txn_count, age). This was the champion
   `anchor_v2` → tuned XGB **LB 0.94858**.
2. **Per-value + digit target encoding (`te1`).** The breakthrough. ~60 keys:
   the **exact value** of all 24 columns, and the **digit keys** of the 6
   decimal columns — the two cents, the integer part, the tens, the one-decimal
   value, and the leading one/two digits. Each key is replaced by its
   out-of-fold fraud rate. A shallow XGB on these: **OOF 0.94918 single**,
   ~+0.001 over everything hand-built.
3. **Diverse stack.** The same encodings read by different model families:
   XGB (`te1`), a **GLM** (logistic regression on the encodings-as-logits,
   OOF 0.94945 — best single, most decorrelated), and **RealMLP**
   (OOF 0.94861 but only ~0.90 correlated — a genuinely different reader).
   Rank-mean blend `te1 + glm` → **LB 0.95062**, up from 0.94858.

The through-line: the score comes from *estimating a conditional fraud rate
well* (at the right resolution, shrunk, leakage-free), not from any clever
transform of the raw measurements.

---

## 3. Why ordinary feature-engineering and modeling hit a ceiling

Before the pivot we spent a long time on "sensible" features — within-profile
z-scores, percentiles, conditional outlier scores — and each moved the score
by ~0.0001. We did not just give up; we **proved** the ceiling with a
similarity analysis, and that proof is the useful part.

- **The mistake analysis.** Take the champion's *buried frauds* — the 1,941
  frauds it scores below p = 0.02. For each, look at the clean rows in its
  neighborhood (matching on 6 of 7 categorical "atoms"). The buried frauds sit
  **inside the clean crowd**: in the joint quantile space, a fraud's nearest
  clean neighbor is about as close as one clean is to another (nearest-neighbor
  ratio ≈ **1.05**). They are not separable from cleans on the observed
  features — the Bayes error here is real and large.
- **The battery screen.** We then tested, *on exactly those buried frauds*,
  whether any within-profile placement feature (signed-z, |z|, percentile,
  deviation) ranks them above their clean confusers better than the raw value.
  Every one was **equal-or-worse** than raw, and the outlierness `|z|` was
  *anti*-correlated (AUC < 0.5): the buried frauds are **typical** inside their
  cells, not extreme. No placement transform can recover a point that sits at
  the center of the clean mass.
- **But the small win was real.** The z-score and rarity features *did* help —
  `riskz` +0.00013, `cvv` +0.00011 OOF. The reconciliation: those features
  catch the **minority** of frauds that *are* conditional outliers (a high risk
  score for an otherwise-ordinary profile). That subset is small, so the gain
  is small (~0.0001), but it is genuine — the model separates the *extreme*
  frauds in their local neighborhood a little better. The bulk of the misses
  are the typical-looking ones, which nothing local can fix.

Conclusion we trust: **on the genuine features, the discriminative signal is
nearly exhausted (~0.949), because the hard frauds are irreducible overlap.**
That is the honest, production-relevant answer, and it is what forced the
reframe.

---

## 4. The real problem: reverse-engineering the generator

The data is synthetic. The generator **draws the label first, then samples
each field from a distribution conditioned on that label** (a language model at
temperature 0.7). That changes the question entirely:

> It is no longer "can we separate fraud from clean?" (we can't, well). It is
> "**can we recover the conditional distributions `P(field | label)` the
> generator sampled from?**" — and that is a statistical *inference* problem.

Why target encoding is the natural estimator, stated cleanly:

- For one field, the Bayes-optimal evidence is the **likelihood ratio**
  `Λ(x) = P(x | fraud) / P(x | clean)`.
- The fraud rate of a value, `p̂(x) = P(fraud | x)`, is a Monte-Carlo estimate
  of that ratio times the prior:
  `logit P(fraud | x) = log Λ(x) + logit P(fraud)`.
- Summing those logits across fields is a **naive-Bayes** posterior — valid
  when the fields are conditionally independent given the label. A linear model
  on the encodings-as-logits (our GLM) *is* this decoder.
- **Joint keys** are exactly the places where that conditional independence
  fails (e.g. merchant_category × risk_score). You find them with an
  independence test and add an interaction term only where the test rejects.

So the "fraud rate of a digit" is not a magic feature — it is a (blunt)
histogram estimate of the generator's conditional density, and the whole
pipeline is a naive-Bayes decoder of the generator. Seen this way, every
competition lever is a statistics question, which is the point.

---

## 5. The statistics to learn here (why this road is worth it)

Each competition move below is a concrete, testable instance of a core idea.
A week is enough to make real progress on these, and they transfer directly to
the next synthetic competition.

| competition lever | the statistics underneath | what to study |
|---|---|---|
| fraud rate of a value/cell | **point estimation** of a proportion; bias–variance of `p̂`; its variance `p(1−p)/n` blows up for rare `p` and small `n` | estimator properties, standard error, the delta method |
| smoothing toward the prior (our `smooth=20`) | **shrinkage / empirical Bayes**: a Beta(α,β) prior on the rate gives the posterior mean `(s+α)/(n+α+β)`; James–Stein shows shrinkage dominates the raw mean | conjugate priors, Beta-Binomial, empirical-Bayes estimation of α,β |
| `P(x\|fraud)` vs `P(x\|clean)` | **density estimation** and **density-ratio estimation** (the likelihood ratio is the sufficient statistic) | KDE, histogram vs kernel bias, the Neyman–Pearson lemma |
| stacking per-field logits (the GLM) | **naive Bayes** and **conditional independence**; why log-odds are additive under independence | the naive-Bayes derivation, log-odds algebra |
| joint keys (category × risk, etc.) | **hypothesis testing for independence**: the G-test is a **likelihood-ratio test**, `G = 2 Σ O log(O/E)`, asymptotically χ² | LRT, χ² / G-test, degrees of freedom, permutation tests |
| out-of-fold encoding, nested inner folds | **leakage as estimation bias**: using a row's own label makes `p̂` optimistically biased; CV is the fix | cross-validation theory, optimism/overfitting of in-sample estimates |
| the 1.7% imbalance everywhere | **estimation under rare events**: effective sample size, variance inflation, why AUC (rank) sidesteps calibration | rare-event inference, ROC/AUC as `P(score_fraud > score_clean)` |

**The one subtlety to internalize now (it is the crux of leakage):** for a
(train, validation) split, every statistic used to encode a row must be
computed from rows *other than that row's evaluation set*. Validation rows are
encoded from the training part only; and even training rows should be encoded
out-of-fold (a nested split) so the model never trains on an encoding that saw
its own label — otherwise the encoded feature is more informative in training
than it will be at validation, and the model mis-weights it. This matches your
own principle ("statistics on the train, not the full train"); the nested inner
fold is the extra step that extends it to the training rows themselves. Using
the *full* training set to encode the **test** rows is fine — the test set is
never the set the statistics are scored against.

---

## 6. Where we stand

- **LB 0.95062** (`stack1` = TE-XGB + GLM), up from the pre-pivot 0.94858.
- Pipelines in `experiments/target_encoding/` (`build_te_model.py`,
  `build_te_stack.py`, `build_realmlp.py`); ledger in `EXPERIMENTS.md`.
- **Next, the statistical way:** replace the noisy exact-value rate histograms
  with properly shrunk / smoothed estimates of `P(field | label)`, choose
  interaction terms by an independence test rather than by hand, and combine
  everything as an explicit log-odds (naive-Bayes) decoder with tested
  departures from independence.
