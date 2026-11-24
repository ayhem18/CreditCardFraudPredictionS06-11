# The generator-fingerprint direction — in plain terms

This folder is the turning point of the competition. After a long stretch
of feature engineering that moved the score by ~0.0001 at a time, one idea
moved it by **~0.001 in a single model**. This note explains *why*, simply,
so the idea sticks.

Files here:
- `build_te_model.py` — the pipeline (per-value + digit target encoding →
  shallow XGB). Produces `storage/artifacts/training/icU_te1/`.

---

## 1. First, the honest morale note

You spent ~3 hours and the metric barely moved. That felt like failure. It
wasn't — here is what those hours actually bought:

1. **A proven negative.** You didn't just *guess* that conditional features
   were a dead end — you built a screen on the exact frauds the model misses
   and *proved* they sit inside the clean crowd. Most people grind blindly
   for days without ever establishing that. Knowing where the ceiling is, is
   real knowledge.
2. **The machinery.** The fold-safe rate code, the blending harness, the
   clean CV discipline, the Kaggle submit automation — all of it carried
   straight into the win. The TE model plugged into the blend on day one
   because that scaffolding already existed.
3. **The reframe that found the lever.** The breakthrough didn't come from a
   cleverer feature. It came from asking a different question: *how was this
   data actually made?* That question is the whole game in synthetic Kaggle
   competitions, and you now know to ask it first.

And about the "tiny" numbers: on a crowded synthetic leaderboard, 0.9487 →
0.9496 is not tiny — hundreds of submissions are packed into that band, so a
0.001 AUC gain is a large *rank* move. The metric compresses huge effort
into small deltas. That is the texture of the game, not a verdict on you.

---

## 2. The key realisation: how this data was born

This is a **synthetic** dataset. A generator made all 700k training rows.
The competitors at the top reverse-engineered *how*, and it changes
everything:

> The generator writes the **label first** (fraud or not), and **then**
> fills in the fields — amount, risk score, hours, etc. — using a language
> model that has *seen the label*.

Read that twice. It means every field was chosen **knowing whether the row
is fraud**. So the fields are not neutral measurements that we must
interpret — they are **clues the generator left behind**. A particular
amount like `1371.37` is slightly more common in fraud rows than clean rows,
not because $1371.37 is inherently suspicious, but because the generator
*tended to write that value when the label was fraud*.

These traces are called **generator fingerprints**.

---

## 3. Why everything we did before only sipped at the signal

All session we treated the numbers as **meaningful quantities** and asked
"is this risk score high *for its context*?" — z-scores, percentiles,
conditional outliers. We were doing *statistics on the values*.

But the signal isn't in what the value *means*. It's in the **exact value
itself** — the fingerprint. "Is `1371.37` a fraud-ish value?" is a better
question than "is 1371 a high amount?" We were reading the sentence; the
answer was in the handwriting.

That is exactly why our earlier `arate` features half-worked and our grid
IDs *poisoned* the model (§6/§7 of EXPERIMENTS.md): we had stumbled onto the
edge of this — rates of coarse value buckets — and stopped one step short of
doing it on the **exact values and their digits**.

---

## 4. The technique: target encoding, explained simply

**Target encoding** = replace a category by *its fraud rate*. If transactions
with amount exactly `1371.37` are fraud 8% of the time, replace `1371.37`
with `0.08`. Now the model reads the fingerprint directly as a number.

The catch — and it is the whole craft — is **leakage**. If you compute
"amount 1371.37 → 8% fraud" using a row's *own* label, you've cheated: the
feature secretly contains the answer, and it falls apart on the test set. So
the rate for each row is computed **only from other rows** (out-of-fold):

- Split the data into 5 folds.
- To encode the rows in fold 1, use folds 2–5 to compute the rates.
- (And a nested inner split so even the training rows never see themselves.)
- **Smoothing**: a value seen only twice gets pulled toward the overall
  fraud rate, so rare values don't produce wild, overfit rates. We use
  weight 20 — a value needs ~20 rows before its own rate is trusted.

That's it. No row ever sees its own label; every value is replaced by its
honestly-estimated fraud rate.

---

## 5. The keys: the value, and the digits it was written with

We don't encode only the exact value. The generator's fingerprints live at
several resolutions, so we build a *family* of keys from one number. For
`amount = 1371.37`:

| key | value | what it groups together |
|---|---|---|
| exact value | `1371.37` | the sharpest fingerprint |
| cents (two decimals) | `37` | rows whose amount ends in `.37` |
| integer part | `1371` | rows near $1371 |
| tens | `137` | a coarser bucket |
| one decimal | `1371.3` | medium resolution |
| leading 1 / 2 digits + length | `4,1` / `4,13` | "a 4-digit amount starting 13" |

Each key is target-encoded (its own out-of-fold fraud rate). The exact value
is the strongest signal but is rare (noisy); the digit keys are coarser but
seen far more often, so they're stable. Together they cover the fingerprint
at every zoom level. We build these for all 24 columns (exact value) and the
digit keys for the 6 columns that have decimals — about 60 encoded features.

(The very top solutions add two more: **GPT-2 token keys** — the number as
the language model literally tokenised it, so `0.8` and `0.80` differ — and
**joint keys** for pairs of columns whose combined fraud rate isn't just the
sum of the parts, like merchant-category × risk-score. Those are the next
step, worth a little more.)

---

## 6. Why one strong model isn't enough: diverse stacking

Our whole earlier blend was XGBoost variants — 0.99 correlated, so averaging
them barely helped (like asking six people who all read the same newspaper).
The TE model is only 0.97 correlated with the champion, so it *already*
blends better. The top teams go further and stack **different model
families** on the same encodings — a plain logistic regression, a neural net
(RealMLP), even a GPT-2 model — because each makes *different mistakes*.
Averaging models that err differently is where the last ~0.005 lives.

---

## 7. The one-paragraph summary

The data is synthetic; the generator wrote the label first and then the
fields, so the **exact values and their digits are clues to the label**. You
read those clues by replacing each value (and each digit-bucket of it) with
its **out-of-fold fraud rate** — target encoding, computed so no row sees its
own label. A shallow tree on ~60 such encodings scores ~0.949 by itself
(where all our hand-reasoned features sat at ~0.9482), and blending it with a
few *different* model families is how the leaderboard reaches ~0.955. The
lesson that generalises: **for synthetic competitions, understand how the
data was generated before you engineer a single feature.**

---

## 8. Where we are

- `icU_te1` (this model): OOF **0.94918** single.
- `icU_te1_base` (TE + our base, blended): OOF **0.94959** — submitted.
- Next: joint keys + token keys (push the TE model), then a GLM and RealMLP
  member, then stack. Target: the 0.955 band.
