# Claude's breakdown notes — tokenization & representation

**What this document is.** A breakdown of the *decision space* behind the
tokenization and embedding steps, written as a response to
[tokenization_for_mere_humans.md](tokenization_for_mere_humans.md) and
[representation_for_mere_humans.md](representation_for_mere_humans.md). It is
the layer that should sit *above* any contract or code: the axes of choice, the
couplings between them, and the assumptions each choice silently commits to.

**What this document is not.** Not a set of decisions, not a design, not a
review of code. Where I state a preference it is marked **My lean** and is an
opinion formed before any literature review — treat it as a hypothesis to
confirm or kill, not a conclusion. No implementation.

**Reading order.** Part A is the one reframe that changes how to read
everything else. Part B is the "mental PCA" — the axes themselves. Part C is
the governing question that should weight the axes. Part D points at the
standard literature so the review is fast. Part E is a suggested order for
closing the open axes.

---

## Part A — the reframe: the objective governs the representation

The two docs treat tokenization and embedding as the problem. They are not the
problem; they are *downstream of the pretraining objective*, and right now the
objective is the unwritten variable that most of the open questions resolve to.

- The digit scheme only pays off under an objective that **predicts at the
  field level and reconstructs conditional distributions** — masked modeling
  that learns `P(field | other fields)`, which composes into the generator's
  `P(field | label)`. That is the entire reason this approach fits the
  competition (JOURNEY.md §4).
- **The masking *unit* is itself a choice, not a hyperparameter**, and it is the
  single most load-bearing one (Part B.2). You cannot judge the vocab/embedding
  design in isolation because its payoff is conditional on it.

So: the representation is the bet, and the objective decides what the bet is
*on*. Any contract written before the objective is chosen will encode an
objective by accident — which is roughly what happened in the vibe-coded
attempt.

---

## Part A2 — measurement discipline (DECIDED; it is the backbone)

Introduced by the user: with many choices to make, there is no meaningful
comparison *within* the transformer approach or *against* the TE/GLM/XGB stack
without CV discipline. Correct — and it resolves more cleanly than "4-of-5 folds
everywhere."

**The invariant.** *No held-out label may influence a row's own OOF prediction.
Features are not labels.*

- **5× fine-tuning with honest OOF: mandatory.** The only way to A/B transformer
  variants and to compare the approach against the existing stack.
- **Label-free pretraining does NOT need to exclude the held-out fold's
  features.** It sees X, not y. And Kaggle is **transductive** — test features
  are known at inference and will be in pretraining — so a validation row's
  features being in pretraining *mirrors* the test condition. Excluding them
  estimates the wrong thing (inductive generalization, pessimistic). ⇒ **A single
  pretraining on all train+test features, shared across the 5 fine-tunes, is both
  correct and cheaper.** (Caveat: this estimates transductive test performance,
  which is what Kaggle scores; strict inductive generalization would differ, but
  that is not the estimand here. No train/test shift per JOURNEY.md, so no
  distributional catch.)
- **Label-as-field is the exception where the 5× retraining IS required.** There
  the label lives *inside* the self-supervised objective, so the invariant bites:
  the held-out fold's label must be withheld and never reconstructed → per-fold
  retraining of the self-supervised model. This is the real cost of L, and an
  argument for the two-stage split.
- **Refinement on "only P can use test data" (user's point 2).** Both approaches
  can include test *features* (the test label is simply absent / always-masked).
  The true difference: the label-free stage cannot leak labels by construction,
  so its pretraining is **reusable across folds**; label-as-field cannot reuse,
  because its objective touches labels. So the advantage of P is not "it can see
  test" — it is "clean, cheap, correct CV."

---

## Part B — the decision axes (the mental PCA)

Each axis is written as: **the axis → the options → what it couples to → the
assumption it commits to → My lean.**

### B.1 — Granularity of the numeric representation: scalar vs digit-sequence

- **Options.** (a) one token per numeric feature, a learned projection of the
  scalar (the mainstream tabular-transformer choice); (b) a sequence of digit
  tokens, zero-padded to the column width (your choice).
- **Couples to.** Everything — sequence length, slot embeddings, the objective.
- **Assumption committed.** (a) assumes *magnitude* is the signal; (b) assumes
  *digits at specific resolutions* are the signal.
- **My lean.** (b) is the correct departure here, and it is well-grounded: your
  JOURNEY.md shows the LB moved on fraud rate at digit resolutions (the two
  cents, leading one/two digits, the tens), not on magnitude. The architecture
  follows the evidence. Worth stating this justification explicitly in the doc,
  because it is the load-bearing reason to reject the default. It also names the
  failure mode (B.5): if the model underperforms the digit-TE GLM, it is because
  attention failed to reassemble the multi-digit patterns that TE handed the GLM
  pre-computed.

### B.2 — The pretraining objective and the masking unit **(dominating axis)**

- **Options.** Masked modeling at the *digit-token* level vs at the *whole-field*
  level (mask every digit of `amount_usd` at once). Plus the broader objective
  family (masked reconstruction vs contrastive vs autoregressive).
- **Couples to.** The vocab (field-level masking wants a field-level `[MASK]`
  convention that preserves feature+slot identity so the model knows *which*
  field to reconstruct), the leading-zero problem (B.10), and whether the label
  can enter the sequence (B.8).
- **Assumption committed.** Digit-level masking lets the model predict a masked
  cents digit from the dollars digit of the *same value* — intra-number
  correlation, largely structural and *not* the generator conditional you want.
  Field-level masking forces prediction of a field from the *other fields and
  the label* — which is exactly `P(field | rest)`, the target.
- **DECIDED — field-level masked modeling** (user's point 1). Mask every digit
  of a numeric field at once (and the single token of a categorical field).
  Still the piece to write a third `mere_humans` doc on, because the masking
  *convention* and the label handling (B.8) hang off it.
- **Sub-decision — the mask token (user's point 6: one mask per feature?).** A
  per-feature mask token only carries what the **feature embedding already
  carries**. The real hinge: *do masked positions retain their feature + slot
  embeddings?* If yes (you want this, so the model knows *what* to predict), a
  single generic `[MASK]` per slot + the retained feature/slot embeddings already
  routes "predict `amount`'s digit at the tenths slot" — per-feature mask tokens
  are then redundant params. Field-level masking of a 6-slot number = six
  `[MASK]`s, each keeping its own slot embedding. **My lean:** keep feature+slot
  at masked positions; one generic `[MASK]`, or at most one per input type
  (digit vs categorical). Per-feature masks only earn their keep if feature/slot
  embeddings are *dropped* at masked positions — which there's no reason to do.

### B.3 — Digit semantics: universal vs per-column (the real axis behind "2 vs 3 tables")

- **Reframe.** The "2 tables vs 3 tables" question is a *consequence*, not the
  decision. The actual decision is a hypothesis: **does "9" mean nine everywhere
  (universal), or is each column's digits its own fingerprint alphabet
  (per-column)?**
  - Universal ⇒ shared 0–9 vocab ⇒ *need* a feature embedding to inject column
    context ⇒ "3 tables."
  - Per-column ⇒ column identity baked into the digit id ⇒ no feature embedding
    for numerics ⇒ "2 tables."
- **Key subtlety.** Under *summation* the two are nearly equal in expressivity:
  `token_emb[9] + feat_emb[amount]` is already unique per (digit, column). The
  real difference is **gradient-tying** — shared digits mean learning about "9"
  in `amount` also updates "9" in `account_balance`. So this is the classic
  shared-vs-separate embedding tradeoff: **statistical transfer / fewer params
  (shared) vs flexibility / no wrong prior (separate).**
- **Data regime.** ~1M unlabeled rows for pretraining ⇒ per-column can afford
  the params; but transfer rarely hurts and "digits are the decimal system" is a
  defensible prior for LLM-generated text.
- **My lean.** Start universal + feature embedding (it is the prior your own
  "numbers are real numbers" intuition endorses, and cheaper); keep per-column as
  the named ablation that would falsify the prior. Pick the belief, don't flip a
  coin.

### B.4 — Slot indexing: left-index vs place-value **(latent bug in the shared scheme)**

- **The trap.** If `digit_slot_embedding` is a *single table shared across
  columns* (as the 3-table section reads literally), then slot 0 of `amount_usd`
  (int_slots=4 → thousands) and slot 0 of `account_balance_usd` (int_slots=5 →
  ten-thousands) receive the *same* embedding despite being different powers of
  ten. The model is told the leading padded digit means the same magnitude
  across columns when it does not.
- **Fix.** Index slots by **power of ten relative to the decimal point**
  (… +2 hundreds, +1 tens, 0 ones, −1 tenths, −2 hundredths), not by left-index
  into the padded string. Then "the tens digit" shares one embedding across every
  column — exactly the consistency the universal-digit hypothesis (B.3) wants.
- **Note.** The per-column slot table (your version-2 sketch) sidesteps this
  automatically. So B.3 and B.4 interact: *if* shared digits, slots must be
  place-value; if per-column, the issue dissolves.
- **My lean.** Define slots as place value regardless — it is strictly more
  principled and costs nothing.

### B.5 — Embedding composition: additive separability

- **The assumption.** `final = token + feature + slot` (BERT-style) assumes the
  three factors combine *linearly and separably* at the input.
- **Where it bites.** The signal you care about is often a *multi-digit pattern
  at one resolution* ("leading two digits", "the two cents"). A single-digit
  additive representation hands the model isolated (digit, slot) atoms and asks
  **attention across positions** to reassemble them. The transformer *can* do
  this, but this is precisely where it must earn its keep over the GLM (which got
  the multi-digit slices for free). Not a flaw — just know this is where the work
  moves, and it is the thing to probe if the model disappoints.
- **User's point 5 — replace the sum with `nn.Linear`?** A linear layer over
  *concatenated learnable embeddings* is **mathematically identical** to the
  additive scheme: `W·[e_tok ; e_feat ; e_slot] = W_tok e_tok + W_feat e_feat +
  W_slot e_slot`, and each `W_· E_·` folds back into a reparameterized embedding
  table. Same function class, only more params — it does **not** escape
  additivity.
- **To break separability you need a nonlinearity or a multiplicative
  interaction:** a small **MLP** on the concatenation (Linear → activation →
  Linear), or **gating / FiLM** where the slot (and/or feature) embedding
  produces a scale+shift applied *multiplicatively* to the digit embedding. FiLM
  is the elegant fit because it *is* the stated intuition — "a 0 in the integer
  slot ≠ a 0 in a decimal slot" is the slot **modulating** the digit, which is
  multiplicative. **My lean:** start additive (honest baseline; the transformer's
  first layer already supplies downstream nonlinearity), keep FiLM/MLP fusion as
  a named ablation — not the starting point.

### B.6 — in_between columns: per-column num vs cat

- **The axis.** `in_between_as_num` folds these columns into the shared 0–9
  digit vocab; `in_between_as_cat` gives each distinct value its own id.
- **The strain.** `time_of_day_hour` (cyclical 0–23), `day_of_week` (cyclical
  0–6), `cvv_retry_count` (≈categorical 0–3) and `customer_age` (genuine
  magnitude) share nothing digit-wise. `as_num` ties them to `amount`'s digits
  via the shared vocab — the coupling is most strained exactly where place-value
  is least meaningful.
- **DECIDED (user's point 4): per-column, not a global switch** ("does
  place-value mean anything for this column?"). Magnitude columns (age, card
  age, txn count) → `as_num` is defensible; cyclical/low-card columns (hour, day,
  cvv) → `as_cat` is cleaner and avoids fake place-value. High cardinality is a
  non-problem for the model, so it is not a reason to pick `as_num`.

### B.7 — Permutation structure (a good implicit choice — make it explicit)

- By encoding feature identity via the feature embedding and digit order via the
  slot embedding, you have effectively **removed any global row-level positional
  embedding**. That is correct for tabular data: a row is a *set* of fields, not
  a sequence, and you want permutation-invariance *across* features but order
  *within* a number.
- **Action.** State this as a deliberate decision ("no row-level position
  embedding; feature identity + place-value slot replace it"). The vibe-coded
  version may have bolted on a standard sequence position embedding and
  reintroduced a phantom feature-order dependence — worth checking as a likely
  cause of the loss going the wrong way.

### B.8 — The label as a field

- Because the generator **draws the label first**, `is_fraud` is just another
  field — the root field. If it is a *maskable field* during pretraining (on
  train rows), the pretraining task and the downstream task become the *same
  task*: a clean conceptual unification.
- **Coupling / caution.** Test rows have no label (always-masked there);
  train/test asymmetry must be designed, not left implicit, and leakage rules
  from JOURNEY.md §5 still apply. This is an axis to resolve alongside B.2.

### B.9 — Boolean flags: shared vs per-flag True/False

- Same universal-vs-column structure as B.3, but tiny (6 flags × 2). `used_vpn=
  True` and `is_foreign=True` mean different things; the feature embedding
  disambiguates under sharing, but giving each flag its own id removes the
  ambiguity for negligible cost.
- **My lean.** Per-flag here — the tie breaks toward clarity because it is cheap.

### B.10 — Minor factors (named so they are not forgotten)

- **Discard-extra-decimals.** Fine decision; the dropped slot was constant for
  all but one row (near-zero entropy, pure length tax). Reframe the rationale
  from "discards a fingerprint" to "demotes a one-row fingerprint we could
  recover as a flag if ever needed." Cannot move AUC (1–2 rows, two unlabeled).
- **Leading-zero dominance.** Fixed-width padding makes high slots overwhelmingly
  "0" (constant) — trivial to predict and capacity-wasting under digit-level
  masking (another vote for field-level, B.2). Also means magnitude ≈ count of
  leading zeros, which is useful signal.
- **Missing values / NaN.** Unaddressed. Confirm the data is complete; if any
  column can be missing it needs a dedicated token, and a "missing" field must
  not collide with a "masked" field (B.2).

---

## Part C — the governing goal (REFRAMED per user's point 7)

The stated goal is **learning to use transformers on tabular data and getting a
model that actually trains** — not the competition ranking. That reweights
everything below.

- **Priority 1 — a minimal, correct, instrumented baseline whose training loss
  goes DOWN and whose every component is explainable.** The vibe-coded attempt's
  loss going *up* is the first thing to understand, not to scale past. It is
  almost always one of: LR / scheduler; a **masking bug** (loss scored on
  unmasked positions, or the target value leaking into the input of its own
  prediction); target/prediction **misalignment**; or a **phantom row-position
  embedding** (B.7) reintroducing feature-order dependence. Reproduce a clean
  descending curve on a tiny subset *before* scaling anything.
- **Priority 2 — honest OOF (Part A2)** so your own A/B choices between the axes
  are measured, not guessed. This is where CV discipline pays off for *learning*,
  independent of any ranking.
- **Demoted, still true (the old governing question).** What could the
  transformer buy over the GLM at LB 0.95062? **Field interactions** — joint
  structure where conditional independence fails (hand-found keys like
  `merchant_category × risk_score`), which a pretrained transformer learns
  automatically. Keep this only as a **sanity ceiling**: the discriminative limit
  is near (~0.949, JOURNEY.md §3), so the job is the *interaction residual* above
  independence, not breaking the Bayes error. Do not let this drive the design
  while Priority 1 is unmet.

---

## Part D — map to the standard literature (targets for the review)

Pointers, not a review — names so the search is fast. Verify specifics against
primary sources; these are from memory.

- **FT-Transformer / feature tokenizer** — the mainstream "one token per
  feature, scalar via learned projection" baseline (the B.1 option you are
  departing from). Worth knowing to justify the departure.
- **TabTransformer** — categoricals through a transformer, numerics handled
  separately; the column-embedding idea behind B.7.
- **TabFormer (IBM)** — field/transaction tokenization for tabular/transaction
  data; closest in spirit to your digit-as-token scheme and the row-as-sequence
  framing.
- **"Numbers as tokens" / digit & scientific-notation tokenization** — the NLP
  literature on how models represent numerals; directly relevant to B.3/B.4
  (universal digits, place-value) and to whether digit decomposition carries
  signal.
- **NPT (Non-Parametric Transformers) / attention across rows** — a different
  axis entirely (attending over *other rows*, not just fields) worth knowing
  exists, as it is an alternative way to capture the generator's conditionals.
- **Masked modeling design** — BERT (token-level) vs span/field masking; SpanBERT
  for the span idea behind B.2's field-level masking.
- **Empirical-Bayes / shrinkage target encoding** — the statistical path from
  JOURNEY.md §5, the thing the transformer is competing against; keep it as the
  baseline to beat.

---

## Part E — status and suggested order

**Decided so far.** Part A2 (CV discipline: shared label-free pretraining +
5× fine-tune, honest OOF) · B.1 (digit-sequence) · B.2 (field-level masking,
generic `[MASK]` + retained feature/slot) · B.6 (per-column num/cat).

**Still open, in order:**

1. **B.8 — label as a field.** The next `mere_humans` doc. Resolve the train/test
   label asymmetry and whether the two-stage split (P) is kept over label-as-field
   (L); Part A2 already argues P is the cheaper/cleaner CV. This is the piece
   that is currently load-bearing and unwritten.
2. **B.3 + B.4 together.** Universal-vs-per-column digit hypothesis and the slot
   definition (place-value, not left-index); decide as a pair. Pick the belief,
   name the ablation.
3. **B.5 fusion.** Confirm additive baseline; FiLM/MLP as a named ablation
   (plain `nn.Linear` is *not* an alternative — it equals addition).
4. **B.9 / B.10** — low stakes; decide quickly and move on.

**But Part C (reframed) sets the actual first move:** before closing axes 2–4,
stand up the **minimal baseline that trains** (Priority 1) under the decided
choices, with honest OOF wiring (Priority 2). A clean descending loss curve on a
tiny subset is the milestone that tells us the contract is sound; only then do the
remaining axes carry measurable consequences worth A/B-ing.
