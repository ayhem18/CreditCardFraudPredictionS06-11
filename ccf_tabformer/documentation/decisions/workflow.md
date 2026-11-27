# Workflow — how the components connect

**What this folder is.** `decisions/` is the *breakdown → choice* layer that must
exist before any contract or code (see
[claude_breakdown_notes.md](../claude_breakdown_notes.md), and the project's
working rule: breakdown → choice → contract → code). One `.md` per component
explains it in simple terms and lists the strategies it may take; this file is
the map that ties them together. A coding plan comes *after* these are agreed.

**Two conventions this folder obeys.**
- **Data contract** (`dl_toolbox/standards/data_contract`): a frozen data
  *level* is three objects — entity, collection, process — with explicit
  identity and no defaults. Its own philosophy says *experiment tracking (models,
  losses, seeds) is later context, not a level of this contract.*
- **Oracle pattern** (`.../dcm_input_phase`): each configurable step is a package
  with an `oracle` (`strategy: str → class`), a preparation/strategy pair, and
  required, exact-key `args`. A string in a config file selects an
  implementation. Class composition assembles the chosen strategies.

**The experiment-visibility premise.** An experiment is a config file where
every field is required and defaults are avoided. So every choice below appears
in the config as an explicit `{strategy, args}` block. Reading the config tells
you the entire encoding, without reading code.

---

## The dividing line: frozen levels vs runtime modules

The data contract draws the line for us. A step is a **frozen data-contract
level** when its output is a deterministic function of (upstream data +
a signature) and is worth freezing with identity for reuse. A step is a
**runtime module** when it carries learnable parameters or randomness per
epoch — i.e. "experiment tracking," which the contract explicitly excludes.

| # | component | frozen level or runtime module | breakdown axis |
|---|---|---|---|
| 0 | `corrected_table` | frozen level (already exists: `storage/data/corrected`) | — |
| 1 | `field_spec` | frozen level | B.6 |
| 2 | `vocabulary` | frozen level | B.3, B.9 |
| 3 | `inner_table` (tokenized rows) | frozen level | B.1, B.3, B.4 |
| 4 | `embedding` | runtime module | B.3, B.4, B.5 |
| 5 | `masking` | runtime module | B.2, B.8 |
| 6 | `encoder` | runtime module | — |
| 7 | `heads` / `objective` | runtime module | B.2, B.8 |
| 8 | `experiment` / `cv` | orchestration (composes all of the above) | Part A2 |

Each frozen level (1–3) is produced by a process with an oracle, so its strategy
is swappable and its output is reusable across folds and experiments. Each
runtime module (4–7) is selected by an oracle too (strategy string → class), but
is composed inside the model/train step, not frozen as data.

---

## The components, in plain terms

**0. `corrected_table`.** The pooled train+test working copy with the three
one-off extra decimals dropped (`storage/data/corrected`). Upstream; unchanged.

**1. `field_spec` — the typed column contract.** Per column: its role (numeric /
in-between / categorical / boolean), and for in-between columns the *per-column*
num-vs-cat routing (B.6). Plus the measured facts a tokenizer needs: digit-slot
and decimal-slot widths (place-value aware, B.4) and the lexicographic
categorical ids. *Strategy axis:* how roles are assigned (likely one strategy,
`declared`, with the routing in its args; a second strategy is promoted only if
it earns it — rule 10). *Identity:* `corrected_table` hash + declaration
signature.

**2. `vocabulary` — the global token inventory.** Digits, categorical values,
booleans, and the special tokens (the mask token(s)). Its *shape* is the
universal-vs-per-column digit decision (B.3) and shared-vs-per-flag booleans
(B.9). *Strategy axis:* `universal_digits` vs `per_column_digits`. *Identity:*
`field_spec` hash + vocab strategy signature.

**3. `inner_table` — rows as sequences (the "inner table strategy").** Each row
becomes a fixed-length sequence of token ids *plus the aligned structural
streams*: a feature-id stream (which column each position belongs to) and a
slot/place-value stream (B.4), and a real-vs-pad marker. This is the
`(B, S, ·)` substrate the model reads. *Strategy axis:* the layout —
`fixed_width_padded` first. *Identity:* `corrected_table` hash + `vocabulary`
hash + layout signature.

> **Resolved (Q1): two levels.** The *inventory* is `vocabulary` (2); the
> *per-row sequence* is `inner_table` (3), which folds the value→id mapping into
> its construction. No separate `tokenizer` level.

**4. `embedding` — ids → vectors, and how they combine.** Token, feature, and
slot embeddings, and the composition (additive baseline; FiLM/MLP as the
non-additive ablation — plain `nn.Linear` is *not* an alternative, it equals
addition, B.5). Table-sharing is coupled to the `vocabulary` strategy (B.3).
*Strategy axis:* composition = `additive` | `film` | `mlp`. Runtime.

**5. `masking` — what the model must reconstruct.** Field-level masking (B.2):
mask every position of a chosen field at once, keep its feature+slot embeddings,
one generic `[MASK]` per masked position. Produces masked inputs + reconstruction
targets. The label field's handling (B.8) lives here. *Strategy axis:*
`field_level` (+ label policy). Off (or label-only) at finetuning. Runtime,
per-epoch random.

**6. `encoder` — the transformer backbone.** Depth, width, heads, etc. *Strategy
axis:* likely one backbone for now; configured, oracle only if a second appears.
Runtime.

**7. `heads` / `objective` — what the loss reads.** Pretraining: per-field
reconstruction heads over masked positions + their loss. Finetuning: the fraud
head + loss, scored by AUC. *Strategy axis:* head/objective per phase. Runtime.

**8. `experiment` / `cv` — the orchestration that composes everything.** The
top-level config whose every field is required. Encodes Part A2: **one shared
label-free (unsupervised reconstruction) pretraining** over all corrected rows
(train+test features), then **5-fold finetuning with honest OOF**. It names, by
`{strategy, args}`, the choice taken at each component above.

---

## The two runs

**Unsupervised reconstruction (pretraining), once, shared.**
`corrected_table → field_spec → vocabulary → inner_table` (all frozen, reused) →
`embedding → masking(field_level) → encoder → heads(reconstruction)`. Label-free,
so it may use every row including test features (Part A2); the single pretrained
encoder is reused by all 5 folds.

**Finetuning, 5×, honest OOF.** The same frozen `inner_table` → the pretrained
`embedding + encoder` → `heads(fraud)`, fine-tuned on 4 folds' labels, predicting
the 5th. Masking off (or label-only). OOF over the 5 folds is the comparison
currency — against transformer variants and against the TE/GLM/XGB stack.

(The alternative, *supervised reconstruction* / label-as-field, would collapse
runs 1–2 into one but forces per-fold retraining because the label is inside the
objective — Part A2. User is inclined toward unsupervised reconstruction; this
doc assumes it.)

---

## Resolved decomposition

- **Q1 — two levels:** `vocabulary` (inventory) + `inner_table` (per-row
  sequence, value→id folded in). No separate `tokenizer`.
- **Q2 — freeze `inner_table`** as a data-contract level (identity = corrected
  hash + vocabulary hash + layout signature), reused across folds and
  experiments. `field_spec` and `vocabulary` also frozen.
- **Q3 — one doc per component, thin ones grouped.** `corrected_table` folded
  into this workflow; `encoder` + `heads` share one doc.

## The per-step docs

Data side (frozen levels): [field_spec.md](field_spec.md) ·
[vocabulary.md](vocabulary.md) · [inner_table.md](inner_table.md).
Model side (runtime modules): [embedding.md](embedding.md) ·
[masking.md](masking.md) · [encoder_and_heads.md](encoder_and_heads.md).
Orchestration: [experiment_and_cv.md](experiment_and_cv.md).

Each is plain-terms description + strategy menu. The implementation plan that
mirrors this workflow onto the existing `ccf_tabformer_dl/` tree is
[coding_plan.md](coding_plan.md).
