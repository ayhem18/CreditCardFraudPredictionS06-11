# field_spec — the typed column contract

*Frozen data-contract level. Breakdown axis: B.6. See
[workflow.md](workflow.md).*

## In plain terms

`field_spec` is the single source of truth about the 24 feature columns. For each
column it records **what kind of column it is** and **the facts a tokenizer needs
to encode it**. Everything downstream (`vocabulary`, `inner_table`) reads this and
nothing re-derives column facts on its own — that is how the "no column
knowledge scattered in scripts" property holds.

It consumes the `corrected_table` and produces one frozen spec per column.

## What each column's spec carries

- **role** — one of: `numeric`, `in_between`, `categorical`, `boolean`.
- **routing** (in_between only, B.6) — `as_num` or `as_cat`, decided *per column*
  by "does place-value mean anything here?".
- **measured facts** (role-dependent, computed from the corrected strings):
  - numeric / in_between_as_num: `int_slots`, `decimal_slots` (place-value
    aware, B.4), hence `digit_slots`.
  - categorical: the lexicographic value→id map.
  - in_between_as_cat: the distinct-value→id map.
  - boolean: the per-flag `False`/`True` → local id map (lexicographic: `False=0`,
    `True=1`). Vocabulary assigns each flag its own global pair; this entity
    records the written values so the tokenizer can look them up.

> **Contract note for the coding plan.** Because roles need *different* required
> fields and the contract forbids `None` slots (philosophy rule 7), the clean
> shape is **role-specific entities** (`NumericColumnSpec`,
> `CategoricalColumnSpec`, `BooleanColumnSpec`, and in_between routed to one of
> the first two) rather than one entity with nullable fields. The collection is
> the frozen set of column specs; identity = `corrected_table` hash + the
> declaration signature. The exact entity taxonomy is finalized in the coding
> plan, not here.

## The strategy axis (oracle)

What varies is **how roles and routing are assigned**. Measured facts are not a
strategy — they are a deterministic read of the corrected strings.

### `declared` (the only strategy for now)

Roles and per-column routing are **given explicitly** in `args` — no inference,
no magic, full visibility. This is the honest default: the human states the
contract; the process verifies it against the data (every declared categorical
value actually appears, widths match, etc.).

A second strategy (e.g. `inferred`) is promoted only if it earns its place
(philosophy rule 10). Not now.

## The current routing choice (decided)

All 8 in_between columns are **`as_cat`**. Cardinality is the model's problem,
not the tokenizer's: a transformer vocab of a few thousand in-between values is
cheap next to a 50k NLP vocab, and place-value is the wrong prior for several of
these fields (cyclical hour/dow, tiny `cvv_retry_count`). Routing is per-column
in the declaration, so a later `as_num` on a subset is a new `field_spec` hash,
not a rewrite of the entity types.

`as_num` is parsed as a legal routing string and then rejected
(`NotImplementedError`) until it earns a strategy.

The entity taxonomy (coding-plan note above) is: `NumericColumnSpec`,
`CategoricalColumnSpec` (also used for in_between `as_cat`),
`BooleanColumnSpec`. The collection is the frozen set of those specs; identity =
`corrected_table` hash + the declaration signature.

## Deferred

- Promoting `as_num` for a subset of in-between columns, if reconstruction
  shows the categorical treatment is the bottleneck.

Links: [[claude-breakdown-notes B.6]], consumed by
[vocabulary.md](vocabulary.md) and [inner_table.md](inner_table.md).
