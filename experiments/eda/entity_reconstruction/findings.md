# Entity reconstruction — feasibility probe (2026-11-17)

**Question:** can rows be linked into per-card transaction histories
(no `card_id`/`person_id` in the data)? Mechanism proposed: if
`account_balance_usd` is a running balance, a true successor row of row *i*
must satisfy `balance_next ≈ balance_i − amount_i`; block by
(customer_age, card_type), match successors by balance, chain into histories.

**Verdict: negative — the data contains no users to reconstruct.** The
generator drew every row independently from per-column marginals. Three
independent diagnostics agree:

1. **Successor lookup = null.** Match rate of "balance − amount" within
   (customer_age × card_type) blocks equals the shuffled-amount null at every
   tolerance from $0.00 to $5.00 (ratio 0.98–1.03; e.g. 3.27% vs 3.22% null at
   ±1 cent). A running balance would spike above the null at small ε. It does
   not. (`match_rates.csv`)
2. **Id order carries nothing.** Train ids are contiguous 0..699,959 and
   file-sorted, so generation order is preserved — yet adjacent-id chain rate
   is 0.001% (pure collision) and adjacent ids fall in the same age × card_type
   block at chance (0.671% vs 0.668%). Test ids continue at 699,960: one global
   id sequence, split — not shuffled, but never grouped per user either.
3. **Balance behaves like an independent draw.** corr(log balance, log amount)
   = +0.017; P(balance < amount) = 1.07% (impossible for a post-transaction
   running balance); min balance $58.25.

Secondary answers (the "maybe other features are holder-constant" question):
**none are.** Matched successor pairs agree on card_age_months, auth_method,
channel, device_type, merchant_category, distance_from_home_km, … at the same
rate as random same-block pairs (`pair_agreement.csv`). There are also 0
duplicate 24-column feature vectors in 699,960 rows.

Consistent with earlier findings in `time_feature_interactions` (fraud ⊥ time,
merchant mix ⊥ time, age ⊥ hour): the generator is iid per column.

## Implications

- Entity-level features (per-user velocity vs own history, deviation from own
  balance trajectory, sequence position) are **impossible in this dataset** —
  there is no per-user structure, hidden or otherwise. Don't re-test.
- The chaining method itself is sound and would work on data with real running
  balances (e.g. PaySim-style old/new balance pairs) — worth remembering for
  future datasets.

Artifacts: `storage/artifacts/entity_reconstruction/` — `report.txt`,
`match_rates.csv`, `pair_agreement.csv`, `successor_pairs_eps1c.csv`.
Script: `feasibility_probe.py` (runs headless in ~1 min, seed 42).

## Probe 2 — txn_count as a per-user sequence number (2026-11-17)

Second mechanism (user): within (age × card_type) blocks, drop
`txn_count_last_24h == 0` rows, sort by balance (spending-only ⇒ balance
order ≈ time order), check the group holds a full count ladder 1..N, then
link N → N−1 ladder pairs as consecutive transactions and verify
`balance gap = amount + amount`. Script: `count_chain_probe.py`
(`count_chain_report.txt`).

- **Ladder sanity check passes — but exactly at the null level.** 95.625% of
  blocks cover 1..N; the within-block count-shuffle null gives 95.625% and
  0.05 missing levels both ways. With ~800 kept rows/block and max count 11,
  full coverage is automatic. The check is uninformative here, as suspected.
- **Monotonicity dead.** Spearman(balance rank within block, count) = +0.0013
  vs nulls ±0.003 — counts carry no order along balance.
- **Linkage step dead.** 119,142 ladder pairs; P(|gap − (amt+amt)| ≤ 1c) =
  0.003% vs null 0.002%; median residual $1,048.76 ≈ null $1,047.85.
- **Internal contradiction (legible evidence):** P(hours_since < 24 | count
  == 0) = 96.4% — 18,462 rows say "no transaction in the last 24 hours"
  while also saying the last transaction was < 24 h ago. Coherent data would
  have ~0%.
- **The one real column-to-column relationship in the dataset:**
  Spearman(velocity_score, txn_count) = +0.61 (Pearson +0.62). The generator
  built velocity from the count (or both from one latent) — they are near
  duplicates for modeling. Everything else measured is independent.

Final verdict across both probes: rows are iid draws from per-column
marginals; no hidden users, no entity structure. Closed — do not re-test.
