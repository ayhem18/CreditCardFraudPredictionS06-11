# Profile analysis — catalog and procedure

One list of profiles. For each cell, three measurements on the same rows:

1. **Label census.** How many of the 6,931 no-alarm frauds sit here, at what rate, what share.
2. **Numeric vs parent.** Does $X$ in this cell sit in a different part of the parent’s distribution? (rank-space shift → possible `pq_` grain.)
3. **Numeric × label inside the cell.** Given this profile value, does $X$ separate fraud from non-fraud? Quantified as the **distribution of pairwise quantile differences** (below). This is the measurement that answers “a computer far from home should be suspicious.”

No training. `merchant_tier` is not used. `day_of_week` is parked. There is no `credit_score`; that slot is `velocity_score` and `merchant_risk_score`.

Scope: **no-alarm train** only (~646,938 rows, **6,931 frauds**, ~1.07% rate). About 60% of all train frauds (11,689). Alarm flags are constant here and are not columns.

## Feature types

From [`features.py`](../../../baselines/features.py). In-between columns are **profile axes** when windowed, **numeric outcomes** when unbinned. One role per combination.

**Categorical:** `merchant_category` (12), `is_new_merchant` (2), `auth_method` (5), `device_type` (8), `card_type` (5), `channel` (5).

**In-between:** `time_of_day_hour` (circular `0..23`), `customer_age` (linear integers on the observed range).

**Numerical outcomes:** featured `amount_usd`, `velocity_score`, `merchant_risk_score`, **`distance_from_home_km`**, **`customer_age`** (when age is not a profile axis). Also measured: `hours_since_last_txn`, `card_age_months`, `account_balance_usd`. Parked: `txn_count_last_24h`, `cvv_retry_count`, `prior_disputes`.

### In-between windows (systematic, not lazy)

**Hour.** Circular. Lengths `{4, 6, 8}`, every start in `0..23` → 72. Label `[s, (s+L) mod 24)` end exclusive. Complement = the other hours.

**Age.** Linear. Lengths `{8, 12, 16}` years, every integer start on the observed range. Label `[s, s+L)` end exclusive. Complement = the other ages inside the parent slice.

Search uses the overlapping family. Distill keeps at most one window per `(parent slice, numeric)`.

## Profile catalog

Each profile has a **parent** (nested reference for measurement 2). `pq_` candidates are vs parent, not vs global.

**P1 — one categorical.** Each of the six cats, plus the no-slice baseline. Parent: all no-alarm. Numerics: featured list including unbinned `customer_age` and `distance_from_home_km`. Numeric-vs-parent is descriptive only (tree already has the cat).

**P2 — `merchant_category × is_new_merchant`.** Parent: `merchant_category`. Numerics: `amount_usd`, `velocity_score`, `merchant_risk_score`.

**P3 — `merchant_category × auth_method`.** Parent: `merchant_category`. Numerics: `amount_usd`, `merchant_risk_score`.

**P4 — `device_type ×` each other categorical.** Parent: `device_type`. Numerics: `amount_usd`, `velocity_score`, **`distance_from_home_km`**, **`customer_age`**.

- × `merchant_category`, `is_new_merchant`, `auth_method`, `card_type`, `channel`

**P5 — `channel ×` each other categorical.** Parent: `channel`. Same numerics as P4. `channel × device_type` is computed once with P4.

**P6 — `device_type | channel` as one categorical.** Join; theoretical 40, observed support is the alphabet (expected ~25; do not impute missing pairs). Parent: `device_type`, and separately `channel`. Same numerics as P4. This is the running example: e.g. computer × a channel, then `distance_from_home_km` vs the label.

**P7 — `merchant_category × hour_window`.** Parent: that merchant’s other hours. Numerics: `amount_usd`, `velocity_score`.

**P8 — `merchant_category × age_window`.** Parent: that merchant’s other ages. Numerics: `amount_usd`, `velocity_score`.

**P9 — `device_type × age_window`.** Parent: `device_type`. Numerics: **`distance_from_home_km`**, `amount_usd`. This is the second example: computer **and** older, then distance vs the label. Age is a profile axis here, not the outcome.

Queued: three-way `merchant × is_new × (hour|age) window` only if both 2-ways hit; `device_type × age_window × channel` only if P6 and P9 both hit for distance; alarm region.

## Measurement 1 — label census

Every observed cell of P1–P9. Empty theoretical cells are absent. No occupancy skip.

- `n`, `n_fraud`, `rate`, `lift` = rate / region rate
- `share_frauds` = `n_fraud / 6931`, `share_rows` = `n / 646938`

Tags: **bulky_cold** if `n_fraud >= 100` and `rate <=` region rate; **hot** if `lift >= 2` and `n_fraud >= 20`; else **other**. Do not loosen after seeing the table. Bulky_cold cells are plotted first; they are not a filter on measurements 2–3.

## Measurement 3 — numeric × label inside the cell (materialized $d_{ij}$)

Motive: the no-alarm crowd is expected to be **extremely compact** — a fraud often has a near-double among non-frauds in the same cell. Aggregates (AUC, mean $d$) can look like “$X$ does nothing” while a handful of frauds sit in a tail, or the opposite: AUC slightly up while every fraud still has a non-fraud within $0.01$ quantile. The object is the **pair matrix**, used as a lens on compactness, not a thing we immediately reduce to one number.

### Shared quantile map

All rows in the cell. $F$ = empirical CDF of $X$ on those rows (ties → mid-rank). $q = F(X) \in [0,1]$ for every row. Fraud and non-fraud use this **same** $F$. Class-specific CDFs would make both $q$ samples uniform and every $d$ noise.

### The matrix

$$d_{ij} = q^1_i - q^0_j \qquad i=1..m \text{ (fraud)},\quad j=1..n \text{ (non-fraud)}.$$

$d_{ij} > 0$: fraud $i$ ranks higher on $X$ than non-fraud $j$ inside this cell.

**Materialize.** Persist $q^1$, $q^0$, and the row ids. $d$ is exactly the outer difference $q^1\mathbf{1}^\top - \mathbf{1}(q^0)^\top$, so those two vectors **are** the full signed matrix (lossless, tiny). Also write an explicit `d_ij` array (`float32`) when $m\cdot n \le 2\cdot 10^7$ (~80 MB) so a cell can be opened and sliced. Above that cap, do not allocate $m\times n$ in RAM; compute per-row order statistics from the two vectors (same numbers). Never drop the pair-level view in favour of convolution-only summaries.

In **one** numeric, signed $d$ is rank-1. Then:

- $\min_j d_{ij} = q^1_i - \max_j q^0_j$ (same $j$ for every fraud: the highest non-fraud).
- $\max_j d_{ij} = q^1_i - \min_j q^0_j$.
- “Average of the 5 smallest **signed** $d_{ij}$ per fraud” is $q^1_i$ minus the mean of the 5 largest $q^0$ — a shift of the fraud’s own quantile, not 5 different neighbours.

Those signed extrema are still worth reporting (is **any** fraud above **all** non-frauds? $\max_i \min_j d_{ij} > 0$). They do not measure compactness. Compactness uses **absolute** gaps and **joint** $q$.

### Compactness lenses (why the pairs exist)

$k=5$ unless noted. Occupancy: `n_fraud >= 20` and `n_nonfraud >= 500`.

**Per fraud $i$ (saved as a table, one row per fraud per cell per $X$):**

- $\min_j d_{ij}$ and $\max_j d_{ij}$ (signed; the “other way around” on the same row is $-\max$ / $-\min$, i.e. non-fraud minus fraud).
- mean of the 5 smallest **$|d_{ij}|$** — 5 nearest non-frauds on this quantile line. Near $0$ = this fraud is glued to the crowd.
- mean of the 5 largest **$|d_{ij}|$** — far crowd, for contrast.
- count of non-frauds with $|d_{ij}| \le 0.01$ and $\le 0.05$ (how many doubles at two tightness levels).

**Per cell (from those rows):**

- median over frauds of (mean 5-NN $|d|$). **Compactness score.** Expected small.
- share of frauds with at least one non-fraud in $|d|\le 0.01$ (almost-doubles).
- share of frauds with $\min_j d_{ij} > 0$ (fraud above every non-fraud — a separated tail, the opposite of compact).
- global $\min_{i,j} d_{ij}$ and $\max_{i,j} d_{ij}$.
- mean $d$ $= 2\mathrm{AUC}-1$. Floor $|\mathrm{mean}(d)| \ge 0.10$ still flags “$X$ ranks the label”; compactness can be high even when this fires.

**Joint $q$ (the computer + old + far stack, and the real matrix).** For P4/P5/P6/P9, build a vector of within-cell quantiles for (`distance_from_home_km`, `customer_age`, `amount_usd`) — drop a coordinate if it is a profile axis in that combination. Distance in the unit cube: $d^{\mathrm{joint}}_{ij} = \lVert q^1_i - q^0_j \rVert_2$. This matrix is **not** rank-1. Materialize under the same pair cap. Per fraud: 5-NN mean joint distance, 5-largest mean, counts within Euclidean $0.05$ / $0.10$. This is the 1-cell version of the old rarity/difficulty idea: how many non-fraud doubles in the numeric subspace.

**Pictures:** histogram of 5-NN $|d|$ across frauds; scatter $q$ vs 5-NN $|d|$ (tail frauds that are also isolated vs tail frauds that still have a double). Required for every P6 cell on `distance_from_home_km` and on the joint vector; also bulky_cold cells.

**Worked questions**

- P6 computer × channel, $X=$ `distance_from_home_km`: mean $d > 0$ is “far is riskier”; median 5-NN $|d|\approx 0$ is “still compact — a far fraud still has a far non-fraud.”
- Same cell, joint $(q_{\mathrm{distance}}, q_{\mathrm{age}})$: does adding age isolate any frauds that distance alone does not?
- P9 computer × old window, $X=$ distance: given both cats, is distance still compact against the label?

## Measurement 2 — numeric vs parent (label-blind)

Does the cell change what $X$ looks like, regardless of fraud?

1. Skip unless `n_cell >= 500` and `n_parent - n_cell >= 500`.
2. $F$ = CDF of $X$ on the **parent** rows (P1: all no-alarm).
3. Gate: KS $D = \sup |F_\text{cell} - F_\text{parent}|$. Hit if $D \ge 0.05$.
4. Companions: $F(\mathrm{p50}_\text{cell}) - F(\mathrm{p50}_\text{parent})$ (same for p90); mean parent-rank.
5. P7–P9: max $D$ must beat $1.1\times$ a scramble of hour or age inside the parent. At most one window per `(parent slice, numeric)`.
6. P2–P6: optional scramble of the extra factor within the parent.

Hits vs parent are `pq_` candidates (fold-safe, `min_group = 100`, fallback to parent then global). P1 vs-parent hits are descriptions. Do not add `profile_id` dummies.

Measurements 2 and 3 do not gate each other. A computer cell can have the same distance law as all computers (no vs-parent hit) and still have frauds in the far-distance tail (measurement 3 hit). That second fact is the suspicious-computer story.

## Positive outcome

- Census table: `(profile, cell)` → n, n_fraud, rate, lift, shares, tag.
- Quantile vectors (always) and explicit `d_ij` (under the pair cap) per `(profile, cell, numeric)` and per joint vector.
- Per-fraud compactness table: min/max signed $d$, mean 5-NN $|d|$, mean 5-far $|d|$, double counts at $0.01$ / $0.05$.
- Per-cell compactness rollup: median 5-NN, share almost-doubles, share fully separated, mean $d$ / AUC.
- Vs-parent table: $D$, $\Delta F$, mean parent-rank, verdict `fe_candidate` / `descriptive_only` / `below_floor` / `too_thin`.
- Pictures listed under measurement 3.
- Observed size of P6.

Expectation to test, not assume: median 5-NN $|d|$ very small in most cells (compact). A useful numeric is one that **breaks** compactness for a non-trivial share of frauds (those frauds’ 5-NN $|d|$ large, or $\min_j d_{ij} > 0$), not merely one whose mean $d$ is $0.05$.

A later OOF (not this document): `pq_` on vs-parent hits, and/or $F_\text{cell}(X)$ on cells where compactness actually breaks. Success is AUC.

## Files

- this contract: `profile_numeric_shift_plan.md`
- implementation (later): notebook in this folder; tables, per-cell `q` / `d_ij`, and plots under `storage/artifacts/profile_numeric_shift/`
- does not grow `time_pocket_mining.ipynb`
