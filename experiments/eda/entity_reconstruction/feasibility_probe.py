"""Feasibility probe: can rows be chained into per-card transaction histories?

Idea (user, 2026-11-17): account_balance_usd may be a running balance. If so,
a true successor row of row i (same card holder, no deposit in between)
satisfies

    balance_next ~= balance_i - amount_i

Rows are blocked by (customer_age, card_type) — features suspected constant
per holder — and successor candidates are looked up by near-exact balance
match. Observed match rates are compared against a shuffled-amount null: if
balance is drawn independently per row, observed ~= null and linkage is
impossible.

Also probed here:
  * id structure (are ids contiguous? does file order preserve generation
    order? do adjacent ids already chain users?)
  * duplicate feature rows
  * which features agree between matched successor pairs -> candidate
    holder-constant features, compared against random same-block pairs.

Run:
    ./.venv/Scripts/python.exe experiments/eda/entity_reconstruction/feasibility_probe.py

Artifacts: storage/artifacts/entity_reconstruction/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "storage" / "artifacts" / "entity_reconstruction"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42)

TEST_COLUMNS = [
    "id", "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "merchant_risk_score",
    "prior_disputes", "merchant_category", "card_type", "auth_method",
    "channel", "device_type", "is_foreign_transaction", "is_new_merchant",
    "used_vpn", "ip_country_mismatch", "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
]

CAT_PAIR_COLS = [
    "card_age_months", "auth_method", "channel", "device_type",
    "merchant_category", "time_of_day_hour", "day_of_week", "txn_count_last_24h",
    "cvv_retry_count", "prior_disputes",
]
CONT_PAIR_COLS = [
    "distance_from_home_km", "hours_since_last_txn", "velocity_score",
    "merchant_risk_score", "amount_usd",
]

EPS_CENTS = [0, 1, 2, 5, 10, 25, 50, 100, 500]
SCALE = 10**8  # composite key multiplier: block_code * SCALE + balance_cents
N_SAMPLE = 20_000
N_NULL = 3

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def pct(x: float) -> str:
    return f"{100 * x:.3f}%"


train = pd.read_csv(ROOT / "storage" / "raw_data" / "train.csv")
target_cols = [c for c in train.columns if c not in TEST_COLUMNS]
say(f"train shape: {train.shape}; extra cols beyond test schema: {target_cols}")
if len(target_cols) == 1:
    t = train[target_cols[0]]
    say(f"target '{target_cols[0]}': positives {int(t.sum()):,} / {len(t):,} = {pct(t.mean())}")

n = len(train)
bal = train["account_balance_usd"].to_numpy(dtype=np.float64)
amt = train["amount_usd"].to_numpy(dtype=np.float64)
bal_c = np.round(bal * 100).astype(np.int64)
amt_c = np.round(amt * 100).astype(np.int64)
bal_offset = int(-bal_c.min())
say(f"balance usd: min {bal.min():.2f} max {bal.max():.2f}  |  amount: min {amt.min():.2f} max {amt.max():.2f}")
say(f"P(balance < amount) = {pct(np.mean(bal < amt))}   corr(log balance, log amount) = "
    f"{np.corrcoef(np.log1p(bal), np.log1p(amt))[0, 1]:+.4f}")

# ---------------------------------------------------------------- id structure
ids = train["id"].to_numpy()
id_min, id_max = int(ids.min()), int(ids.max())
contiguous = (id_max - id_min + 1) == n and len(np.unique(ids)) == n
file_sorted = bool(np.all(np.diff(ids) >= 0))
say(f"\n[id structure] min {id_min:,} max {id_max:,} contiguous={contiguous} file_sorted={file_sorted}")
adj_chain_rate = None
if contiguous:
    order = np.argsort(ids) if not file_sorted else np.arange(n)
    b_adj = bal_c[order]
    a_adj = amt_c[order]
    gaps = np.abs(b_adj[1:] - b_adj[:-1] + a_adj[:-1])
    adj_chain_rate = float(np.mean(gaps <= 1))
    say(f"[adjacent-id chain] P(|bal_next - bal_cur + amt_cur| <= 1 cent) = {pct(adj_chain_rate)}")
    blk_adj = None  # filled later once blocks exist

# ------------------------------------------------------------------- blocking
age = train["customer_age"].to_numpy()
ctype = train["card_type"].astype(str).to_numpy()
block_codes = pd.factorize(pd.Series(age).astype(str).str.cat(ctype, sep="|"))[0].astype(np.int64)
n_blocks = int(block_codes.max()) + 1
sizes = np.bincount(block_codes)
say(f"\n[blocking] {n_blocks} blocks (customer_age x card_type); sizes min {sizes.min()} "
    f"median {int(np.median(sizes))} max {sizes.max()}; "
    f"P(two random rows same block) = {pct(np.sum(sizes.astype(np.float64) ** 2) / n ** 2)}")

comp = np.sort(block_codes * SCALE + (bal_c + bal_offset))
order_inv = np.argsort(block_codes * SCALE + (bal_c + bal_offset))  # position -> original row

sample = RNG.choice(n, size=N_SAMPLE, replace=False)


def same_block_match_rate(target_c: np.ndarray, rows: np.ndarray, eps: int) -> float:
    q_lo = block_codes[rows] * SCALE + (target_c + bal_offset - eps)
    q_hi = block_codes[rows] * SCALE + (target_c + bal_offset + eps)
    lo = np.searchsorted(comp, q_lo, side="left")
    hi = np.searchsorted(comp, q_hi, side="right")
    return float(np.mean(hi > lo))


def global_match_rate(target_c: np.ndarray, eps: int) -> float:
    bal_sorted = np.sort(bal_c + bal_offset)
    lo = np.searchsorted(bal_sorted, target_c + bal_offset - eps, side="left")
    hi = np.searchsorted(bal_sorted, target_c + bal_offset + eps, side="right")
    return float(np.mean(hi > lo))


def match_row_counts(target_c: np.ndarray, rows: np.ndarray, eps: int) -> np.ndarray:
    q_lo = block_codes[rows] * SCALE + (target_c + bal_offset - eps)
    q_hi = block_codes[rows] * SCALE + (target_c + bal_offset + eps)
    lo = np.searchsorted(comp, q_lo, side="left")
    hi = np.searchsorted(comp, q_hi, side="right")
    return hi - lo


say("\n[successor lookup] target = balance - amount; P(at least one same-block candidate)")
say(f"{'eps':>8} {'observed':>10} {'null(3 perms)':>14} {'ratio':>7} {'global(no block)':>17}")
rate_table = []
for eps in EPS_CENTS:
    obs = same_block_match_rate(bal_c[sample] - amt_c[sample], sample, eps)
    null_rates = [
        same_block_match_rate(bal_c[sample] - amt_c[RNG.permutation(n)][sample], sample, eps)
        for _ in range(N_NULL)
    ]
    glb = global_match_rate(bal_c[sample] - amt_c[sample], eps)
    null_mean = float(np.mean(null_rates))
    ratio = obs / null_mean if null_mean > 0 else float("inf")
    say(f"{eps:>8} {pct(obs):>10} {pct(null_mean):>14} {ratio:>7.2f} {pct(glb):>17}")
    rate_table.append({"eps_cents": eps, "observed": obs, "null": null_mean,
                       "ratio": ratio, "global_no_block": glb})

dep1 = same_block_match_rate(bal_c[sample] + amt_c[sample], sample, 1)
say(f"[deposit direction] P(same-block candidate at balance + amount, eps=1c) = {pct(dep1)}")

multi = match_row_counts(bal_c[sample] - amt_c[sample], sample, 1)
hit = multi[multi > 0]
if len(hit):
    say(f"[branching] among sampled rows with >=1 candidate at eps=1c: "
        f"mean candidates {hit.mean():.2f}, median {int(np.median(hit))}, "
        f"P(>=2 candidates) = {pct(np.mean(hit >= 2))}")

# ------------------------------------------------- matched-pair agreement stats
target1 = bal_c[sample] - amt_c[sample]
q_lo = block_codes[sample] * SCALE + (target1 + bal_offset - 1)
q_hi = block_codes[sample] * SCALE + (target1 + bal_offset + 1)
lo = np.searchsorted(comp, q_lo, side="left")
hi = np.searchsorted(comp, q_hi, side="right")
has = hi > lo
i_rows = sample[has]
j_rows = order_inv[lo[has]]
j_rows = np.where(j_rows == i_rows, order_inv[np.minimum(lo[has] + 1, n - 1)], j_rows)

pairs = pd.DataFrame({
    "i_row": i_rows,
    "j_row": j_rows,
    "balance_i": bal[i_rows],
    "balance_j": bal[j_rows],
    "amount_i": amt[i_rows],
    "residual_cents": bal_c[j_rows] - (bal_c[i_rows] - amt_c[i_rows]),
})
pairs.to_csv(OUT / "successor_pairs_eps1c.csv", index=False)

# random same-block partner for baseline agreement
block_rows: dict[int, np.ndarray] = {}
sort_by_block = np.argsort(block_codes, kind="stable")
bs = block_codes[sort_by_block]
starts_b = np.searchsorted(bs, np.arange(n_blocks), side="left")
ends_b = np.searchsorted(bs, np.arange(n_blocks), side="right")
partner = np.empty(N_SAMPLE, dtype=np.int64)
for k, r in enumerate(sample):
    b = block_codes[r]
    seg = sort_by_block[starts_b[b]:ends_b[b]]
    partner[k] = seg[RNG.integers(len(seg))]

say("\n[pair agreement] matched successor pairs (eps=1c) vs random same-block pairs")
say(f"{'feature':>26} {'matched':>10} {'random':>10}")
agree_rows = []
for col in CAT_PAIR_COLS:
    v = train[col].astype(str).to_numpy()
    m_agree = float(np.mean(v[i_rows] == v[j_rows]))
    r_agree = float(np.mean(v[sample] == v[partner]))
    say(f"{col:>26} {pct(m_agree):>10} {pct(r_agree):>10}")
    agree_rows.append({"feature": col, "matched_agree": m_agree, "random_agree": r_agree})
for col in CONT_PAIR_COLS:
    v = train[col].to_numpy(dtype=np.float64)
    m_abs = float(np.median(np.abs(v[i_rows] - v[j_rows])))
    r_abs = float(np.median(np.abs(v[sample] - v[partner])))
    say(f"{col + ' |median abs diff':>26} {m_abs:>10.3f} {r_abs:>10.3f}")
    agree_rows.append({"feature": col + "_medabs", "matched_agree": m_abs, "random_agree": r_abs})
pd.DataFrame(agree_rows).to_csv(OUT / "pair_agreement.csv", index=False)

# ---------------------------------------------------------------- dup feature rows
feat_cols = [c for c in TEST_COLUMNS if c != "id"]
dup_rows = n - len(train.drop_duplicates(subset=feat_cols))
say(f"\n[duplicates] rows sharing a full feature vector (24 cols): {dup_rows:,} ({pct(dup_rows / n)})")

# adjacency inside blocks (needs blocks defined above)
if contiguous:
    same_blk_adj = float(np.mean(block_codes[order][1:] == block_codes[order][:-1]))
    exp_same = float(np.sum(sizes.astype(np.float64) ** 2) / n ** 2)
    say(f"[adjacent-id block] P(adjacent ids same block) = {pct(same_blk_adj)} vs chance {pct(exp_same)}")

pd.DataFrame(rate_table).to_csv(OUT / "match_rates.csv", index=False)
(OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
say(f"\nartifacts -> {OUT}")
