"""Second probe: txn_count_last_24h as a per-user sequence number.

Vision (user, 2026-11-17): within (customer_age, card_type) blocks, drop rows
with txn_count_last_24h == 0, sort by balance (a spending-only account's
balance decreases over time, so balance order should equal time order within
one user), then:
  1. sanity check: with N = max(txn_count) in the block, does the block
     contain every count level 1..N (a full "ladder")?
  2. linkage: rows adjacent in balance order whose counts descend by exactly 1
     (N above N-1) are hypothesized consecutive transactions of one user;
     their balance gap should equal the sum of both amounts.

Nulls: counts permuted within block (ladder coverage, count-vs-balance-order
monotonicity), amounts permuted globally (balance-gap check). Plus two
near-free checks for ANY per-user process: txn_count vs hours_since_last_txn
hard-edge violations, and velocity_score vs txn_count dependence.

Run:
    ./.venv/Scripts/python.exe experiments/eda/entity_reconstruction/count_chain_probe.py

Artifacts: storage/artifacts/entity_reconstruction/count_chain_report.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "storage" / "artifacts" / "entity_reconstruction"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(43)

lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    lines.append(text)


def pct(x: float) -> str:
    return f"{100 * x:.3f}%"


train = pd.read_csv(ROOT / "storage" / "raw_data" / "train.csv")
n = len(train)
cnt = train["txn_count_last_24h"].to_numpy(np.int64)
hrs = train["hours_since_last_txn"].to_numpy(np.float64)
vel = train["velocity_score"].to_numpy(np.float64)
bal_c = np.round(train["account_balance_usd"].to_numpy(np.float64) * 100).astype(np.int64)
amt_c = np.round(train["amount_usd"].to_numpy(np.float64) * 100).astype(np.int64)

age = train["customer_age"].to_numpy()
ctype = train["card_type"].astype(str).to_numpy()
block_codes = pd.factorize(pd.Series(age).astype(str).str.cat(ctype, sep="|"))[0].astype(np.int64)
n_blocks = int(block_codes.max()) + 1

# ------------------------------------------------------------ global shape
vals, counts = np.unique(cnt, return_counts=True)
top = min(len(vals), 14)
say(f"txn_count_last_24h distribution (values {vals[0]}..{vals[-1]}):")
for v, c in list(zip(vals, counts))[:top]:
    say(f"  count={v:>3}: {c:>8,} ({pct(c / n)})")
if len(vals) > top:
    say(f"  ... {len(vals) - top} higher values up to {vals[-1]} (max)")
say(f"rows with count == 0 (dropped per the probe design): {int(counts[vals == 0][0]) if (vals == 0).any() else 0:,}")

# ------------------------------------- consistency: count vs hours_since_last_txn
# If hours_since < 24 there WAS a prior transaction inside the window, so the
# count cannot be 0 (self-exclusive convention) or <= 1 (self-inclusive).
say("\n[consistency] txn_count vs hours_since_last_txn hard edges")
say(f"P(hours_since < 24) = {pct(np.mean(hrs < 24))}")
for lo in (0, 1):
    m = cnt <= lo
    ind = np.mean(m) * np.mean(hrs < 24)
    say(f"  P(count <= {lo}) = {pct(np.mean(m))};  P(hours < 24 | count <= {lo}) = "
        f"{pct(np.mean(hrs[m] < 24))}   (independence predicts {pct(ind)})")
m1 = cnt == 0
say(f"  P(hours < 24 | count == 0) = {pct(np.mean(hrs[m1] < 24))}; "
    f"P(hours >= 24 | count == 0) = {pct(np.mean(hrs[m1] >= 24))}")

say("\n[consistency] velocity_score vs transaction frequency")
say(f"  Spearman(velocity, txn_count) = {spearmanr(vel, cnt).statistic:+.4f}")
say(f"  Pearson (velocity, txn_count) = {pearsonr(vel, cnt).statistic:+.4f}")
say(f"  Spearman(txn_count, hours_since) = {spearmanr(cnt, hrs).statistic:+.4f}")

# ------------------------------------------------------------- ladder check
keep = cnt >= 1
kept_idx = np.flatnonzero(keep)
kb = block_codes[kept_idx]
kc = cnt[kept_idx]
kb_order = np.argsort(kb, kind="stable")
kb_sorted = kb[kb_order]
starts = np.searchsorted(kb_sorted, np.arange(n_blocks), side="left")
ends = np.searchsorted(kb_sorted, np.arange(n_blocks), side="right")
kept_sorted_counts = kc[kb_order]


def block_ladder_stats(counts_by_block_sorted: np.ndarray) -> tuple[float, float]:
    """Fraction of blocks whose kept counts cover 1..max, mean missing levels."""
    full = 0
    miss_sum = 0.0
    n_used = 0
    for b in range(n_blocks):
        seg = counts_by_block_sorted[starts[b]:ends[b]]
        if len(seg) == 0:
            continue
        n_used += 1
        n_max = int(seg.max())
        present = np.bincount(seg, minlength=n_max + 1)
        missing = int(np.sum(present[1:n_max + 1] == 0))
        miss_sum += missing
        full += missing == 0
    return full / n_used, miss_sum / n_used


obs_full, obs_missing = block_ladder_stats(kept_sorted_counts)
say(f"\n[ladder sanity check] kept rows per block; N = max kept count in block")
say(f"  blocks with FULL coverage of 1..N: {pct(obs_full)}   mean missing levels: {obs_missing:.2f}")
for p in range(3):
    shuffled = kept_sorted_counts.copy()
    for b in range(n_blocks):
        seg = slice(starts[b], ends[b])
        shuffled[seg] = RNG.permutation(shuffled[seg])
    n_full, n_missing = block_ladder_stats(shuffled)
    say(f"  null (within-block count shuffle) #{p}: full {pct(n_full)}, mean missing {n_missing:.2f}")

# --------------------------------------- monotonicity: count vs balance order
# ascending balance = (hypothesized) time-forward order; a per-user burst
# would show counts descending along it. Spearman close to 0 under iid.
order = np.lexsort((bal_c, block_codes))  # ascending balance, then block groups rows
sorted_blocks = block_codes[order]
starts_f = np.searchsorted(sorted_blocks, np.arange(n_blocks), side="left")
ends_f = np.searchsorted(sorted_blocks, np.arange(n_blocks), side="right")
r_bal = np.empty(n, dtype=np.float64)
for b in range(n_blocks):
    seg = slice(starts_f[b], ends_f[b])
    r_bal[order[seg]] = np.arange(ends_f[b] - starts_f[b], dtype=np.float64)

rho_obs = spearmanr(r_bal[keep], cnt[keep]).statistic
say(f"\n[monotonicity] Spearman(balance rank within block, txn_count) = {rho_obs:+.4f}")
for p in range(3):
    pc = cnt.copy()
    for b in range(n_blocks):
        seg = block_codes == b
        pc[seg] = RNG.permutation(pc[seg])
    say(f"  null (within-block count shuffle) #{p}: {spearmanr(r_bal[keep], pc[keep]).statistic:+.4f}")

# ------------------------------------------------------- linkage step pairs
# adjacent (ascending balance, same block) kept rows whose counts descend by
# exactly 1 -> hypothesized consecutive transactions of one user. Running
# balance predicts: bal_higher = bal_lower + amt_lower_row + amt_higher_row.
lo_pos, hi_pos = np.flatnonzero(keep)[kb_order][:-1], np.flatnonzero(keep)[kb_order][1:]
same = kb_sorted[:-1] == kb_sorted[1:]
ladder = kept_sorted_counts[:-1] == kept_sorted_counts[1:] + 1
pair_lo = lo_pos[ladder & same]
pair_hi = hi_pos[ladder & same]
gap = bal_c[pair_hi] - bal_c[pair_lo]
pred = amt_c[pair_lo] + amt_c[pair_hi]
err = np.abs(gap - pred)
say(f"\n[linkage step] adjacent-balance ladder pairs (count N above N-1): {len(pair_lo):,}")
for eps in (0, 1, 5, 100):
    say(f"  P(|balance gap - (amt + amt)| <= {eps}c) = {pct(np.mean(err <= eps))}")
for p in range(3):
    ap = amt_c[RNG.permutation(n)]
    err_n = np.abs(gap - (ap[pair_lo] + ap[pair_hi]))
    say(f"  null #{p} (amounts permuted): eps=1c {pct(np.mean(err_n <= 1))}, eps=100c {pct(np.mean(err_n <= 100))}")
say(f"  median |gap - predicted| = ${np.median(err) / 100:,.2f}   (null median "
    f"${np.median(np.abs(gap - (amt_c[RNG.permutation(n)][pair_lo] + amt_c[RNG.permutation(n)][pair_hi]))) / 100:,.2f})")

(OUT / "count_chain_report.txt").write_text("\n".join(lines), encoding="utf-8")
say("\nreport -> storage/artifacts/entity_reconstruction/count_chain_report.txt")
