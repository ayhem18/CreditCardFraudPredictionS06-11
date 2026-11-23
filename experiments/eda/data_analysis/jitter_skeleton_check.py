"""Verify the integer-skeleton + uniform-jitter hypothesis on the continuous numerics.

Hypothesis under test (2026-11-20): every continuous numeric column is
``integer_anchor + U(-0.5, +0.5) noise`` rounded to cents, i.e. the
fractional part is exchangeable noise and the integer skeleton is the
generator's native alphabet. Three training decisions rest on it:
de-jittering (drop the fraction), per-anchor value-ID embeddings, and
label-preserving jitter re-draw as augmentation.

Battery, per column, on pooled train+test unless noted:

1. floor view (the user's spec): top-100 most frequent floor(x) anchors;
   is x - floor(x) ~ U(0, 1) inside each anchor (chi-square, 10 bins)?
2. round view (the stricter generator story): top-100 round(x) anchors;
   is x - round(x) ~ U(-0.5, 0.5) inside each anchor?
3. homogeneity across anchors: each anchor's fraction vs the pooled
   fraction (not vs the theoretical uniform).
4. rare anchors: the same floor-view battery on the 100 least frequent
   anchors with n >= 100 (does the noise behave differently in the tail?).
5. noise vs anchor magnitude: Spearman(anchor, |x - round(x)|) - does
   bigger money get bigger noise?
6. noise vs target within anchors (train only): per-anchor fraud-vs-clean
   mean-difference z-scores of x - round(x); under the hypothesis they are
   standard normal.
7. noise vs other columns: Spearman of each column's round-residual
   against the other columns' skeletons and residuals.
8. discreteness of the noise: share of values off the cent grid, number
   of distinct fractional levels.

Usage::

    python experiments/eda/data_analysis/jitter_skeleton_check.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy import stats

from ccf_ml.file_utils import get_test_csv_path, get_train_csv_path

COLUMNS = [
    "amount_usd",
    "account_balance_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "velocity_score",
    "merchant_risk_score",
]
N_ANCHORS = 100
UNIFORM_HALF_WIDTH = 0.5
UNIFORM_STD = np.sqrt(1.0 / 12.0)


def chi2_uniform_p(values: np.ndarray, *, lo: float, hi: float, bins: int = 10) -> float:
    observed, _ = np.histogram(values, bins=bins, range=(lo, hi))
    expected = np.full(bins, len(values) / bins)
    if np.any(expected < 5):
        return float("nan")
    return float(stats.chisquare(observed, expected).pvalue)


def chi2_pooled_p(
    values: np.ndarray, pooled_probs: np.ndarray, *, bins: int = 10
) -> float:
    observed, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    expected = pooled_probs * len(values)
    if np.any(expected < 5):
        return float("nan")
    return float(stats.chisquare(observed, expected).pvalue)


def battery(values: np.ndarray, anchors: np.ndarray, *, label: str) -> dict[str, object]:
    """Chi-square battery of `values` (all in [0, 1)) within `anchors`."""
    frame = pd.DataFrame({"anchor": anchors, "frac": values})
    grouped = frame.groupby("anchor")["frac"]
    counts = grouped.size()
    selected = counts[counts >= 100].sort_values(ascending=False)
    top = selected.head(N_ANCHORS)
    p_unif: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    fracs_by_anchor = {a: grouped.get_group(a).to_numpy() for a in top.index}
    pooled, _ = np.histogram(frame.loc[frame["anchor"].isin(top.index), "frac"], bins=10, range=(0.0, 1.0))
    pooled_probs = pooled / pooled.sum()
    p_pooled: list[float] = []
    for a in top.index:
        v = fracs_by_anchor[a]
        p_unif.append(chi2_uniform_p(v, lo=0.0, hi=1.0))
        p_pooled.append(chi2_pooled_p(v, pooled_probs))
        means.append(float(v.mean()))
        stds.append(float(v.std()))
    p_unif_arr = np.array([p for p in p_unif if not np.isnan(p)])
    p_pooled_arr = np.array([p for p in p_pooled if not np.isnan(p)])
    pvs = stats.kstest(p_unif_arr, "uniform").pvalue if len(p_unif_arr) else float("nan")
    return {
        "label": label,
        "n_anchors": len(top),
        "min_anchor_n": int(top.min()),
        "mean_abs_dev_from_0.5": float(np.max(np.abs(np.array(means) - 0.5))),
        "max_abs_dev_std": float(np.max(np.abs(np.array(stds) - UNIFORM_STD))),
        "p_uniform": {
            "frac_p_below_0.05": float((p_unif_arr < 0.05).mean()),
            "n_p_below_0.05": int((p_unif_arr < 0.05).sum()),
            "min_p": float(p_unif_arr.min()) if len(p_unif_arr) else float("nan"),
            "p_values_uniform_p": float(pvs),
        },
        "p_pooled": {
            "frac_p_below_0.05": float((p_pooled_arr < 0.05).mean()),
            "min_p": float(p_pooled_arr.min()) if len(p_pooled_arr) else float("nan"),
        },
    }


def print_battery(result: dict[str, object]) -> None:
    p_uniform = result["p_uniform"]
    p_pooled = result["p_pooled"]
    print(f"  [{result['label']}] anchors={result['n_anchors']} min_n={result['min_anchor_n']}")
    print(
        f"    max |mean(frac)-0.5| = {result['mean_abs_dev_from_0.5']:.4f} | "
        f"max |std(frac)-{UNIFORM_STD:.4f}| = {result['max_abs_dev_std']:.4f}"
    )
    print(
        f"    vs U(0,1): {p_uniform['n_p_below_0.05']}/{result['n_anchors']} anchors p<0.05 "
        f"(expect ~5) | min p = {p_uniform['min_p']:.4g} | "
        f"KS(p-values ~ U(0,1)) p = {p_uniform['p_values_uniform_p']:.3f}"
    )
    print(
        f"    vs pooled frac: {p_pooled['frac_p_below_0.05']:.2f} of anchors p<0.05 | "
        f"min p = {p_pooled['min_p']:.4g}"
    )


def main() -> None:
    train = pd.read_csv(get_train_csv_path())
    test = pd.read_csv(get_test_csv_path())
    pooled = pd.concat([train.drop(columns=["is_fraud"]), test], ignore_index=True)
    residuals: dict[str, np.ndarray] = {}
    skeletons: dict[str, np.ndarray] = {}

    for column in COLUMNS:
        x = pooled[column].to_numpy(dtype=np.float64)
        floor_anchor = np.floor(x)
        frac_floor = x - floor_anchor
        round_anchor = np.round(x)
        res_round = x - round_anchor
        residuals[column] = res_round
        skeletons[column] = round_anchor

        print(f"\n== {column} ==")
        b_floor = battery(frac_floor, floor_anchor, label="floor view: top-100 frequent")
        print_battery(b_floor)
        anchor_counts = pd.Series(floor_anchor).value_counts()
        rare_anchors = anchor_counts[anchor_counts >= 100].nsmallest(N_ANCHORS)
        rare_set = rare_anchors.index.to_numpy()
        if len(rare_set) >= 20:
            sel = np.isin(floor_anchor, rare_set)
            b_rare = battery(frac_floor[sel], floor_anchor[sel], label=f"floor view: {len(rare_set)} rarest (n>=100)")
            print_battery(b_rare)
        rho_mag, _ = stats.spearmanr(round_anchor, np.abs(res_round))
        print(f"    Spearman(anchor, |resid|) = {rho_mag:+.4f}  (0 => noise size independent of anchor)")

    # noise vs target within anchors (train only)
    print("\n== noise vs target, within anchor (train) ==")
    y = train["is_fraud"].to_numpy(dtype=bool)
    for column in COLUMNS:
        res = residuals[column][: len(train)]
        anc = skeletons[column][: len(train)]
        frame = pd.DataFrame({"anc": anc, "res": res, "y": y})
        z_scores: list[float] = []
        for anchor, grp in frame.groupby("anc"):
            fraud = grp.loc[grp["y"], "res"]
            clean = grp.loc[~grp["y"], "res"]
            if len(fraud) < 20 or len(clean) < 20:
                continue
            se = np.sqrt(UNIFORM_HALF_WIDTH**2 / 3.0 * (1 / len(fraud) + 1 / len(clean)))
            z_scores.append(float((fraud.mean() - clean.mean()) / se))
        z = np.array(z_scores)
        print(
            f"  {column:24s} anchors={len(z):4d} |z|: p50={np.quantile(np.abs(z), 0.5):.2f} "
            f"p95={np.quantile(np.abs(z), 0.95):.2f} max={np.abs(z).max():.2f} "
            f"(n|z|>2.96 = {(np.abs(z) > 2.96).sum()}, expect {0.003 * len(z):.1f})"
        )

    # noise vs other columns
    print("\n== Spearman(residual, other skeletons/residuals) ==")
    res_frame = pd.DataFrame({f"res_{c}": residuals[c] for c in COLUMNS})
    skel_frame = pd.DataFrame({f"skel_{c}": skeletons[c] for c in COLUMNS})
    both = pd.concat([res_frame, skel_frame], axis=1)
    corr = both.corr(method="spearman")
    for column in COLUMNS:
        row = corr.loc[f"res_{column}"].drop(f"res_{column}")
        strongest = row.abs().idxmax()
        print(f"  {column:24s} max |rho| = {row.abs().max():.4f} (vs {strongest})")

    # discreteness of the noise
    print("\n== noise discreteness ==")
    for column in COLUMNS:
        x = pooled[column].to_numpy(dtype=np.float64)
        off_cent = (x * 100.0) % 1.0 != 0.0
        frac = x - np.round(x)
        print(
            f"  {column:24s} off-cent-grid share = {off_cent.mean():.4f} | "
            f"distinct residuals = {len(np.unique(np.round(frac, 6)))}"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - analysis script: show full failure
        print(f"jitter check failed: {exc}", file=sys.stderr)
        raise
