"""Conditional-CDF maps: numerical value x categorical context — v1.

The user's intuition algorithm (2026-11-23): integer-ize each numeric,
take its most frequent integer values (top 10% of distinct), and show
P(numeric <= value | category) for every selected value x category value.
If a raw numeric must not be read in isolation from a categorical, the
conditional CDF visibly departs from the marginal CDF — no test verdict,
the picture itself reformulates the hypothesis.

Note: for integer thresholds P(x <= v) == P(anchor(x) <= v), so each map
covers the raw and the anchor view at once. Labels are never used.

Outputs per (numeric, categorical): a two-panel figure — left, the
conditional CDF; right, the deviation from the marginal CDF (expected
minus unexpected). Plus per-pair stdout narration and a summary table.

Run:  ./.venv/Scripts/python.exe experiments/eda/profile_interactions/numeric_cat_interactions_v1.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from ccf_ml.cross_validation.alarm import alarm_flag_mask
from ccf_ml.cross_validation.constants import TARGET_COLUMN
from ccf_ml.file_utils import get_script_plot_dir, get_script_table_dir, get_train_csv_path

TOP_SHARE = 0.10
CATS = ["channel", "device_type", "is_new_merchant", "merchant_category",
        "is_alarm", "card_type", "auth_method"]
NUMERICS = ["amount_usd", "account_balance_usd", "hours_since_last_txn",
            "distance_from_home_km", "velocity_score", "merchant_risk_score",
            "customer_age", "card_age_months", "cvv_retry_count",
            "txn_count_last_24h", "prior_disputes"]

train = pd.read_csv(get_train_csv_path())
train["is_alarm"] = alarm_flag_mask(train).to_numpy(dtype=bool)
assert TARGET_COLUMN not in [c for c in train.columns if c in CATS]  # labels unused below

plot_dir = get_script_plot_dir("numeric_cat_interactions_v1", section="cdf_maps", group="profile_interactions")
table_dir = get_script_table_dir("numeric_cat_interactions_v1", group="profile_interactions")
print(f"rows: {len(train)} | numerics: {len(NUMERICS)} | categoricals: {len(CATS)}")

cat_levels = {cat: sorted(train[cat].unique().tolist(), key=str) for cat in CATS}
summary_rows = []

for numeric in tqdm(NUMERICS, desc="numerics", position=0):
    v_int = np.floor(train[numeric].to_numpy(dtype=np.float64)).astype(np.int64)
    counts = pd.Series(v_int).value_counts()
    n_sel = max(1, int(TOP_SHARE * len(counts)))
    selected = np.sort(counts.index.to_numpy()[:n_sel])  # most frequent, then ascending for display
    print(f"\n[{numeric}] distinct ints: {len(counts)} | selected top {n_sel} frequent values "
          f"({selected.min()}..{selected.max()}, median {int(np.median(selected))})")

    for cat in tqdm(CATS, desc=f"  {numeric}", position=1, leave=False):
        levels = cat_levels[cat]
        code = pd.Categorical(train[cat], categories=levels).codes.astype(np.int64)
        max_v = int(v_int.max())
        joint = np.bincount(code * (max_v + 1) + v_int, minlength=len(levels) * (max_v + 1))
        tab = joint.reshape(len(levels), max_v + 1).astype(np.float64)
        cum = tab.cumsum(axis=1)
        p_cond = cum / cum[:, -1:]
        p_marg = cum.sum(axis=0) / tab.sum()  # tab.sum() == n rows; cum.sum() would sum the whole cumulative matrix
        dev = p_cond[:, selected] - p_marg[selected][None, :]

        # stdout narration: where does the conditional CDF depart most from marginal?
        flat = np.abs(dev).argmax()
        ci, vi = np.unravel_index(flat, dev.shape)
        summary_rows.append({
            "numeric": numeric, "categorical": cat,
            "max |dev|": round(float(np.abs(dev).max()), 3),
            "at value": int(selected[vi]), "at category": str(levels[ci]),
            "P(cond)": round(float(p_cond[ci, selected[vi]]), 3),
            "P(marg)": round(float(p_marg[selected[vi]]), 3),
            "median |dev|": round(float(np.median(np.abs(dev))), 3),
        })
        print(f"  {cat:18s} max|P(cond)-P(marg)| = {np.abs(dev).max():.3f} "
              f"at value {int(selected[vi])}, category {levels[ci]} "
              f"({p_cond[ci, selected[vi]]:.3f} vs {p_marg[selected[vi]]:.3f})")

        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.28 * n_sel + 1.5)),
                                 gridspec_kw={"width_ratios": [1, 1]})
        im0 = axes[0].pcolormesh(p_cond[:, selected], cmap="viridis", vmin=0, vmax=1)
        axes[0].set_title(f"P({numeric} <= v | {cat})")
        vabs = max(float(np.abs(dev).max()), 0.02)
        im1 = axes[1].pcolormesh(dev, cmap="coolwarm", vmin=-vabs, vmax=vabs)
        axes[1].set_title("deviation from marginal P(<= v)")
        for ax in axes:
            ax.set_xlabel(f"{cat} (level order as listed)")
            ax.set_ylabel("v (most frequent integer values, ascending)")
            ax.set_xticks(np.arange(len(levels)) + 0.5)
            ax.set_xticklabels([str(l) for l in levels], rotation=45, ha="right", fontsize=7)
        step = max(1, n_sel // 45)
        for ax in axes:
            ax.set_yticks(np.arange(0, n_sel, step) + 0.5)
            ax.set_yticklabels([str(int(v)) for v in selected[::step]], fontsize=6)
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        fig.tight_layout()
        fig.savefig(plot_dir / f"{numeric}__{cat}.png", dpi=110)
        plt.close(fig)

summary = pd.DataFrame(summary_rows).sort_values("max |dev|", ascending=False)
summary.to_csv(table_dir / "conditional_cdf_deviation_summary.csv", index=False)
print("\n=== strongest conditional-CDF departures (see maps for the shapes) ===")
print(summary.head(25).to_string(index=False))
print(f"\nmaps -> {plot_dir}\nsummary -> {table_dir / 'conditional_cdf_deviation_summary.csv'}")
