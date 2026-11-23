"""Conditional-CDF departures — v2 plots (user-specified bubble design).

Same label-free statistic as v1: P(numeric <= v | category) at the most
frequent integer values (top 10% of distinct), against the marginal
P(numeric <= v). x = selected common values, y = category values; a heat
map would be natural but x is too long, so TWO scatter plots per pair:

- cdf__*.png   — circle diameter = P(num <= v | cat): read down a column
  of x; categories with similar diameters group together w.r.t. the
  numeric.
- dev__*.png   — circle diameter = |P(num <= v | cat) - P(num <= v)|
  (diameter cannot be negative; color carries the sign: red = conditional
  above marginal, blue = below).
- hist__*.png  — for categoricals with <= 5 levels: the N histograms
  plotted together (density, x capped at p99.5).

Run:  ./.venv/Scripts/python.exe experiments/eda/profile_interactions/numeric_cat_interactions_v2.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from ccf_ml.cross_validation.alarm import alarm_flag_mask
from ccf_ml.file_utils import get_script_plot_dir, get_train_csv_path

TOP_SHARE = 0.10
HISTO_MAX_LEVELS = 5
CATS = ["channel", "device_type", "is_new_merchant", "merchant_category",
        "is_alarm", "card_type", "auth_method"]
NUMERICS = ["amount_usd", "account_balance_usd", "hours_since_last_txn",
            "distance_from_home_km", "velocity_score", "merchant_risk_score",
            "customer_age", "card_age_months", "cvv_retry_count",
            "txn_count_last_24h", "prior_disputes"]

train = pd.read_csv(get_train_csv_path())
train["is_alarm"] = alarm_flag_mask(train).to_numpy(dtype=bool)
plot_dir = get_script_plot_dir("numeric_cat_interactions_v2", section="plots", group="profile_interactions")
print(f"rows: {len(train)} | numerics: {len(NUMERICS)} | categoricals: {len(CATS)}")

cat_levels = {cat: sorted(train[cat].unique().tolist(), key=str) for cat in CATS}


def base_axes(levels, cat, numeric, selected, title):
    fig, ax = plt.subplots(figsize=(14, 1.2 + 0.55 * len(levels)))
    if selected.max() / max(selected.min(), 1) > 50:
        ax.set_xscale("log")
    ax.set_yticks(np.arange(len(levels)))
    ax.set_yticklabels([str(l) for l in levels])
    ax.set_xlabel(f"{numeric}: most common integer values (top {int(TOP_SHARE * 100)}%, ascending)")
    ax.set_ylabel(cat)
    ax.set_title(title)
    return fig, ax


for numeric in tqdm(NUMERICS, desc="numerics", position=0):
    raw = train[numeric].to_numpy(dtype=np.float64)
    v_int = np.floor(raw).astype(np.int64)
    counts = pd.Series(v_int).value_counts()
    n_sel = max(1, int(TOP_SHARE * len(counts)))
    selected = np.sort(counts.index.to_numpy()[:n_sel])
    print(f"\n[{numeric}] distinct ints: {len(counts)} | selected {n_sel} common values "
          f"({selected.min()}..{selected.max()})")

    for cat in tqdm(CATS, desc=f"  {numeric}", position=1, leave=False):
        levels = cat_levels[cat]
        code = pd.Categorical(train[cat], categories=levels).codes.astype(np.int64)
        max_v = int(v_int.max())
        tab = np.bincount(code * (max_v + 1) + v_int,
                          minlength=len(levels) * (max_v + 1)).reshape(len(levels), max_v + 1).astype(float)
        cum = tab.cumsum(axis=1)
        p_cond_sel = cum[:, selected]
        p_cond = p_cond_sel / cum[:, -1:]
        p_marg_sel = (cum.sum(axis=0) / tab.sum())[selected]
        dev = p_cond - p_marg_sel[None, :]

        ci, vi = np.unravel_index(np.abs(dev).argmax(), dev.shape)
        print(f"  {cat:18s} max|dev| = {np.abs(dev).max():.3f} at v={int(selected[vi])}, "
              f"{levels[ci]} ({p_cond[ci, vi]:.3f} vs {p_marg_sel[vi]:.3f})")

        xs = np.tile(selected, len(levels))
        ys = np.repeat(np.arange(len(levels)), n_sel)

        # plot 1: diameter = P(num <= v | cat) — grouping read
        fig, ax = base_axes(
            levels, cat, numeric, selected,
            f"{numeric} vs {cat} — diameter = P({numeric} <= v | {cat})",
        )
        ax.scatter(xs, ys, s=10 + 600 * p_cond.ravel(), c="#555555", alpha=0.55, edgecolors="none")
        fig.tight_layout()
        fig.savefig(plot_dir / f"cdf__{numeric}__{cat}.png", dpi=110)
        plt.close(fig)

        # plot 2: diameter = |deviation|, color = sign
        fig, ax = base_axes(
            levels, cat, numeric, selected,
            f"{numeric} vs {cat} — diameter = |P(<= v | {cat}) - P(<= v)| (red above, blue below)",
        )
        sizes = 8 + 420 * (np.abs(dev) / max(np.abs(dev).max(), 1e-9)).ravel()
        sign = np.sign(dev.ravel())
        ax.scatter(xs[sign >= 0], ys[sign >= 0], s=sizes[sign >= 0], c="#d65f5f",
                   alpha=0.65, edgecolors="none", label="conditional above marginal")
        ax.scatter(xs[sign < 0], ys[sign < 0], s=sizes[sign < 0], c="#4878d0",
                   alpha=0.65, edgecolors="none", label="conditional below marginal")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"dev__{numeric}__{cat}.png", dpi=110)
        plt.close(fig)

        # plot 3 (low-cardinality cats): the N histograms together
        if len(levels) <= HISTO_MAX_LEVELS:
            cap = float(np.quantile(raw, 0.995))
            positive = raw[raw > 0]
            use_log = len(positive) > 0 and cap / max(positive.min(), 1e-9) > 100
            fig, ax = plt.subplots(figsize=(11, 4.5))
            bins = (np.logspace(np.log10(max(positive.min(), 1e-9)), np.log10(cap), 60)
                    if use_log else np.linspace(float(raw.min()), cap, 60))
            labels = train[cat].to_numpy()
            for level in levels:
                vals = raw[labels == level]
                vals = vals[vals <= cap]
                ax.hist(vals, bins=bins, density=True, alpha=0.45, label=f"{level} (n={len(vals):,})")
            if use_log:
                ax.set_xscale("log")
            ax.set_xlabel(f"{numeric} (x capped at p99.5 = {cap:.4g})")
            ax.set_ylabel("density")
            ax.set_title(f"{numeric} by {cat} — histograms together")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plot_dir / f"hist__{numeric}__{cat}.png", dpi=110)
            plt.close(fig)

print(f"\nplots -> {plot_dir}")
