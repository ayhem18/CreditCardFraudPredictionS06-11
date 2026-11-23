"""Conditional quantile views of numeric x categorical — v3 (user design).

For each (numeric, categorical), on a 1..100 marginal-quantile grid:

- GAP ENVELOPE (user plot 1, refined): band between the most-above and
  most-below category's P(x <= Q_q | cat) around the marginal (zero),
  plus the width curve = max pairwise CDF gap. The widest point is
  annotated with the two categories that form it; a secondary top axis
  shows the marginal quantile VALUES (data units) at every 10th
  quantile.
- CONDITIONAL QUANTILE VALUES (user plot 2): per category, the value of
  each quantile inside the category vs the marginal diagonal (Q-Q).
  Log axes for the money columns.

Scale is on the axes (probability difference / data units) — no
size-encoded channels anywhere.

Run:  ./.venv/Scripts/python.exe experiments/eda/profile_interactions/numeric_cat_interactions_v3.py
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

N_QUANT = 100
CATS = ["channel", "device_type", "is_new_merchant", "merchant_category",
        "is_alarm", "card_type", "auth_method"]
NUMERICS = ["amount_usd", "account_balance_usd", "hours_since_last_txn",
            "distance_from_home_km", "velocity_score", "merchant_risk_score",
            "customer_age", "card_age_months", "cvv_retry_count",
            "txn_count_last_24h", "prior_disputes"]
MONEY = {"amount_usd", "account_balance_usd"}

train = pd.read_csv(get_train_csv_path())
train["is_alarm"] = alarm_flag_mask(train).to_numpy(dtype=bool)
plot_dir = get_script_plot_dir("numeric_cat_interactions_v3", section="plots", group="profile_interactions")
print(f"rows: {len(train)} | numerics: {len(NUMERICS)} | categoricals: {len(CATS)}")

cat_levels = {cat: sorted(train[cat].unique().tolist(), key=str) for cat in CATS}
q_levels = np.arange(1, N_QUANT + 1)

for numeric in tqdm(NUMERICS, desc="numerics", position=0):
    raw = train[numeric].to_numpy(dtype=np.float64)
    marginal_qs = np.quantile(raw, q_levels / N_QUANT)
    use_log = numeric in MONEY

    for cat in tqdm(CATS, desc=f"  {numeric}", position=1, leave=False):
        levels = cat_levels[cat]
        code = pd.Categorical(train[cat], categories=levels).codes.astype(np.int64)

        # P(x <= Q_q | cat) on the marginal-quantile grid + in-category quantile values
        p_cond = np.empty((len(levels), N_QUANT))
        cond_qs = np.empty((len(levels), N_QUANT))
        for i, level in enumerate(levels):
            vals = raw[code == i]
            p_cond[i] = (vals[:, None] <= marginal_qs[None, :]).mean(axis=0)
            cond_qs[i] = np.quantile(vals, q_levels / N_QUANT)

        dev = p_cond - (q_levels / N_QUANT)[None, :]          # vs marginal CDF at Q_q
        above = dev.max(axis=0)
        below = dev.min(axis=0)
        width = above - below
        wq = int(width.argmax())

        # ---- plot 1: gap envelope + width curve
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.fill_between(q_levels, below, above, color="#bbbbbb", alpha=0.6,
                        label="category envelope (min..max P(cond)-P(marg))")
        ax.plot(q_levels, width, color="#d65f5f", lw=1.8,
                label="max pairwise CDF gap (width)")
        ax.axhline(0, color="black", lw=0.8)
        top_cat = levels[int(dev[:, wq].argmax())]
        bot_cat = levels[int(dev[:, wq].argmin())]
        ax.annotate(
            f"q{wq + 1} ({marginal_qs[wq]:.4g}): gap {width[wq]:.3f}\n"
            f"{top_cat} above / {bot_cat} below",
            xy=(wq + 1, width[wq]), xytext=(wq + 1 - 22, width[wq] * 0.92),
            fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8),
        )
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(np.arange(10, N_QUANT + 1, 10))
        sec.set_xticklabels([f"{marginal_qs[q - 1]:.3g}" for q in range(10, N_QUANT + 1, 10)], fontsize=7)
        sec.set_xlabel(f"marginal quantile value ({numeric})", fontsize=8)
        ax.set_xlabel("marginal quantile (1..100)")
        ax.set_ylabel("P(x <= Q_q | cat) - P(x <= Q_q)")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{numeric} x {cat} — where the categorical bites")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(plot_dir / f"gap__{numeric}__{cat}.png", dpi=110)
        plt.close(fig)

        # ---- plot 2: conditional quantile values vs marginal (Q-Q)
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        ax.plot(marginal_qs, marginal_qs, color="black", lw=1, ls="--", label="marginal")
        cmap = plt.get_cmap("tab20")
        for i, level in enumerate(levels):
            ax.plot(marginal_qs, cond_qs[i], lw=1.4, color=cmap(i % 20), label=str(level))
        if use_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlabel(f"marginal quantile value ({numeric})")
        ax.set_ylabel(f"within-category quantile value ({numeric})")
        ax.set_title(f"{numeric} by {cat} — Q-Q (diagonal = marginal)")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        fig.tight_layout()
        fig.savefig(plot_dir / f"qq__{numeric}__{cat}.png", dpi=110)
        plt.close(fig)

        print(f"  {cat:18s} max gap {width.max():.3f} at q{wq + 1} "
              f"({marginal_qs[wq]:.4g}) — {top_cat} above / {bot_cat} below")

print(f"\nplots -> {plot_dir}")
