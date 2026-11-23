"""Within-profile numerical placement — v1 screen.

Premise (user, 2026-11-23): once a categorical profile is set, only the
numerics decide fraud vs clean; raw rates are always low and say little.
Question: does a numeric's POSITION INSIDE its profile (its within-profile
quantile) rank frauds above the cleans of the same profile?

Screen: for every (numeric, profile-family) pair, the within-profile AUC —
the stratified Mann-Whitney over fraud-clean pairs that share the profile —
against the numeric's marginal AUC. High within-profile AUC with the pairs
restricted to same-profile rows is exactly "a metric that ranks fraudulent
transactions higher inside categorical profiles".

Profile families:
- profile7: the user's 7 (channel, device_type, is_new_merchant,
  merchant_category, is_alarm, card_type, auth_method), with escalation:
  full 7 -> largest drop-one group >= 200 rows -> (merchant, auth, is_alarm).
- merchant+is_new_merchant, channel+auth_method, device_type,
  merchant_category (the named candidate aggregates).

This is a label-side screen (leaky direction, no folds) — features get
built fold-safe only for winners.

Run:  ./.venv/Scripts/python.exe experiments/eda/profile_interactions/profile_quantiles_v1.py
Plots/tables: storage/artifacts/profile_interactions/profile_quantiles_v1/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ccf_ml.cross_validation.alarm import alarm_flag_mask
from ccf_ml.cross_validation.constants import TARGET_COLUMN
from ccf_ml.file_utils import get_script_plot_dir, get_script_table_dir, get_train_csv_path

MIN_ROWS = 200
NUMERICS = [
    "amount_usd", "account_balance_usd", "hours_since_last_txn",
    "distance_from_home_km", "velocity_score", "merchant_risk_score",
    "customer_age", "card_age_months", "cvv_retry_count",
    "txn_count_last_24h", "prior_disputes",
]
PROFILE7 = [
    "channel", "device_type", "is_new_merchant", "merchant_category",
    "is_alarm", "card_type", "auth_method",
]

train = pd.read_csv(get_train_csv_path())
train["is_alarm"] = alarm_flag_mask(train).to_numpy(dtype=bool)
y = train[TARGET_COLUMN].to_numpy(dtype=bool)
print(f"rows: {len(train)} | frauds: {int(y.sum())} | rate: {y.mean():.4f}")

plot_dir = get_script_plot_dir("profile_quantiles_v1", section="screen", group="profile_interactions")
table_dir = get_script_table_dir("profile_quantiles_v1", group="profile_interactions")


def codes_of(columns: list[str]) -> np.ndarray:
    codes = np.zeros(len(train), dtype=np.int64)
    for column in columns:
        cc, uniq = pd.factorize(train[column])
        codes = codes * max(len(uniq), 1) + cc.astype(np.int64)
    return codes


def size_map(codes: np.ndarray) -> dict[int, int]:
    return train.assign(k=codes).groupby("k").size().to_dict()


# --- effective profile for family profile7 (full -> best drop-one -> base) ---
full_codes = codes_of(PROFILE7)
full_size = train.assign(k=full_codes).groupby("k").size()
full_size_map = full_size.to_dict()

drop_codes = []
drop_size_maps = []
for j, dropped in enumerate(PROFILE7):
    cols = [c for c in PROFILE7 if c != dropped]
    codes = codes_of(cols)
    drop_codes.append((dropped, codes))
    drop_size_maps.append(train.assign(k=codes).groupby("k").size().to_dict())

base_codes = codes_of(["merchant_category", "auth_method", "is_alarm"])
base_size_map = size_map(base_codes)

n = len(train)
eff_code = np.empty(n, dtype=np.int64)
eff_level = np.empty(n, dtype=object)
full_sizes = np.array([full_size_map.get(int(k), 0) for k in full_codes])
ok_full = full_sizes >= MIN_ROWS
eff_code[ok_full] = full_codes[ok_full]
eff_level[ok_full] = "full7"

rest = np.flatnonzero(~ok_full)
drop_size_matrix = np.empty((len(rest), len(drop_codes)), dtype=np.int64)
for j, (_, codes_j) in enumerate(drop_codes):
    smap = drop_size_maps[j]
    drop_size_matrix[:, j] = [smap.get(int(codes_j[i]), 0) for i in rest]
best_j = drop_size_matrix.argmax(axis=1)
best_size = drop_size_matrix[np.arange(len(rest)), best_j]
ok_drop = best_size >= MIN_ROWS
idx = rest[ok_drop]
for j_local in np.unique(best_j[ok_drop]):
    sel = idx[best_j[ok_drop] == j_local]
    eff_code[sel] = drop_codes[j_local][1][sel]
    eff_level[sel] = f"drop:{drop_codes[j_local][0]}"
still = rest[~ok_drop]
eff_code[still] = base_codes[still]
eff_level[still] = "base(merchant,auth,is_alarm)"
print("profile7 level mix:")
print(pd.Series(eff_level).value_counts().to_string())

FAMILIES: dict[str, np.ndarray] = {
    "profile7": eff_code,
    "merchant+is_new": codes_of(["merchant_category", "is_new_merchant"]),
    "channel+auth": codes_of(["channel", "auth_method"]),
    "device_type": codes_of(["device_type"]),
    "merchant": codes_of(["merchant_category"]),
}


def within_profile_auc(codes: np.ndarray, values: np.ndarray) -> tuple[float, int, int]:
    """Stratified AUC over same-profile fraud-clean pairs; (auc, pairs, groups)."""
    frame = pd.DataFrame({"g": codes, "v": values, "y": y})
    frame["r"] = frame.groupby("g")["v"].rank(method="average")
    grp = frame.groupby("g")["y"].agg(nf="sum", n="size")
    rf = frame[frame["y"]].groupby("g")["r"].sum()
    grp = grp.join(rf.rename("rf"), how="left").fillna({"rf": 0.0})
    grp = grp[(grp["nf"] > 0) & (grp["n"] - grp["nf"] > 0)]
    if grp.empty:
        return float("nan"), 0, 0
    grp["nc"] = grp["n"] - grp["nf"]
    u = grp["rf"] - grp["nf"] * (grp["nf"] + 1) / 2.0
    return float(u.sum() / (grp["nf"] * grp["nc"]).sum()), int((grp["nf"] * grp["nc"]).sum()), len(grp)


def marginal_auc(values: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, values))


rows = []
pct_store: dict[tuple[str, str], np.ndarray] = {}
for numeric in NUMERICS:
    raw = train[numeric].to_numpy(dtype=np.float64)
    for family, codes in FAMILIES.items():
        pct = pd.Series(raw).groupby(codes).rank(pct=True).to_numpy()
        auc_w, pairs, groups = within_profile_auc(codes, pct)
        rows.append({
            "numeric": numeric, "family": family, "groups": groups,
            "within_pairs": pairs,
            "auc_within": round(auc_w, 4),
            "auc_marginal": round(marginal_auc(pct), 4),
            "within-marginal": round(auc_w - marginal_auc(pct), 4),
        })
        pct_store[(numeric, family)] = pct

screen = pd.DataFrame(rows).sort_values("auc_within", ascending=False)
screen.to_csv(table_dir / "within_profile_auc.csv", index=False)
print("\ntop 15 by within-profile AUC:")
print(screen.head(15).to_string(index=False))
print("\nper family, best numeric:")
print(screen.groupby("family").head(1).to_string(index=False))

for rank, row in enumerate(screen.head(3).itertuples()):
    pct = pct_store[(row.numeric, row.family)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(pct[y], bins=40, alpha=0.65, color="#d65f5f", label="frauds", density=True)
    ax.hist(pct[~y], bins=40, alpha=0.65, color="#4878d0", label="cleans", density=True)
    ax.set_xlabel(f"{row.numeric}: within-{row.family} percentile")
    ax.set_ylabel("density")
    ax.set_title(f"within-profile AUC {row.auc_within:.4f} | marginal {row.auc_marginal:.4f} | {row.family}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"top{rank + 1}_{row.numeric}_{row.family.replace('+', '_')}.png", dpi=110)
    plt.close(fig)
print(f"\nwrote {table_dir / 'within_profile_auc.csv'} and top-3 plots to {plot_dir}")
