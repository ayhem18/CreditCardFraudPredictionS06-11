"""Distribution-shift signal map + feature battery + buried-fraud screen.

For the four numerics the user flagged (amount, risk, velocity, age):

A. Which categoricals does the numeric's distribution actually move across?
   -> eta^2 (variance of the RANK-transformed numeric explained by the cat).
   The top cats form the numeric's interacting sub-profile.

B. A battery of within-sub-profile engineered features (signed robust-z,
   |z|, within-cell percentile, raw deviation, cell median as reference),
   built at two granularities: the single top cat, and the top two.

C. The screen: for every candidate feature, the within-k6-cell AUC over
   (buried fraud, its k6 clean confusers) pairs -- does the feature rank the
   frauds the champion MISSED above the cleans it confused them with? The
   raw numeric is the baseline; engineered - raw = added discriminative
   power on exactly the failure set. Leaky-direction screen (no folds).
"""
from __future__ import annotations
import numpy as np, pandas as pd
from ccf_ml.cross_validation.alarm import alarm_flag_mask
from ccf_ml.file_utils import get_train_csv_path

NUMERICS = ["amount_usd", "merchant_risk_score", "velocity_score", "customer_age"]
CATS = ["channel", "device_type", "is_new_merchant", "merchant_category",
        "is_alarm", "card_type", "auth_method"]
MIN_CELL = 200
MAP = "storage/artifacts/xgbU_tuned_av2_oof_analysis/mistakes_v2/tables/map1/"

tr = pd.read_csv(get_train_csv_path())
tr["is_alarm"] = alarm_flag_mask(tr).to_numpy(dtype=bool)
buried = np.load(MAP + "buried_ids.npy")
off = np.load(MAP + "k6_offsets.npy")
cids = np.fromfile(MAP + "k6_clean_ids.i32", dtype=np.int32)
nbr = np.diff(off)
print(f"rows {len(tr)} | buried {len(buried)} | k6 clean pairs {len(cids):,}")


def screen(f: np.ndarray) -> float:
    """Within-k6-cell AUC: buried frauds vs their clean confusers."""
    rep = np.repeat(f[buried], nbr)
    fc = f[cids]
    return float(((rep > fc).sum() + 0.5 * (rep == fc).sum()) / len(fc))


def eta2_rank(num: str, cat: str) -> float:
    """Fraction of the rank-transformed numeric's variance between cat levels."""
    r = pd.Series(tr[num]).rank().to_numpy()
    grand = r.mean()
    ss_tot = ((r - grand) ** 2).sum()
    between = 0.0
    for _, idx in pd.Series(range(len(tr))).groupby(tr[cat].to_numpy()).groups.items():
        g = r[np.asarray(idx)]
        between += len(g) * (g.mean() - grand) ** 2
    return float(between / ss_tot)


def cell_key(cols: list[str]) -> np.ndarray:
    k = tr[cols[0]].astype(str)
    for c in cols[1:]:
        k = k.str.cat(tr[c].astype(str), sep="|")
    return k.to_numpy()


def within_cell_features(num: str, cols: list[str]) -> dict[str, np.ndarray]:
    """Target-free placement features of `num` within the cell `cols`."""
    x = tr[num].to_numpy(dtype=np.float64)
    key = cell_key(cols)
    s = pd.DataFrame({"k": key, "x": x})
    g = s.groupby("k")["x"]
    med = g.transform("median").to_numpy()
    # robust scale (MAD) per cell, global fallback
    gmed_all = np.median(x)
    gscale = 1.4826 * np.median(np.abs(x - gmed_all)) or x.std() or 1.0
    mad = g.transform(lambda v: np.median(np.abs(v - np.median(v)))).to_numpy()
    scale = np.where(mad > 0, 1.4826 * mad, gscale)
    n = g.transform("size").to_numpy()
    pct = s.groupby("k")["x"].rank(pct=True).to_numpy()
    # small cells: fall back to global placement
    small = n < MIN_CELL
    med = np.where(small, gmed_all, med)
    scale = np.where(small, gscale, scale)
    signed = (x - med) / scale
    return {
        "signed_z": signed,
        "abs_z": np.abs(signed),
        "within_pct": pct,
        "raw_dev": x - med,
        "cell_median": med,
    }


# ---- A. signal map -------------------------------------------------------
print("\n=== A. distribution-shift eta^2 (rank) : numeric x cat ===")
eta = pd.DataFrame(
    {num: {cat: round(eta2_rank(num, cat), 4) for cat in CATS} for num in NUMERICS}
).T
print(eta.to_string())
subprofiles = {num: eta.loc[num].sort_values(ascending=False).index.tolist() for num in NUMERICS}
print("\ntop interacting cats per numeric:")
for num in NUMERICS:
    print(f"  {num:22s} {subprofiles[num][:3]}")

# ---- B+C. battery + screen ----------------------------------------------
print("\n=== C. buried-fraud k6 screen (within-cell AUC; raw baseline first) ===")
rows = []
for num in NUMERICS:
    raw = tr[num].to_numpy(dtype=np.float64)
    raw_auc = screen(raw)
    top = subprofiles[num]
    for cols in ([top[0]], top[:2]):
        feats = within_cell_features(num, cols)
        for fname, fval in feats.items():
            rows.append({
                "numeric": num, "subprofile": "+".join(cols),
                "feature": fname, "screen_auc": round(screen(fval), 4),
                "raw_auc": round(raw_auc, 4),
                "added": round(screen(fval) - raw_auc, 4),
            })
    rows.append({"numeric": num, "subprofile": "-", "feature": "RAW",
                 "screen_auc": round(raw_auc, 4), "raw_auc": round(raw_auc, 4), "added": 0.0})

res = pd.DataFrame(rows).sort_values(["numeric", "screen_auc"], ascending=[True, False])
print(res.to_string(index=False))
print("\n=== best engineered feature per numeric (by screen AUC) ===")
best = res[res.feature != "RAW"].sort_values("screen_auc", ascending=False).groupby("numeric").head(2)
print(best.sort_values("screen_auc", ascending=False).to_string(index=False))
out = "C:/Users/bouab/AppData/Local/Temp/claude/41d1353ff04a981e/1a0484e6-4e9f-4cf4-ba21-0bad765521b8/scratchpad/battery_screen_results.csv"
res.to_csv(out, index=False)
print(f"\nwrote {out}")
