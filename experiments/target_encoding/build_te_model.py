"""Per-value + digit out-of-fold target-encoding model (S6E11 winning recipe).

Reproduces the public "generator-fingerprint" technique (megayak / goodpjw2008
/ areebbb): the synthetic generator wrote the label first, then each field
with a language model, so every value and the digits it was written with carry
label signal. We target-encode ~60 keys (exact value of all 24 columns; and
cents / int / ten / dec / leading-1 / leading-2 digit keys of the 6 decimal
columns) out-of-fold, then train our shallow XGB.

Fold-safety, and alignment with the existing blend members:
- OUTER folds = the frozen alarm_profile folds, read from an existing OOF
  (icU_base) so this model's OOF lines up row-for-row with icU_base/riskz/cvv
  and is directly blendable.
- For an outer fold: training rows are encoded by an INNER 5-fold split (no row
  sees its own label); validation + test rows are encoded from the whole outer
  training part; smoothing toward the fold fraud rate (smooth=20).
- Test prediction = mean of the 5 outer-fold models.

Run: ./.venv/Scripts/python.exe -I experiments/target_encoding/build_te_model.py
Writes storage/artifacts/training/icU_te1/{oof.csv,submission.csv,config.json}.
"""
from __future__ import annotations
import json, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from ccf_ml.file_utils import get_train_csv_path, get_test_csv_path

CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
         "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
NUMS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
        "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
        "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"]
FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km",
          "account_balance_usd", "velocity_score", "merchant_risk_score"]
FEATS = NUMS + CATS + FLAGS
SMOOTH, INNER = 20.0, 5
PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
              colsample_bytree=0.8, random_state=42, tree_method="hist",
              eval_metric="auc", n_jobs=-1)
OUT = "storage/artifacts/training/icU_te1"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


tr = pd.read_csv(get_train_csv_path())
te = pd.read_csv(get_test_csv_path())
y = tr["is_fraud"].to_numpy(np.float64)
NTR = len(tr)
full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
log(f"train {NTR} test {len(te)} | fraud {y.mean():.4f}")

# ---- keys: exact value of every column + digit keys of the decimal columns ----
def key_table(df: pd.DataFrame) -> dict[str, np.ndarray]:
    K = {}
    for c in FEATS:
        K[f"v_{c}"] = df[c].astype(str).values if c in CATS else df[c].values
    for c in FLOATS:
        x = df[c].values.astype(np.float64)
        K[f"cents_{c}"] = (np.round(x * 100) % 100).astype(np.int64)
        K[f"int_{c}"] = np.floor(x).astype(np.int64)
        K[f"ten_{c}"] = np.floor(x / 10).astype(np.int64)
        K[f"dec_{c}"] = np.floor(x * 10 + 1e-6).astype(np.int64)
        i = pd.Series(np.floor(x).astype(np.int64)).astype(str)
        K[f"l1_{c}"] = (i.str.len() * 10 + i.str[0].astype(int)).values
        K[f"l2_{c}"] = (i.str.len() * 100 + i.str[:2].astype(int).clip(0, 99)).values
    return K


KEYS = key_table(full)
KNAMES = list(KEYS)
codes = [pd.factorize(pd.Series(KEYS[n]))[0].astype(np.int32) for n in KNAMES]
KMAT = np.column_stack(codes)
NCAT = np.array([c.max() + 1 for c in codes])
del KEYS, codes
log(f"{len(KNAMES)} target-encoded keys")

# ---- raw model block: numerics + flags as-is, cats as integer codes ----
RAW = np.column_stack(
    [full[c].values.astype(np.float32) for c in NUMS]
    + [pd.factorize(full[c], sort=True)[0].astype(np.float32) for c in CATS]
    + [full[c].astype(np.float32).values for c in FLAGS]
)
RAW_NAMES = NUMS + CATS + FLAGS

# ---- outer folds from the frozen alarm_profile OOF (row-aligned with the blend) ----
fold_src = pd.read_csv("storage/artifacts/training/icU_base/oof.csv").sort_values("id").reset_index(drop=True)
assert (fold_src["id"].to_numpy() == tr["id"].to_numpy()).all(), "fold source misaligned with train"
fold_of = fold_src["fold_id"].to_numpy()
N_SPLITS = int(fold_of.max() + 1)


def te_rate(k, t, ncat, prior, smooth=SMOOTH):
    s = np.bincount(k, weights=t, minlength=ncat)
    n = np.bincount(k, minlength=ncat)
    return (s + prior * smooth) / (n + smooth)


Kt = KMAT[NTR:]
oof = np.zeros(NTR, np.float64)
test_pred = np.zeros(len(te), np.float64)
for f in range(N_SPLITS):
    a = np.flatnonzero(fold_of != f)          # outer-train (row indices in train)
    b = np.flatnonzero(fold_of == f)          # outer-val
    ya = y[a]
    Ka, Kb = KMAT[a], KMAT[b]
    Ea = np.zeros(Ka.shape, np.float32)
    Eb = np.zeros(Kb.shape, np.float32)
    Et = np.zeros(Kt.shape, np.float32)
    inner = list(StratifiedKFold(INNER, shuffle=True, random_state=f).split(a, ya))
    for j in range(KMAT.shape[1]):
        for i, o in inner:
            Ea[o, j] = te_rate(Ka[i, j], ya[i], NCAT[j], ya[i].mean())[Ka[o, j]]
        r = te_rate(Ka[:, j], ya, NCAT[j], ya.mean())
        Eb[:, j] = r[Kb[:, j]]
        Et[:, j] = r[Kt[:, j]]
    Xa = np.hstack([RAW[a], Ea]); Xb = np.hstack([RAW[b], Eb]); Xtst = np.hstack([RAW[NTR:], Et])
    m = XGBClassifier(**PARAMS)
    m.fit(Xa, ya)
    oof[b] = m.predict_proba(Xb)[:, 1]
    test_pred += m.predict_proba(Xtst)[:, 1] / N_SPLITS
    log(f"fold {f}: val_auc {roc_auc_score(y[b], oof[b]):.5f}")

auc = roc_auc_score(y, oof)
log(f"OOF AUC {auc:.5f}  (base 0.94819, champion 0.94806)")
for f in range(N_SPLITS):
    b = fold_of == f
    log(f"  fold {f}: {roc_auc_score(y[b], oof[b]):.5f}")

import os
os.makedirs(OUT, exist_ok=True)
pd.DataFrame({"id": tr["id"], "fold_id": fold_of, "is_fraud": y.astype(int), "y_prob": oof}).to_csv(f"{OUT}/oof.csv", index=False)
pd.DataFrame({"id": te["id"], "is_fraud": test_pred}).to_csv(f"{OUT}/submission.csv", index=False)
json.dump({"exp_name": "icU_te1", "model": "xgb_shallow_on_per_value_digit_TE",
           "n_keys": len(KNAMES), "n_raw": RAW.shape[1], "smooth": SMOOTH,
           "oof_auc": auc, "params": PARAMS, "keys": KNAMES}, open(f"{OUT}/config.json", "w"), indent=1)
log(f"wrote {OUT}/oof.csv, submission.csv, config.json")
