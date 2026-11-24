"""TE stack: per-value + digit + JOINT target encoding -> 3 diverse members.

Extends build_te_model.py:
- adds JOINT keys (the non-additive column pairs from the public analysis),
- trains THREE decorrelated members on the shared out-of-fold encodings:
    * XGB  tree  (our depth-5 config)           -> icU_te2
    * LightGBM shallow tree (num_leaves=16)      -> icU_lgb
    * GLM  logistic regression on logit(E)       -> icU_glm
- blends them with the existing base/champion and submits nothing (prints the
  OOF; a separate submit step picks the winner).

Same frozen alarm_profile outer folds (read from icU_base) so every member's
OOF lines up row-for-row and blends with the earlier models.

Run: ./.venv/Scripts/python.exe -I experiments/target_encoding/build_te_stack.py
"""
from __future__ import annotations
import json, os, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
XGB_P = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,
             colsample_bytree=0.8, random_state=42, tree_method="hist", eval_metric="auc", n_jobs=-1)
LGB_P = dict(n_estimators=700, learning_rate=0.03, num_leaves=16, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
JOINT = [
    ("j_category_risk", [("merchant_category", "raw"), ("merchant_risk_score", "raw")]),
    ("j_foreign_ip", [("is_foreign_transaction", "raw"), ("ip_country_mismatch", "raw")]),
    ("j_category_amount", [("merchant_category", "raw"), ("amount_usd", "q")]),
    ("j_txn_velocity", [("txn_count_last_24h", "raw"), ("velocity_score", "q")]),
    ("j_category_age", [("merchant_category", "raw"), ("customer_age", "raw")]),
    ("j_amount_risk", [("amount_usd", "q"), ("merchant_risk_score", "q")]),
    ("j_hours_velocity", [("hours_since_last_txn", "q"), ("velocity_score", "q")]),
    ("j_age_risk", [("customer_age", "raw"), ("merchant_risk_score", "q")]),
    ("j_newmerchant_weekday", [("is_new_merchant", "raw"), ("day_of_week", "raw")]),
    ("j_scam_velocity", [("is_ai_generated_scam_attempt", "raw"), ("velocity_score", "q")]),
    ("j_age_cardage", [("customer_age", "raw"), ("card_age_months", "raw")]),
    ("j_hour_weekday", [("time_of_day_hour", "raw"), ("day_of_week", "raw")]),
    ("j_txn_age", [("txn_count_last_24h", "raw"), ("customer_age", "raw")]),
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


tr = pd.read_csv(get_train_csv_path())
te = pd.read_csv(get_test_csv_path())
y = tr["is_fraud"].to_numpy(np.float64)
NTR = len(tr)
full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
log(f"train {NTR} test {len(te)} fraud {y.mean():.4f}")


def key_table(df):
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


def col_code(df, c, mode):
    return pd.qcut(df[c], 10, labels=False, duplicates="drop").to_numpy() if mode == "q" \
        else pd.factorize(df[c])[0]


KEYS = key_table(full)
for name, parts in JOINT:
    k = np.zeros(len(full), np.int64)
    for c, mode in parts:
        u = col_code(full, c, mode)
        k = k * (int(np.nanmax(u)) + 1) + np.where(np.isnan(u.astype(float)) if u.dtype.kind == "f" else False, -1, u).astype(np.int64)
    KEYS[name] = pd.factorize(k)[0]
KNAMES = list(KEYS)
codes = [pd.factorize(pd.Series(KEYS[n]))[0].astype(np.int32) for n in KNAMES]
KMAT = np.column_stack(codes)
NCAT = np.array([c.max() + 1 for c in codes])
del KEYS, codes
log(f"{len(KNAMES)} keys ({len(JOINT)} joint)")

RAW = np.column_stack(
    [full[c].values.astype(np.float32) for c in NUMS]
    + [pd.factorize(full[c], sort=True)[0].astype(np.float32) for c in CATS]
    + [full[c].astype(np.float32).values for c in FLAGS])

fold_src = pd.read_csv("storage/artifacts/training/icU_base/oof.csv").sort_values("id").reset_index(drop=True)
assert (fold_src["id"].to_numpy() == tr["id"].to_numpy()).all()
fold_of = fold_src["fold_id"].to_numpy()
NSP = int(fold_of.max() + 1)


def te_rate(k, t, ncat, prior):
    s = np.bincount(k, weights=t, minlength=ncat); n = np.bincount(k, minlength=ncat)
    return (s + prior * SMOOTH) / (n + SMOOTH)


def logit(p): return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


Kt = KMAT[NTR:]
members = {m: (np.zeros(NTR), np.zeros(len(te))) for m in ("te2", "lgb", "glm")}
for f in range(NSP):
    a = np.flatnonzero(fold_of != f); b = np.flatnonzero(fold_of == f); ya = y[a]
    Ka, Kb = KMAT[a], KMAT[b]
    Ea = np.zeros(Ka.shape, np.float32); Eb = np.zeros(Kb.shape, np.float32); Et = np.zeros(Kt.shape, np.float32)
    inner = list(StratifiedKFold(INNER, shuffle=True, random_state=f).split(a, ya))
    for j in range(KMAT.shape[1]):
        for i, o in inner:
            Ea[o, j] = te_rate(Ka[i, j], ya[i], NCAT[j], ya[i].mean())[Ka[o, j]]
        r = te_rate(Ka[:, j], ya, NCAT[j], ya.mean())
        Eb[:, j] = r[Kb[:, j]]; Et[:, j] = r[Kt[:, j]]
    Xa, Xb, Xt = np.hstack([RAW[a], Ea]), np.hstack([RAW[b], Eb]), np.hstack([RAW[NTR:], Et])
    # tree members
    for name, Model, P in [("te2", XGBClassifier, XGB_P), ("lgb", LGBMClassifier, LGB_P)]:
        mo = Model(**P); mo.fit(Xa, ya)
        members[name][0][b] = mo.predict_proba(Xb)[:, 1]
        members[name][1][:] += mo.predict_proba(Xt)[:, 1] / NSP
    # GLM on the encodings as logits
    sc = StandardScaler().fit(logit(Ea))
    glm = LogisticRegression(C=1.0, max_iter=2000)
    glm.fit(sc.transform(logit(Ea)), ya)
    members["glm"][0][b] = glm.predict_proba(sc.transform(logit(Eb)))[:, 1]
    members["glm"][1][:] += glm.predict_proba(sc.transform(logit(Et)))[:, 1] / NSP
    log(f"fold {f}: te2 {roc_auc_score(y[b], members['te2'][0][b]):.5f} | "
        f"lgb {roc_auc_score(y[b], members['lgb'][0][b]):.5f} | glm {roc_auc_score(y[b], members['glm'][0][b]):.5f}")

for name, (oof, tst) in members.items():
    d = f"storage/artifacts/training/icU_{name}"; os.makedirs(d, exist_ok=True)
    pd.DataFrame({"id": tr["id"], "fold_id": fold_of, "is_fraud": y.astype(int), "y_prob": oof}).to_csv(f"{d}/oof.csv", index=False)
    pd.DataFrame({"id": te["id"], "is_fraud": tst}).to_csv(f"{d}/submission.csv", index=False)
    json.dump({"exp_name": f"icU_{name}", "n_keys": len(KNAMES), "oof_auc": roc_auc_score(y, oof)}, open(f"{d}/config.json", "w"))
    log(f"{name}: OOF {roc_auc_score(y, oof):.5f}")

# blend search over new + existing members
pool = {m: members[m][0] for m in members}
for ex in ["te1", "base", "champion"]:
    p = {"champion": "unified_feature_campaign/xgbU_tuned_av2_d5s"}.get(ex, ex)
    pool[ex] = pd.read_csv(f"storage/artifacts/training/{p}/oof.csv").sort_values("id")["y_prob"].to_numpy()
R = {k: pd.Series(v).rank(pct=True).to_numpy() for k, v in pool.items()}
def bl(ks): return roc_auc_score(y, np.mean([R[k] for k in ks], axis=0))
log("blends:")
for combo in [["te2"], ["lgb"], ["glm"], ["te2", "lgb", "glm"], ["te2", "glm"], ["te2", "lgb", "glm", "base"],
              ["te2", "lgb", "glm", "base", "champion"], ["te1", "te2", "lgb", "glm"]]:
    log(f"  {'+'.join(combo):40s} {bl(combo):.5f}")
