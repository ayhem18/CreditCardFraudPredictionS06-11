"""RealMLP member on the per-value + digit target encodings (GPU).

A neural-net reader of the same fingerprints as te1 (the TE-XGB). RealMLP
(pytabkit, tuned defaults) is maximally decorrelated from GBMs, so it is the
strongest diversity add for the stack. Same frozen alarm_profile outer folds
(from icU_base) so its OOF lines up and blends with te1 / glm.

Run: ./.venv/Scripts/python.exe -I experiments/target_encoding/build_realmlp.py
Writes storage/artifacts/training/icU_realmlp/.
"""
from __future__ import annotations
import json, os, time
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from pytabkit import RealMLP_TD_Classifier
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "storage/artifacts/training/icU_realmlp"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


log(f"device={DEVICE} (cuda avail={torch.cuda.is_available()})")
tr = pd.read_csv(get_train_csv_path()); te = pd.read_csv(get_test_csv_path())
y = tr["is_fraud"].to_numpy(np.float64); NTR = len(tr)
full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)


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


KEYS = key_table(full); KNAMES = list(KEYS)
codes = [pd.factorize(pd.Series(KEYS[n]))[0].astype(np.int32) for n in KNAMES]
KMAT = np.column_stack(codes); NCAT = np.array([c.max() + 1 for c in codes]); del KEYS, codes
RAW = np.column_stack(
    [full[c].values.astype(np.float32) for c in NUMS]
    + [pd.factorize(full[c], sort=True)[0].astype(np.float32) for c in CATS]
    + [full[c].astype(np.float32).values for c in FLAGS])
log(f"{len(KNAMES)} keys, {RAW.shape[1]} raw cols")

fold_src = pd.read_csv("storage/artifacts/training/icU_base/oof.csv").sort_values("id").reset_index(drop=True)
assert (fold_src["id"].to_numpy() == tr["id"].to_numpy()).all()
fold_of = fold_src["fold_id"].to_numpy(); NSP = int(fold_of.max() + 1)


def te_rate(k, t, ncat, prior):
    s = np.bincount(k, weights=t, minlength=ncat); n = np.bincount(k, minlength=ncat)
    return (s + prior * SMOOTH) / (n + SMOOTH)


Kt = KMAT[NTR:]
oof = np.zeros(NTR); test_pred = np.zeros(len(te))
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
    Xa = np.hstack([RAW[a], Ea]).astype(np.float32)
    Xb = np.hstack([RAW[b], Eb]).astype(np.float32)
    Xt = np.hstack([RAW[NTR:], Et]).astype(np.float32)
    t0 = time.time()
    m = RealMLP_TD_Classifier(device=DEVICE, random_state=42, n_epochs=64, batch_size=2048)
    m.fit(Xa, ya.astype(int))
    oof[b] = m.predict_proba(Xb)[:, 1]
    test_pred += m.predict_proba(Xt)[:, 1] / NSP
    log(f"fold {f}: auc {roc_auc_score(y[b], oof[b]):.5f} ({time.time()-t0:.0f}s)")

auc = roc_auc_score(y, oof)
log(f"RealMLP OOF AUC {auc:.5f}")
os.makedirs(OUT, exist_ok=True)
pd.DataFrame({"id": tr["id"], "fold_id": fold_of, "is_fraud": y.astype(int), "y_prob": oof}).to_csv(f"{OUT}/oof.csv", index=False)
pd.DataFrame({"id": te["id"], "is_fraud": test_pred}).to_csv(f"{OUT}/submission.csv", index=False)
json.dump({"exp_name": "icU_realmlp", "model": "RealMLP_TD", "device": DEVICE, "oof_auc": auc}, open(f"{OUT}/config.json", "w"))
log(f"wrote {OUT}")
