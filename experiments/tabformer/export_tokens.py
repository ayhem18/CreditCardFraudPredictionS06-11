"""Phase A — digit-sequence tokenizer export for the tabular-BERT (TABFORMER_PLAN_V2).

Builds, label-free, from train+test features:
- a fixed-length token-code matrix (one row per transaction): the 6 continuous
  columns written as zero-padded digit sequences (one token per digit slot),
  plus one whole-value token for each small-int / categorical / flag column.
- per-token metadata (kind, owning feature, slot, vocab, allowed support).
- fold ids (from the frozen alarm_profile OOF) + labels for the train rows.

Everything here is pandas-only so it runs in the competition venv and ships to
the training server with storage/raw_data. The reconstruction heads later use
the per-token "allowed" support to mask impossible indices to -inf.

Run: ./.venv/Scripts/python.exe -I experiments/tabformer/export_tokens.py
Writes storage/artifacts/tabformer/tokens/{token_codes.npy,meta.json,
       fold_ids.npy,labels.npy,row_ids.npy}.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from ccf_ml.file_utils import get_train_csv_path, get_test_csv_path

CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
FLAGS = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
         "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
# continuous columns that become fixed-width digit sequences
FLOATS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km",
          "account_balance_usd", "velocity_score", "merchant_risk_score"]
# remaining numerics: small ranges -> one whole-value token each
SMALLINTS = ["txn_count_last_24h", "card_age_months", "customer_age",
             "cvv_retry_count", "time_of_day_hour", "day_of_week", "prior_disputes"]

FEATS = FLOATS + SMALLINTS + CATS + FLAGS
OUT = "storage/artifacts/tabformer/tokens"
FOLD_SRC = "storage/artifacts/training/icU_base/oof.csv"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def split_float_string(s: str) -> tuple[str, str]:
    """('61.16') -> ('61', '16'); ('3233.0') -> ('3233', '0'); ('7') -> ('7', '')."""
    if "." in s:
        a, b = s.split(".", 1)
        return a, b
    return s, ""


def main() -> None:
    tr = pd.read_csv(get_train_csv_path(), dtype=str)
    te = pd.read_csv(get_test_csv_path(), dtype=str)
    n_tr, n_te = len(tr), len(te)
    log(f"train {n_tr} test {n_te}")

    full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
    n = len(full)

    token_cols: list[np.ndarray] = []   # each (n,) int value-code for one token position
    specs: list[dict] = []              # per-token spec, same order as token_cols
    feature_meta: dict[str, dict] = {}  # per-feature info (widths / value maps)

    # ---- continuous columns -> zero-padded digit tokens ----
    for c in FLOATS:
        s = full[c].astype(str)
        parts = s.map(split_float_string)
        int_parts = parts.map(lambda p: p[0])
        dec_parts = parts.map(lambda p: p[1])
        int_w = int(int_parts.map(len).max())
        dec_w = int(dec_parts.map(len).max())
        int_pad = int_parts.str.zfill(int_w)
        dec_pad = dec_parts.str.ljust(dec_w, "0")
        # lossless check: padding must not change the numeric value
        recon = (int_pad + "." + dec_pad).astype(float).to_numpy()
        orig = s.astype(float).to_numpy()
        if not np.allclose(recon, orig, rtol=0, atol=0):
            bad = int(np.argmax(~np.isclose(recon, orig, rtol=0, atol=0)))
            raise ValueError(f"{c}: padding changed value at row {bad}: {s.iloc[bad]} -> {recon[bad]}")
        digits = int_pad + dec_pad  # a string of (int_w + dec_w) digit chars
        width = int_w + dec_w
        if not (digits.map(len) == width).all():
            raise ValueError(f"{c}: inconsistent digit width")
        feature_meta[c] = {"kind": "digit", "int_width": int_w, "dec_width": dec_w, "n_slots": width}
        mat = np.frombuffer("".join(digits.to_numpy()).encode("ascii"), dtype=np.uint8).reshape(n, width) - ord("0")
        for slot in range(width):
            col = mat[:, slot].astype(np.int16)
            allowed = sorted(int(v) for v in np.unique(col))
            token_cols.append(col)
            specs.append({"name": f"{c}@{slot}", "feature": c, "kind": "digit",
                          "slot": slot, "vocab_size": 10, "allowed": allowed})
        log(f"{c:24s} int_w={int_w} dec_w={dec_w} -> {width} digit tokens")

    # ---- small-int columns -> one whole-value token ----
    for c in SMALLINTS:
        v = full[c].astype(float).round().astype(np.int64)
        uniq = sorted(int(x) for x in np.unique(v))
        code_of = {val: i for i, val in enumerate(uniq)}
        col = v.map(code_of).to_numpy().astype(np.int16)
        feature_meta[c] = {"kind": "smallint", "values": uniq}
        token_cols.append(col)
        specs.append({"name": c, "feature": c, "kind": "smallint", "slot": 0,
                      "vocab_size": len(uniq), "allowed": list(range(len(uniq)))})
        log(f"{c:24s} smallint vocab={len(uniq)} ({uniq[:6]}{'...' if len(uniq) > 6 else ''})")

    # ---- categorical columns -> one token ----
    for c in CATS:
        uniq = sorted(full[c].astype(str).unique())
        code_of = {val: i for i, val in enumerate(uniq)}
        col = full[c].astype(str).map(code_of).to_numpy().astype(np.int16)
        feature_meta[c] = {"kind": "cat", "values": uniq}
        token_cols.append(col)
        specs.append({"name": c, "feature": c, "kind": "cat", "slot": 0,
                      "vocab_size": len(uniq), "allowed": list(range(len(uniq)))})
        log(f"{c:24s} cat vocab={len(uniq)}")

    # ---- flags -> one token (0/1) ----
    for c in FLAGS:
        raw = full[c].astype(str).str.strip().str.lower()
        mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
        if not raw.isin(mapping).all():
            raise ValueError(f"{c}: unexpected flag values {sorted(raw.unique())}")
        col = raw.map(mapping).to_numpy().astype(np.int16)
        feature_meta[c] = {"kind": "flag", "values": [0, 1]}
        token_cols.append(col)
        specs.append({"name": c, "feature": c, "kind": "flag", "slot": 0,
                      "vocab_size": 2, "allowed": sorted(int(v) for v in np.unique(col))})

    token_codes = np.column_stack(token_cols).astype(np.int16)
    n_tokens = token_codes.shape[1]
    log(f"token matrix {token_codes.shape} (feature tokens; CLS is added at model build)")

    # ---- folds (train) from the frozen alarm_profile OOF, row-aligned ----
    fold_src = pd.read_csv(FOLD_SRC).sort_values("id").reset_index(drop=True)
    tr_ids = tr["id"].astype(np.int64).to_numpy()
    if not (fold_src["id"].to_numpy() == tr_ids).all():
        raise ValueError("fold source misaligned with train ids")
    fold_ids = fold_src["fold_id"].to_numpy().astype(np.int8)
    labels = tr["is_fraud"].astype(float).round().astype(np.int8).to_numpy()
    row_ids = np.concatenate([tr_ids, te["id"].astype(np.int64).to_numpy()])

    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/token_codes.npy", token_codes)
    np.save(f"{OUT}/fold_ids.npy", fold_ids)
    np.save(f"{OUT}/labels.npy", labels)
    np.save(f"{OUT}/row_ids.npy", row_ids)
    meta = {
        "n_total": n, "n_train": n_tr, "n_test": n_te, "n_tokens": int(n_tokens),
        "feature_order": FEATS, "floats": FLOATS, "smallints": SMALLINTS,
        "cats": CATS, "flags": FLAGS, "feature_meta": feature_meta, "tokens": specs,
        "fold_source": FOLD_SRC, "n_folds": int(fold_ids.max() + 1),
    }
    with open(f"{OUT}/meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    log(f"wrote {OUT}/ (token_codes, meta, fold_ids, labels, row_ids)")

    # ---- sanity gate: decode a sample of digit tokens back to the raw value ----
    _roundtrip_check(tr, te, token_codes, specs, feature_meta, n_tr)
    log("round-trip gate PASSED")


def _roundtrip_check(tr, te, token_codes, specs, feature_meta, n_tr):
    rng = np.random.default_rng(0)
    full_raw = pd.concat([tr[FLOATS], te[FLOATS]], ignore_index=True)
    sample = rng.choice(token_codes.shape[0], size=20000, replace=False)
    slot_cols = {}
    for j, sp in enumerate(specs):
        if sp["kind"] == "digit":
            slot_cols.setdefault(sp["feature"], {})[sp["slot"]] = j
    for c, slots in slot_cols.items():
        fm = feature_meta[c]
        iw, dw = fm["int_width"], fm["dec_width"]
        order = [slots[s] for s in range(iw + dw)]
        digit_mat = token_codes[np.ix_(sample, order)]
        decoded = []
        for row in digit_mat:
            ds = "".join(str(int(d)) for d in row)
            decoded.append(float(ds[:iw] + "." + ds[iw:]) if dw else float(ds))
        decoded = np.array(decoded)
        orig = full_raw[c].to_numpy()[sample].astype(float)
        if not np.allclose(decoded, orig, rtol=0, atol=0):
            bad = int(np.argmax(~np.isclose(decoded, orig, rtol=0, atol=0)))
            raise ValueError(f"round-trip mismatch {c}: {orig[bad]} vs {decoded[bad]}")


if __name__ == "__main__":
    main()
