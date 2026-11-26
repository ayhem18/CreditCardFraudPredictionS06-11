"""Evaluate the tabformer OOF member and its blend value against existing members.

Reads icU_tabformer/oof.csv (copied back from the server) plus the existing
icU_* members from storage/artifacts/training, all row-aligned on the frozen
alarm_profile folds, and prints: each member's OOF AUC, the Spearman
correlation of tabformer with each (lower = more diverse = better blend value),
and rank-mean blends with vs without tabformer.

Run: ./.venv/Scripts/python.exe -I experiments/tabformer/evaluate_blend.py
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

TRAIN = "storage/artifacts/training"
MEMBERS = {
    "tabformer": "icU_tabformer",
    "te1": "icU_te1",
    "te2": "icU_te2",
    "glm": "icU_glm",
    "lgb": "icU_lgb",
    "realmlp": "icU_realmlp",
    "base": "icU_base",
    "stack1": "icU_stack1",
    "champion": "unified_feature_campaign/xgbU_tuned_av2_d5s",
}


def main() -> None:
    loaded: dict[str, pd.Series] = {}
    y: pd.Series | None = None
    ids: pd.Index | None = None
    for name, rel in MEMBERS.items():
        path = f"{TRAIN}/{rel}/oof.csv"
        try:
            frame = pd.read_csv(path).sort_values("id").reset_index(drop=True)
        except FileNotFoundError:
            print(f"  (skip {name}: no oof at {path})")
            continue
        if ids is None:
            ids = frame["id"]
            y = frame["is_fraud"].astype(int)
        elif not frame["id"].equals(ids):
            print(f"  (skip {name}: id misalignment)")
            continue
        loaded[name] = frame["y_prob"].reset_index(drop=True)
    if "tabformer" not in loaded:
        raise SystemExit(f"icU_tabformer/oof.csv not found under {TRAIN}; copy it back first")

    print("\n== solo OOF AUC ==")
    for name, prob in loaded.items():
        print(f"  {name:12s} {roc_auc_score(y, prob):.5f}")

    ranks = {name: prob.rank(pct=True) for name, prob in loaded.items()}
    print("\n== Spearman corr of tabformer vs each (lower = more diverse) ==")
    for name in loaded:
        if name == "tabformer":
            continue
        print(f"  tabformer~{name:10s} {ranks['tabformer'].corr(ranks[name]):.4f}")

    def blend(names: list[str]) -> float:
        mean_rank = sum(ranks[n] for n in names) / len(names)
        return roc_auc_score(y, mean_rank)

    existing = [n for n in loaded if n != "tabformer"]
    print("\n== rank-mean blends ==")
    print(f"  tabformer solo                 {blend(['tabformer']):.5f}")
    if existing:
        print(f"  all existing ({len(existing)})               {blend(existing):.5f}")
        print(f"  all existing + tabformer       {blend(existing + ['tabformer']):.5f}")
        for n in existing:
            print(f"  {n}+tabformer{'':{max(0,14-len(n))}} {blend([n, 'tabformer']):.5f}")


if __name__ == "__main__":
    main()
