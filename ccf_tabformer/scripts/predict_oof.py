"""Assemble the 5-fold OOF + mean-of-folds test submission (training_io format).

Given the per-fold fine-tune runs (named ``<base>_fold{f}``, one best checkpoint
each), this:
- for fold f, loads fold f's checkpoint and predicts its held-out rows -> OOF;
- predicts the test rows with every fold model and averages them -> submission.
Writes oof.csv / submission.csv / config.json so the member blends with the
existing icU_* members via the competition's rank-mean harness.

  python scripts/predict_oof.py --recipe run_recipes/finetune/fivefold.json \
      --runs-root storage/run_artifacts/finetune/fivefold

Copy the written oof.csv + submission.csv into the competition repo at
storage/artifacts/training/icU_tabformer/ to blend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from ccf_tabformer_dl.data_contract.tokens import load_tokenized_table
from ccf_tabformer_dl.task_engines.finetune.config import FineTuneConfig
from ccf_tabformer_dl.tabformer_components.model.classifier import FineTuneModel, HeadConfig, load_finetune_checkpoint
from ccf_tabformer_dl.tabformer_components.model.network import TabFormerConfig
from ccf_tabformer_dl.tabformer_components.model.vocab import build_global_vocab, to_global_ids

PREDICT_BATCH = 8192
EXP_NAME = "icU_tabformer"


def _best_checkpoint(run_dir: Path) -> Path:
    catalog = json.loads((run_dir / "checkpoints" / "catalog.json").read_text(encoding="utf-8"))
    files = catalog["current"]["files"]
    if not files:
        raise FileNotFoundError(f"no current checkpoint in {run_dir}")
    return run_dir / "checkpoints" / files[0]["path"]


def _build_model(config: FineTuneConfig, vocab) -> FineTuneModel:
    return FineTuneModel(
        vocab=vocab,
        encoder_config=TabFormerConfig(
            d_model=config.model.d_model, n_layers=config.model.n_layers,
            n_heads=config.model.n_heads, ffn_mult=config.model.ffn_mult,
            dropout=config.model.dropout,
        ),
        head_config=HeadConfig(hidden=config.head.hidden, dropout=config.head.dropout),
    )


@torch.no_grad()
def _predict(net: FineTuneModel, global_ids: np.ndarray, device: torch.device) -> np.ndarray:
    out = np.empty(len(global_ids), dtype=np.float64)
    for start in range(0, len(global_ids), PREDICT_BATCH):
        chunk = torch.as_tensor(global_ids[start : start + PREDICT_BATCH], dtype=torch.long, device=device)
        logit = net(tokens=chunk)
        out[start : start + PREDICT_BATCH] = torch.sigmoid(logit).float().cpu().numpy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble OOF + submission from fold checkpoints.")
    parser.add_argument("--recipe", required=True, help="Fine-tune recipe (model/head/tokens_dir).")
    parser.add_argument("--runs-root", required=True, help="Dir holding the <base>_fold{f} run dirs.")
    parser.add_argument("--out", default="outputs/icU_tabformer", help="Output dir for training_io files.")
    args = parser.parse_args()

    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    base = recipe["human_readable_name"]
    config = FineTuneConfig.parse(recipe["experiment_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    table = load_tokenized_table(tokens_dir=Path(config.tokens_dir))
    vocab = build_global_vocab(table.specs)
    train_global = to_global_ids(table.train_codes, vocab=vocab)
    test_global = to_global_ids(table.test_codes, vocab=vocab)
    n_folds = table.n_folds

    oof = np.full(table.n_train, np.nan, dtype=np.float64)
    test_pred = np.zeros(table.n_test, dtype=np.float64)
    runs_root = Path(args.runs_root)
    for f in range(n_folds):
        run_dir = runs_root / f"{base}_fold{f}"
        net = _build_model(config, vocab)
        load_finetune_checkpoint(net, checkpoint=_best_checkpoint(run_dir), map_location="cpu")
        net = net.to(device).eval()
        val_idx = np.flatnonzero(table.fold_ids == f)
        oof[val_idx] = _predict(net, train_global[val_idx], device)
        test_pred += _predict(net, test_global, device) / n_folds
        print(f"[predict] fold {f}: val_auc {roc_auc_score(table.labels[val_idx], oof[val_idx]):.5f}")

    if np.isnan(oof).any():
        raise RuntimeError("some train rows were never in a held-out fold")
    auc = float(roc_auc_score(table.labels, oof))
    per_fold = [
        float(roc_auc_score(table.labels[table.fold_ids == f], oof[table.fold_ids == f]))
        for f in range(n_folds)
    ]
    mean_per_fold = float(np.mean(per_fold))
    print(f"[predict] mean per-fold AUC {mean_per_fold:.5f} | pooled OOF AUC {auc:.5f}")
    if mean_per_fold - auc > 0.01:
        print(
            "[predict] WARNING: pooled << mean per-fold -> fold scores not comparable "
            "across folds (under-trained / miscalibrated models); the member will not "
            "blend well as-is. Expect agreement once the models converge."
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_ids = table.row_ids[: table.n_train]
    test_ids = table.row_ids[table.n_train :]
    pd.DataFrame(
        {"id": train_ids, "fold_id": table.fold_ids, "is_fraud": table.labels.astype(int), "y_prob": oof}
    ).to_csv(out / "oof.csv", index=False)
    pd.DataFrame({"id": test_ids, "is_fraud": test_pred}).to_csv(out / "submission.csv", index=False)
    (out / "config.json").write_text(
        json.dumps(
            {"exp_name": EXP_NAME, "model": "tabformer_mfm_finetune", "oof_auc": auc,
             "n_folds": n_folds, "base_run": base, "runs_root": str(runs_root)},
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"[predict] wrote {out}/oof.csv, submission.csv, config.json")


if __name__ == "__main__":
    main()
