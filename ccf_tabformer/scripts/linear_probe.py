"""Linear-probe read of a pretrained encoder: does CLS carry fraud signal?

Loads a pretrained MFM checkpoint, extracts the (unmasked) CLS representation
for every train row, and fits a per-fold logistic regression on the frozen
alarm_profile folds -> out-of-fold ROC-AUC. This is the cheap gate on
representation quality before committing to full fine-tuning.

  python scripts/linear_probe.py --recipe <recipe.json> --checkpoint <ckpt>

NOTE: a probe AUC is only meaningful on a *real* pretraining run; on the tiny
local smoke checkpoint it just confirms the extraction + probe plumbing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ccf_tabformer_dl.data.tokens import load_tokenized_table
from ccf_tabformer_dl.model.network import MaskedFeatureModel, TabFormerConfig
from ccf_tabformer_dl.model.vocab import build_global_vocab, to_global_ids
from ccf_tabformer_dl.pretraining.config import PretrainConfig

EXTRACT_BATCH = 4096


def load_model(*, config: PretrainConfig, vocab, checkpoint: Path, device: torch.device) -> MaskedFeatureModel:
    net = MaskedFeatureModel(
        vocab=vocab,
        config=TabFormerConfig(
            d_model=config.model.d_model, n_layers=config.model.n_layers,
            n_heads=config.model.n_heads, ffn_mult=config.model.ffn_mult,
            dropout=config.model.dropout,
        ),
    )
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    stripped = {k[len("net.") :]: v for k, v in state.items() if k.startswith("net.")}
    net.load_state_dict(stripped, strict=True)
    return net.to(device).eval()


@torch.no_grad()
def extract_cls(*, net: MaskedFeatureModel, global_ids: np.ndarray, device: torch.device) -> np.ndarray:
    feats = []
    for start in range(0, len(global_ids), EXTRACT_BATCH):
        chunk = torch.as_tensor(global_ids[start : start + EXTRACT_BATCH], dtype=torch.long, device=device)
        cls_hidden, _ = net.encoder(tokens=chunk, masked=None)
        feats.append(cls_hidden.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def probe(*, features: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray) -> tuple[float, dict[int, float]]:
    n_folds = int(fold_ids.max()) + 1
    oof = np.zeros(len(labels))
    per_fold = {}
    for f in range(n_folds):
        tr = fold_ids != f
        va = fold_ids == f
        scaler = StandardScaler().fit(features[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(features[tr]), labels[tr])
        oof[va] = clf.predict_proba(scaler.transform(features[va]))[:, 1]
        per_fold[f] = float(roc_auc_score(labels[va], oof[va]))
    return float(roc_auc_score(labels, oof)), per_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear-probe a pretrained MFM encoder.")
    parser.add_argument("--recipe", required=True, help="Pretraining recipe JSON (for model config + tokens_dir).")
    parser.add_argument("--checkpoint", required=True, help="Path to the .ckpt to probe.")
    args = parser.parse_args()

    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    config = PretrainConfig.parse(recipe["experiment_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    table = load_tokenized_table(tokens_dir=Path(config.tokens_dir))
    vocab = build_global_vocab(table.specs)
    net = load_model(config=config, vocab=vocab, checkpoint=Path(args.checkpoint), device=device)

    train_global = to_global_ids(table.train_codes, vocab=vocab)
    features = extract_cls(net=net, global_ids=train_global, device=device)
    auc, per_fold = probe(features=features, labels=table.labels, fold_ids=table.fold_ids)

    print(f"[linear-probe] device={device} features={features.shape}")
    for f in sorted(per_fold):
        print(f"  fold {f}: AUC {per_fold[f]:.5f}")
    print(f"[linear-probe] OOF AUC {auc:.5f}")


if __name__ == "__main__":
    main()
