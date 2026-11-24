# Running MFM pretraining on the server

The full-scale recipe is `run_recipes/pretrain/server.json`
(`mfm_v1_d768_l12`): bidirectional encoder, **d_model 768 × 12 layers ×
12 heads ≈ 85.4M params**, full train+test corpus (999,944 rows, label-free),
~21% token masking, **2-GPU DDP**, bf16-mixed, 30 epochs, batch 2048/GPU.

## 0. Prerequisites on the server

Three things must be present at the right places:

1. **The project** — this `ccf_tabformer/` tree.
2. **`dl_toolbox`** — the editable dependency. `ccf_tabformer_dl/pyproject.toml`
   points at it with `dl-toolbox = { path = "../../../../dl_toolbox", editable = true }`
   (i.e. `<...>/DL_projects/dl_toolbox` relative to the package dir). Put
   `dl_toolbox/` at that relative location, **or** edit that one line to the
   actual path on the server (absolute paths are fine), **or** install it from
   git. Without it `uv sync` cannot resolve the dependency.
3. **The token export** — the 103 MB folder with `meta.json` + `*.npy`, placed
   at `ccf_tabformer/data/tokens/` (the recipe's `tokens_dir: "data/tokens"` is
   resolved relative to the project root). Copy it from the machine where
   Phase A ran:
   `kaggle_comps/CreditCardFraud/ccf_tabformer/data/tokens/`  (or the original
   `kaggle_comps/CreditCardFraud/storage/artifacts/tabformer/tokens/`).
   The raw CSVs are **not** needed for pretraining — only this folder is.

## 1. Build the venv (Linux → cu124 torch, chosen automatically)

```bash
cd ccf_tabformer/ccf_tabformer_dl
uv venv ../.venv --python 3.12
VIRTUAL_ENV="$(cd .. && pwd)/.venv" uv sync --active
```

## 2. Launch pretraining (2-GPU DDP, single command)

```bash
cd ccf_tabformer
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python scripts/pretrain.py server.json
```

Lightning spawns the two DDP workers itself; the engine binds the 2 freest
CUDA devices. Add `--force` only to overwrite a previous run of the same name.

## 3. Watch it

- MLflow UI: `.venv/bin/python scripts/mlflow_run.py --host 0.0.0.0 --port 5000`
  then open the host:5000. Metric to watch: `recon_acc/train/epoch/mean`
  (monitored, mode=max).
- Checkpoints: `storage/run_artifacts/pretrain/v1/mfm_v1_d768_l12/checkpoints/`
  (`save_top_k=1` best + `save_last`). The `.ckpt` is what the fine-tune /
  linear-probe step loads.

## 4. Sanity probe after it finishes

```bash
.venv/bin/python scripts/linear_probe.py \
  --recipe run_recipes/pretrain/server.json \
  --checkpoint storage/run_artifacts/pretrain/v1/mfm_v1_d768_l12/checkpoints/live/<id>.ckpt
```

Prints the frozen-encoder out-of-fold fraud AUC (the local 2-layer smoke model
already gave 0.892; a real run should do better).

## 5. Scaling headroom

85M params + batch 2048 use a small fraction of 2×60 GB. If you want more
capacity, raise in `server.json` → `experiment_config.model`: `d_model`
(1024), `n_layers` (16–24), and/or `data.batch_size` (4096–8192), and bump
`optimizer.learning_rate` roughly with batch size. Keep `n_heads` dividing
`d_model`.
