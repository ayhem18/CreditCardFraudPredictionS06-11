# Running MFM pretraining on the server

The full-scale recipe is `run_recipes/supervised_reconstruction/pretrain/server.json`
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
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python scripts/unsupervised_reconstruction/pretrain.py server.json
```

Lightning spawns the two DDP workers itself; the engine binds the 2 freest
CUDA devices. Add `--force` only to overwrite a previous run of the same name.

## 3. Watch it

- MLflow UI: `.venv/bin/python scripts/tracking/mlflow_run.py --host 0.0.0.0 --port 5000`
  then open the host:5000. Metric to watch: `recon_acc/train/epoch/mean`
  (monitored, mode=max).
- Checkpoints: `storage/run_artifacts/pretrain/v1/mfm_v1_d768_l12/checkpoints/`
  (`save_top_k=1` best + `save_last`). The `.ckpt` is what the fine-tune /
  linear-probe step loads.

## 4. Sanity probe after it finishes

```bash
.venv/bin/python scripts/fine_tune_classification/linear_probe.py \
  --recipe run_recipes/supervised_reconstruction/pretrain/server.json \
  --checkpoint storage/run_artifacts/pretrain/v1/mfm_v1_d768_l12/checkpoints/live/<id>.ckpt
```

Prints the frozen-encoder out-of-fold fraud AUC (the local 2-layer smoke model
already gave 0.892; a real run should do better).

## 5. Fold-0 fine-tune pilot (the decision gate)

Once pretraining has a checkpoint, run the three regimes on fold 0 to decide
whether pretraining earns its keep before spending on all 5 folds. Point them
at the pretrained checkpoint with `--checkpoint` (its filename is a hash):

```bash
cd ccf_tabformer
CKPT=storage/run_artifacts/pretrain/v1/mfm_v1_d768_l12/checkpoints/live/<id>.ckpt

.venv/bin/python scripts/fine_tune_classification/finetune.py pilot_fold0_pretrained.json --checkpoint "$CKPT"
.venv/bin/python scripts/fine_tune_classification/finetune.py pilot_fold0_random.json
.venv/bin/python scripts/fine_tune_classification/finetune.py pilot_fold0_frozen.json     --checkpoint "$CKPT"
```

- Fine-tuning runs **single-GPU** (so the val-AUC metric is the whole fold, no
  DDP shard-gathering). The two GPUs can run two regimes at once via
  `CUDA_VISIBLE_DEVICES=0` / `=1`.
- Monitored metric: `auc/val/epoch/auc` (mode=max). Read the best per run from
  its `checkpoints/catalog.json` (`current.files[0].score`) or MLflow.

**Decision gate:** proceed to the full 5-fold (Phase D: OOF + submission +
blend) only if `pretrained` ≳ `random` (pretraining helps, or at least a
useful decorrelated member) and the val AUC is competitive (≈ ≥ 0.943, near
RealMLP's per-fold). `frozen` is the cheap linear-probe-style floor.

## 6. Full 5-fold + OOF/submission + blend (Phase D)

Once the pilot passes, train all 5 folds with the winning regime (default:
pretrained-finetune, `fivefold.json`), then assemble the OOF + submission.

```bash
cd ccf_tabformer
CKPT=storage/run_artifacts/pretrain/v1/mfm_v1_d768_l12/checkpoints/live/<id>.ckpt

for f in 0 1 2 3 4; do
  .venv/bin/python scripts/fine_tune_classification/finetune.py fivefold.json --fold "$f" --checkpoint "$CKPT"
done
# (two GPUs: run two folds at once with CUDA_VISIBLE_DEVICES=0 / =1)

.venv/bin/python scripts/fine_tune_classification/predict_oof.py \
  --recipe run_recipes/classification_fine_tune/full_run/fivefold.json \
  --runs-root storage/run_artifacts/finetune/fivefold \
  --out outputs/icU_tabformer
```

`predict_oof` prints **mean per-fold AUC** and **pooled OOF AUC** — they should
agree. A large gap (it warns) means the fold models aren't comparable across
folds (under-trained); train longer before trusting the OOF/blend.

Then copy `outputs/icU_tabformer/{oof.csv,submission.csv}` back to the
competition repo at `storage/artifacts/training/icU_tabformer/` and, in the
competition venv:

```powershell
./.venv/Scripts/python.exe -I experiments/tabformer/evaluate_blend.py
```

It prints the member's OOF AUC, its Spearman correlation with each existing
member (lower = more diverse = more blend value), and the rank-mean blend with
vs without tabformer. If it lifts the blend, submit `submission.csv` via
`scripts/kaggle_submit.py` (ask first).

## 7. Scaling headroom

85M params + batch 2048 use a small fraction of 2×60 GB. If you want more
capacity, raise in `server.json` → `experiment_config.model`: `d_model`
(1024), `n_layers` (16–24), and/or `data.batch_size` (4096–8192), and bump
`optimizer.learning_rate` roughly with batch size. Keep `n_heads` dividing
`d_model`.
