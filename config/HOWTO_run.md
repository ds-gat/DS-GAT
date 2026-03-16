# How to run DS-GAT experiments

`models.py` uses argparse — configs in this folder are reference documentation.
Copy the CLI command from the YAML header and adapt as needed.

---

## Environment setup

```bash
conda activate torchgpu
```

Install dependencies (only once):

```bash
# 1. PyTorch — match your CUDA version (example: CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. PyTorch Geometric
pip install torch-geometric

# 3. torch-scatter — must match torch + CUDA version
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cu118.html"

# 4. Everything else
pip install -r requirements.txt
```

---

## Main model — DSGAT2

### cn15k

```bash
python models.py \
  --gpu 0 --dataset cn15k --modelname DSGAT2 --weighted \
  --test-graph-size -1 --edge-weight-mode bayesian \
  --negative-sample 50 --embedding_dim 400 --score complex \
  --graph-batch-size -1 --lr 0.001 --lr-patience 100 \
  --lr-decay-factor 0.5 --min-lr 1e-5 \
  --early-stop-patience 20 --early-stop-delta 0.001 \
  --n-epochs 3000 --evaluate-every 300 \
  --numhops 2 --dropout 0.2 --nll-lambda 0.2
```

### nl27k

```bash
python models.py \
  --gpu 0 --dataset nl27k --modelname DSGAT2 --weighted \
  --test-graph-size -1 --edge-weight-mode bayesian \
  --negative-sample 50 --embedding_dim 500 --score complex \
  --graph-batch-size -1 --lr 0.001 --lr-patience 100 \
  --lr-decay-factor 0.5 --min-lr 1e-5 \
  --early-stop-patience 20 --early-stop-delta 0.001 \
  --n-epochs 3000 --evaluate-every 300 \
  --numhops 2 --dropout 0.35 --nll-lambda 0.2
```

### ppi5k

```bash
python models.py \
  --gpu 0 --dataset ppi5k --modelname DSGAT2 --weighted \
  --test-graph-size -1 --edge-weight-mode bayesian \
  --negative-sample 50 --embedding_dim 400 --score complex \
  --graph-batch-size -1 --lr 0.001 --lr-patience 100 \
  --lr-decay-factor 0.5 --min-lr 1e-5 \
  --early-stop-patience 20 --early-stop-delta 0.001 \
  --n-epochs 3000 --evaluate-every 300 \
  --dropout 0.4 --nll-lambda 0.2
```

---

## Ablation models — DSGATA1 / DSGATA2

Replace `--modelname DSGATA1` with `DSGATA2` for the other ablation.

### cn15k

```bash
python models.py \
  --gpu 0 --dataset cn15k --modelname DSGATA1 --weighted \
  --test-graph-size -1 --edge-weight-mode bayesian \
  --negative-sample 50 --embedding_dim 400 --score complex \
  --graph-batch-size -1 --lr 0.001 --lr-patience 100 \
  --lr-decay-factor 0.5 --min-lr 1e-5 \
  --early-stop-patience 20 --early-stop-delta 0.001 \
  --n-epochs 3000 --evaluate-every 300 \
  --numhops 2 --dropout 0.2 --nll-lambda 0.2
```

### nl27k

```bash
python models.py \
  --gpu 0 --dataset nl27k --modelname DSGATA1 --weighted \
  --test-graph-size -1 --edge-weight-mode bayesian \
  --negative-sample 50 --embedding_dim 500 --score complex \
  --graph-batch-size -1 --lr 0.001 --lr-patience 100 \
  --lr-decay-factor 0.5 --min-lr 1e-5 \
  --early-stop-patience 20 --early-stop-delta 0.001 \
  --n-epochs 3000 --evaluate-every 300 \
  --numhops 2 --dropout 0.35 --nll-lambda 0.2
```

### ppi5k

```bash
python models.py \
  --gpu 0 --dataset ppi5k --modelname DSGATA1 --weighted \
  --test-graph-size -1 --edge-weight-mode bayesian \
  --negative-sample 50 --embedding_dim 400 --score complex \
  --graph-batch-size -1 --lr 0.001 --lr-patience 100 \
  --lr-decay-factor 0.5 --min-lr 1e-5 \
  --early-stop-patience 20 --early-stop-delta 0.001 \
  --n-epochs 3000 --evaluate-every 300 \
  --numhops 2 --dropout 0.4 --nll-lambda 0.2
```

### DistMult instead of ComplEx

Add `--score dismult` to any command above.

---

## SLURM (cluster)

```bash
sbatch run_dsgat_cn15kc.sh
sbatch run_dsgat_nl27kc.sh
sbatch run_dsgat_ppi5kc.sh
```

---

## Output

Saved to `output/` (created automatically):

| File | Contents |
|---|---|
| `output/{model}_{score}_{dataset}t_best_mrr_model.pth` | Best checkpoint |
| `output/metrics_{model}_{score}_{dataset}_{timestamp}.csv` | Test metrics (MRR, Hits@K, wMRR) |
| `output/training_curves_{model}_{score}_{dataset}_{timestamp}.csv` | Training log |

---

## Parameter reference

| Argument | Default | Used in DS-GAT |
|---|---|---|
| `--modelname` | `RGCN` | `DSGAT2`, `DSGATA1`, `DSGATA2` |
| `--score` | `dismult` | `complex` (all DS-GAT runs) |
| `--embedding_dim` | `100` | 400 (cn15k, ppi5k) / 500 (nl27k) |
| `--numhops` | `2` | 2 |
| `--dropout` | `0.2` | 0.2 (cn15k) / 0.35 (nl27k) / 0.4 (ppi5k) |
| `--nll-lambda` | `0.1` | 0.2 |
| `--edge-weight-mode` | `bayesian` | `bayesian` |
| `--negative-sample` | `1` | 50 |
| `--graph-batch-size` | `30000` | -1 (all triplets) |
| `--test-graph-size` | `-1` | -1 (all train triplets) |
| `--weighted` | `false` | always set |
| `--early-stop-patience` | `20` | 20 eval intervals |
| `--evaluate-every` | `500` | 300 |
| `--lr-patience` | `500` | 100 |
| `--gpu` | `-1` | 0 |
