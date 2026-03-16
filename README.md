# DS-GAT

## Setup

```bash
# 1. PyTorch with CUDA (example: CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. PyTorch Geometric
pip install torch-geometric

# 3. torch-scatter (must match torch + CUDA version)
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cu118.html"

# 4. Remaining dependencies
pip install -r requirements.txt
```

---

## Running DS-GAT

All experiments use `models.py` with argparse. Configs in `config/` document the exact hyperparameters used.

### Main model (DSGAT2)

**cn15k**
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

**nl27k**
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

**ppi5k**
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

### Ablation models (DSGATA1, DSGATA2)

Replace `--modelname DSGATA1` with `DSGATA2` for the second ablation.

**cn15k**
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

**nl27k**
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

**ppi5k**
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

### On a SLURM cluster

```bash
sbatch run_dsgat_cn15kc.sh
sbatch run_dsgat_nl27kc.sh
sbatch run_dsgat_ppi5kc.sh
```

---

## Output

Results are saved to `output/` (created automatically):

| File | Contents |
|---|---|
| `output/{model}_{score}_{dataset}t_best_mrr_model.pth` | Best checkpoint |
| `output/metrics_{model}_{score}_{dataset}_{timestamp}.csv` | Test metrics (MRR, Hits@K) |
| `output/training_curves_{model}_{score}_{dataset}_{timestamp}.csv` | Training log |