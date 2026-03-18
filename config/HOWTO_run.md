# How to run DS-GAT experiments

All experiments are launched through `models.py`. The recommended way is to pass a YAML config
file with `--config`; every value in the file becomes a default that any CLI argument can override.

---

## 1. Environment setup

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

## 2. Running with a config file (recommended)

```bash
python models.py --config <path/to/config.yaml>
```

Any value in the YAML can be overridden on the command line:

```bash
# Run cn15k config on GPU 1 with a different dropout
python models.py --config config/DSGAT2_cn15k.yaml --gpu 1 --dropout 0.3
```

### Main model — DSGAT2

Config files are in `config/`:

```bash
python models.py --config config/DSGAT2_cn15k.yaml
python models.py --config config/DSGAT2_nl27k.yaml
python models.py --config config/DSGAT2_ppi5k.yaml
```

### Ablation models — DSGATA1 / DSGATA2

Config files are in `config/ablation/`:

```bash
python models.py --config config/ablation/DSGATA1_cn15k.yaml
python models.py --config config/ablation/DSGATA1_nl27k.yaml
python models.py --config config/ablation/DSGATA1_ppi5k.yaml

python models.py --config config/ablation/DSGATA2_cn15k.yaml
python models.py --config config/ablation/DSGATA2_nl27k.yaml
python models.py --config config/ablation/DSGATA2_ppi5k.yaml
```

### GAT baseline models — EGAT / WSGAT / GATV2

Config files are in `config/GAT/`:

```bash
python models.py --config config/GAT/EGAT_cn15k.yaml
python models.py --config config/GAT/EGAT_nl27k.yaml
python models.py --config config/GAT/EGAT_ppi5k.yaml

python models.py --config config/GAT/WSGAT_cn15k.yaml
python models.py --config config/GAT/WSGAT_nl27k.yaml
python models.py --config config/GAT/WSGAT_ppi5k.yaml

python models.py --config config/GAT/GATV2_cn15k.yaml
python models.py --config config/GAT/GATV2_nl27k.yaml
python models.py --config config/GAT/GATV2_ppi5k.yaml
```

---

## 3. SLURM (cluster)

Each sbatch script runs EGAT, WSGAT, and GATV2 sequentially for one dataset,
loading hyperparameters from `config/GAT/`:

```bash
sbatch run_cn15k_baselines.sh
sbatch run_nl27k_baselines.sh
sbatch run_ppi5k_baselines.sh
```

Main model jobs:

```bash
sbatch run_dsgat_cn15kc.sh
sbatch run_dsgat_nl27kc.sh
sbatch run_dsgat_ppi5kc.sh
```

---

## 4. Output

All outputs are saved to `output/` (created automatically):

| File | Contents |
|---|---|
| `output/{model}_{score}_{dataset}t_best_mrr_model.pth` | Best checkpoint (highest validation MRR) |
| `output/metrics_{model}_{score}_{dataset}_{timestamp}.csv` | Test metrics (MRR, Hits@K, wMRR) |
| `output/training_curves_{model}_{score}_{dataset}_{timestamp}.csv` | Training log (loss, val MRR, time per epoch) |

---

## 5. Parameter reference

| Argument | Default | DS-GAT value |
|---|---|---|
| `--config` | `None` | path to YAML file |
| `--modelname` | `RGCN` | `DSGAT2`, `DSGATA1`, `DSGATA2`, `EGAT`, `WSGAT`, `GATV2` |
| `--score` | `dismult` | `complex` (DS-GAT) / `dismult` (GAT baselines) |
| `--embedding_dim` | `100` | 400 (cn15k, ppi5k) / 500 (nl27k) |
| `--numhops` | `2` | 2 |
| `--dropout` | `0.2` | 0.2 (cn15k DSGAT2) / 0.35 (nl27k, baselines) / 0.4 (ppi5k DSGAT2) |
| `--nll-lambda` | `0.1` | 0.2 (DS-GAT only) |
| `--edge-weight-mode` | `bayesian` | `bayesian` |
| `--weighted` | `false` | always `true` |
| `--negative-sample` | `1` | 50 |
| `--graph-batch-size` | `30000` | -1 (all triplets) |
| `--test-graph-size` | `-1` | -1 (all train triplets) |
| `--early-stop-patience` | `20` | 20 eval intervals |
| `--early-stop-delta` | `0.001` | 0.001 |
| `--evaluate-every` | `500` | 300 |
| `--lr` | `0.001` | 0.001 |
| `--lr-patience` | `500` | 100 |
| `--lr-decay-factor` | `0.5` | 0.5 |
| `--min-lr` | `1e-5` | 1e-5 |
| `--grad-norm` | `1.0` | 1.0 |
| `--seed` | `42` | 42 |
| `--gpu` | `-1` | 0 |
