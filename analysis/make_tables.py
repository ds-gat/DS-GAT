"""Rebuild the result tables of the paper from the CSV files in results/.

  python analysis/make_tables.py

Prints (mean ± std over runs):
  Table 1  link prediction (MRR, wMRR, Hits@1/3/10)   results/link_prediction/<model>/
  Table 2  confidence prediction MSE                   idem
  Table 3  ablations (seed 42)                         results/ablation/
  Table 5  hyperparameter sensitivity (seed 42)        results/sensitivity/
  Table 6  training time per run in hours              results/timing/<model>/
"""
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DATASETS = ["ppi5k", "cn15k", "nl27k"]
MODELS = ["GATv2", "EGAT", "WSGAT", "UKGE", "UPGAT", "PASSLEAF", "BEUrRE", "ssCDL", "DS-GAT"]


def overall(path):
    for row in csv.DictReader(open(path)):
        if row.get("relation") == "overall":
            return row
    return None


def dataset_of(path):
    return next(d for d in DATASETS if f"_{d}_" in os.path.basename(path))


def fmt(values, digits=3):
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.{digits}f}±{sd:.{digits}f}"


def table_link_prediction():
    print("Table 1: link prediction (mean±std over runs)")
    print(f"{'model':9s} {'dataset':7s} {'n':>2s}  {'MRR':12s} {'wMRR':12s} {'H@1':12s} {'H@3':12s} {'H@10':12s}")
    for m in MODELS:
        files = glob.glob(f"{ROOT}/link_prediction/{m}/metrics_*.csv")
        for ds in DATASETS:
            rows = [overall(f) for f in files if dataset_of(f) == ds]
            if not rows:
                continue
            cols = [fmt([float(r[k]) for r in rows]) for k in ["mrr", "wmrr", "hits@1", "hits@3", "hits@10"]]
            print(f"{m:9s} {ds:7s} {len(rows):2d}  " + " ".join(f"{c:12s}" for c in cols))
    print()


def table_confidence():
    print("Table 2: confidence prediction MSE (mean±std over runs)")
    for m in MODELS[3:]:
        files = glob.glob(f"{ROOT}/link_prediction/{m}/metrics_*.csv")
        cells = []
        for ds in DATASETS:
            vals = [float(overall(f)["mse"]) for f in files if dataset_of(f) == ds]
            cells.append(fmt(vals, 4))
        print(f"{m:9s} " + "  ".join(f"{ds}={c}" for ds, c in zip(DATASETS, cells)))
    print()


def table_ablation():
    print("Table 3: ablations (seed 42)")
    names = {"DSGAT3": "DS-GAT", "DSGAT3A1": "A1", "DSGAT3A2": "A2", "DSGAT3NE": "A3"}
    for tag, name in names.items():
        cells = []
        for ds in DATASETS:
            f = glob.glob(f"{ROOT}/ablation/metrics_{tag}_complex_{ds}_s42_*.csv")[0]
            r = overall(f)
            cells.append(f"{ds}: MRR={float(r['mrr']):.3f} wMRR={float(r['wmrr']):.3f} "
                         f"H@10={float(r['hits@10']):.3f} MSE={float(r['mse']):.4f}")
        print(f"{name:7s} " + " | ".join(cells))
    print()


def table_sensitivity():
    print("Table 5: sensitivity (seed 42; Val = best validation MRR)")
    for f in sorted(glob.glob(f"{ROOT}/sensitivity/metrics_*.csv")):
        tag = re.match(r"metrics_(.*)_s42_\d{8}_\d{6}\.csv", os.path.basename(f)).group(1)
        curves = sorted(glob.glob(f"{ROOT}/sensitivity/training_curves_{tag}_s42_*.csv"))
        val = max(float(r["val_mrr"]) for r in csv.DictReader(open(curves[-1])) if r["val_mrr"]) if curves else float("nan")
        r = overall(f)
        print(f"{tag:42s} Val={val:.3f} MRR={float(r['mrr']):.3f} MSE={float(r['mse']):.4f}")
    print()


def table_timing():
    print("Table 6: training time per run in hours (one value per seed: the last "
          "completed timing file, which accumulates resumed segments)")
    for m in MODELS:
        per_run = defaultdict(list)
        for f in glob.glob(f"{ROOT}/timing/{m}/timing_*.csv"):
            rows = list(csv.DictReader(open(f)))
            if not rows or rows[0].get("status", "completed") != "completed":
                continue
            seed = re.search(r"_s(\d+)_", os.path.basename(f)).group(1)
            per_run[(dataset_of(f), seed)].append((os.path.basename(f)[-19:], float(rows[0]["total_train_time_h"])))
        cells = []
        for ds in DATASETS:
            vals = [sorted(v)[-1][1] for k, v in per_run.items() if k[0] == ds]
            cells.append(f"{ds}={fmt(vals, 2)} (n={len(vals)})" if vals else f"{ds}=n/a")
        print(f"{m:9s} " + "  ".join(cells))
    print()


if __name__ == "__main__":
    table_link_prediction()
    table_confidence()
    table_ablation()
    table_sensitivity()
    table_timing()
