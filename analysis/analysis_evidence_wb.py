"""
analysis_evidence_wb.py

Two analyses on the final DSGAT3 checkpoints (Section 6.3 of the paper):

  1. Relational-evidence coverage (Table 4): test MRR of DS-GAT on the test
     triples WITH at least one supporting training fact and on those WITHOUT,
     using the same supports as the model.
  2. Role of W_B (ablation A1):
       - ratio ||W_B h_conf|| / ||h_neigh|| per layer (mean over nodes)
       - test MRR when the term W_B h_conf is removed at inference only
         (same trained weights), compared with the full model.

Usage (GPU node, one dataset per call; only evaluation, no training):
  python analysis/analysis_evidence_wb.py --dataset nl27k
Output: output/analysis_dsgat3/evidence_wb_<dataset>.json and a printed summary.
"""

import os
os.environ.setdefault("DSGAT_EVAL_ESTIMATOR", "mean")   # as in the final runs

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import glob
import json
import time
import numpy as np
import torch

from models_base.dsgat2 import EGATConv28
from models_base.dsgat3 import DSGAT3
from utils import calc_mrr
from analysis_dsgat3 import DATASETS, OUT_DIR, seed_of, build_model, forward


def layers_of(model):
    return [m for m in model.modules() if isinstance(m, EGATConv28)]


def patch_reduce(layer, stats, drop_wb):
    """Replace the reduce function of one layer: records the norms of the two
    streams and optionally drops W_B h_conf (inference-only ablation)."""
    def reduce_func(nodes):
        alpha = torch.softmax(nodes.mailbox['a'], dim=1)
        h_topo = torch.sum(alpha * nodes.mailbox['h'], dim=1)            # [n, H, d]
        lam_w = nodes.mailbox['lam_w']
        agg = ((lam_w * nodes.mailbox['m_b']).sum(dim=1) / (lam_w.sum(dim=1) + 1e-8))
        conf = layer.W_B(agg).unsqueeze(1).expand(-1, layer._num_heads, -1)
        stats["neigh"] += h_topo.norm(dim=-1).mean(dim=1).sum().item()
        stats["conf"] += conf.norm(dim=-1).mean(dim=1).sum().item()
        stats["n"] += h_topo.shape[0]
        return {'h': h_topo if drop_wb else h_topo + conf}
    layer.reduce_func = reduce_func


def mrr_on(model, data, emb, triplets, device):
    if len(triplets) == 0:
        return float("nan")
    score_fn = lambda h, r, t, s, c: model._score_for_eval_ids(h, r, t, s, c)
    score_fn.needs_ids = True
    mrr, _ = calc_mrr(emb, model.relation_embedding, triplets.to(device),
                      data["all_t"].to(device), hits=[1], score_func=score_fn)
    return float(mrr)


def run_dataset(ds, device):
    ckpts = sorted(glob.glob(f"output/DSGAT3_complex_{ds}_s*t_best_mrr_model.pth"), key=seed_of)
    print(f"[{ds}] DS-GAT checkpoints: {[seed_of(c) for c in ckpts]}", flush=True)
    res = {"dataset": ds, "seeds": [], "mrr_all": [], "mrr_sup": [], "mrr_nosup": [],
           "frac_sup": None, "ratio": [], "mrr_no_wb": []}
    for ck in ckpts:
        s = seed_of(ck)
        print(f"\n[{ds}] seed {s}", flush=True)
        model, data = build_model(DSGAT3, ck, ds, device)
        test = data["test_t"]
        with torch.no_grad():
            rows, _, _ = model._supports(test.to(device))
            has = torch.zeros(len(test), dtype=torch.bool)
            has[rows.unique().cpu()] = True
            res["frac_sup"] = float(has.float().mean())

            # full model, with the norm statistics of the two streams
            layers = layers_of(model)
            stats = [dict(neigh=0.0, conf=0.0, n=0) for _ in layers]
            for L, st in zip(layers, stats):
                patch_reduce(L, st, drop_wb=False)
            emb = forward(model, data, data["train_c"], device)
            res["ratio"].append([st["conf"] / max(st["neigh"], 1e-12) for st in stats])
            res["mrr_all"].append(mrr_on(model, data, emb, test, device))
            res["mrr_sup"].append(mrr_on(model, data, emb, test[has], device))
            res["mrr_nosup"].append(mrr_on(model, data, emb, test[~has], device))

            # same weights, W_B h_conf removed at inference
            for L in layers:
                patch_reduce(L, dict(neigh=0.0, conf=0.0, n=0), drop_wb=True)
            emb = forward(model, data, data["train_c"], device)
            res["mrr_no_wb"].append(mrr_on(model, data, emb, test, device))
        res["seeds"].append(s)
        print(f"  supported={res['frac_sup']:.3f}  MRR all={res['mrr_all'][-1]:.4f} "
              f"sup={res['mrr_sup'][-1]:.4f} nosup={res['mrr_nosup'][-1]:.4f}  "
              f"ratio(conf/neigh)={res['ratio'][-1]}  MRR without W_B h_conf={res['mrr_no_wb'][-1]:.4f}",
              flush=True)
        del model; torch.cuda.empty_cache()
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/evidence_wb_{ds}.json", "w") as f:
        json.dump(res, f, indent=1)
    ms = lambda x: f"{np.nanmean(x):.4f}+-{np.nanstd(x):.4f}"
    print(f"\n[{ds}] n={len(res['seeds'])}  supported test triples={res['frac_sup']:.3f}")
    print(f"  MRR all={ms(res['mrr_all'])}  supported={ms(res['mrr_sup'])}  "
          f"unsupported={ms(res['mrr_nosup'])}")
    r = np.array(res["ratio"])
    print("  ||W_B h_conf|| / ||h_neigh||: " +
          "  ".join(f"layer {i+1}={r[:, i].mean():.3f}+-{r[:, i].std():.3f}" for i in range(r.shape[1])))
    print(f"  MRR without W_B h_conf at inference={ms(res['mrr_no_wb'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if a.gpu >= 0 and torch.cuda.is_available() else "cpu")
    t0 = time.time()
    run_dataset(a.dataset, dev)
    print(f"elapsed {(time.time() - t0) / 60:.1f} min")
