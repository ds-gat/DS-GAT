"""
analysis_dsgat3.py

Analyses of Section 6 of the paper on the final model DSGAT3:

  1. Posterior calibration: corr(c, mu) and corr(c, sigma^2) per layer
  2. Stream independence: corr(alpha_ij, lambda_r w_ij) per layer
  3. Robustness: test MRR when Gaussian noise is added to the confidence of
     the training facts used at inference. The same noisy confidences feed
     message passing AND the relational evidence (supports), so the noise
     reaches every place where DSGAT3 reads an observed confidence.
     DS-GAT (all seeds) vs DSGAT3A2 (no Bayesian learning).

Two steps:
  python analysis/analysis_dsgat3.py --dataset cn15k        # GPU, one dataset -> npz
  python analysis/analysis_dsgat3.py --aggregate            # CPU, npz -> figures + summary

H1 (Levene, Spearman nu_r) is run separately with analysis/test_assumptions.py.
Run from the repository root; checkpoints are read from output/ (written by models.py).
"""

import os
os.environ.setdefault("DSGAT_EVAL_ESTIMATOR", "mean")   # as in the final runs

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse
import glob
import json
import re
import time
import numpy as np
import torch

from models_base.dsgat3 import DSGAT3
from models_base.dsgat3_ablation import DSGAT3A2
from utils import load_data, build_test_graph_directed, calc_mrr

DATASETS = ["ppi5k", "cn15k", "nl27k"]
EMB_DIM = {"ppi5k": 400, "cn15k": 400, "nl27k": 500}
NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
OUT_DIR = "output/analysis_dsgat3"


def seed_of(path):
    return int(re.search(r"_s(\d+)t_best_mrr_model\.pth$", path).group(1))


def build_model(cls, ckpt_path, dataset, device):
    entity2id, relation2id, (train_t, train_c), (valid_t, _), (test_t, _) = \
        load_data(f"data/{dataset}")
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    n_ent = sd["entity_embedding.weight"].shape[0] if "entity_embedding.weight" in sd \
        else len(set(entity2id.values()))
    node_features = torch.zeros(n_ent, EMB_DIM[dataset])   # overwritten by the state dict
    model = cls(node_features=node_features, num_relations=len(relation2id),
                embedding_dim=EMB_DIM[dataset], dropout=0.0, num_layers=2,
                score_function="complex", beta_lambda=0.1)
    model.load_state_dict(sd)
    model.set_support_graph(train_t, train_c, num_entities=n_ent)
    model.to(device).eval()
    data = dict(train_t=train_t, train_c=np.asarray(train_c, dtype=np.float32),
                test_t=torch.LongTensor(test_t),
                all_t=torch.LongTensor(np.concatenate((train_t, valid_t, test_t))),
                n_ent=n_ent, n_rel=len(relation2id))
    return model, data


def forward(model, data, conf, device):
    g = build_test_graph_directed(data["n_ent"], data["n_rel"], data["train_t"], conf)
    return model(g.entity.to(device), g.edge_index.to(device), g.edge_type.to(device),
                 edge_weight=g.edge_weight.to(device))


def test_mrr(model, data, conf, device):
    model.set_support_graph(data["train_t"], conf, num_entities=data["n_ent"])
    emb = forward(model, data, conf, device)
    score_fn = lambda h, r, t, s, c: model._score_for_eval_ids(h, r, t, s, c)
    score_fn.needs_ids = True
    mrr, _ = calc_mrr(emb, model.relation_embedding, data["test_t"].to(device),
                      data["all_t"].to(device), hits=[1, 3, 10], score_func=score_fn)
    return float(mrr)


def robustness(model, data, device):
    out = {}
    for sigma in NOISE_LEVELS:
        rng = np.random.default_rng(1000 + int(sigma * 100))   # same noise for every model
        c = data["train_c"]
        noisy = c if sigma == 0 else np.clip(c + rng.normal(0, sigma, len(c)), 0, 1).astype(np.float32)
        out[sigma] = test_mrr(model, data, noisy, device)
        print(f"    sigma={sigma:.2f}  MRR={out[sigma]:.4f}", flush=True)
    model.set_support_graph(data["train_t"], data["train_c"], num_entities=data["n_ent"])
    return out


def run_dataset(ds, device, skip_robustness=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpts = sorted(glob.glob(f"output/DSGAT3_complex_{ds}_s*t_best_mrr_model.pth"), key=seed_of)
    a2 = sorted(glob.glob(f"output/DSGAT3A2_complex_{ds}_s*t_best_mrr_model.pth"), key=seed_of)
    print(f"[{ds}] DS-GAT checkpoints: {[seed_of(c) for c in ckpts]}  A2: {[seed_of(c) for c in a2]}")
    res = {"dataset": ds, "seeds": [], "rob": {}, "rob_a2": {}, "corr_mu": [], "corr_var": [],
           "corr_stream": []}
    mu_all, var_all = [], []
    for ck in ckpts:
        s = seed_of(ck)
        print(f"\n[{ds}] DS-GAT seed {s}", flush=True)
        model, data = build_model(DSGAT3, ck, ds, device)
        with torch.no_grad():
            forward(model, data, data["train_c"], device)
            c = data["train_c"]
            mu = [m.squeeze(-1).cpu().numpy() for m, _ in model._conf_dist_cache]
            var = [v.squeeze(-1).cpu().numpy() for _, v in model._conf_dist_cache]
            res["corr_mu"].append([float(np.corrcoef(c, m)[0, 1]) for m in mu])
            res["corr_var"].append([float(np.corrcoef(c, v)[0, 1]) for v in var])
            # same statistic as banalisisb_multi.plot_stream_heatmaps_multi
            corr_st = []
            for lc in model._stream_cache:
                al, lw = lc["alpha"].cpu().numpy(), lc["lam_w"].cpu().numpy()
                n = min(len(al), len(lw))
                corr_st.append(float(np.corrcoef(np.stack([al[:n], lw[:n]], 0))[0, 1]))
            res["corr_stream"].append(corr_st)
            mu_all.append(np.stack(mu)); var_all.append(np.stack(var))
            print(f"  corr(c,mu)={res['corr_mu'][-1]}  corr(c,var)={res['corr_var'][-1]}  "
                  f"corr(streams)={res['corr_stream'][-1]}", flush=True)
            if not skip_robustness:
                res["rob"][s] = robustness(model, data, device)
        res["seeds"].append(s)
        del model; torch.cuda.empty_cache()
    if not skip_robustness:
        for ck in a2:
            s = seed_of(ck)
            print(f"\n[{ds}] A2 seed {s}", flush=True)
            model, data = build_model(DSGAT3A2, ck, ds, device)
            with torch.no_grad():
                res["rob_a2"][s] = robustness(model, data, device)
            del model; torch.cuda.empty_cache()
    with open(f"{OUT_DIR}/results_{ds}.json", "w") as f:
        json.dump(res, f, indent=1)
    np.savez_compressed(f"{OUT_DIR}/arrays_{ds}.npz", conf=data["train_c"],
                        mu=np.stack(mu_all), var=np.stack(var_all))
    print(f"\n[{ds}] saved {OUT_DIR}/results_{ds}.json")


def aggregate():
    import plots as B
    res, arr = {}, {}
    for ds in DATASETS:
        p = f"{OUT_DIR}/results_{ds}.json"
        if os.path.exists(p):
            res[ds] = json.load(open(p))
            arr[ds] = np.load(f"{OUT_DIR}/arrays_{ds}.npz")
    ms = lambda x: (float(np.mean(x)), float(np.std(x)))
    print("\n=== Posterior calibration and stream independence (mean +- std over seeds) ===")
    for ds, r in res.items():
        for L in range(len(r["corr_mu"][0])):
            m, v, st = (ms([x[L] for x in r[k]]) for k in ("corr_mu", "corr_var", "corr_stream"))
            print(f"{ds:6s} layer {L+1}: corr(c,mu)={m[0]:.4f}+-{m[1]:.4f}  "
                  f"corr(c,sigma2)={v[0]:.3f}+-{v[1]:.3f}  corr(streamA,B)={st[0]:.3f}+-{st[1]:.3f}"
                  f"  (n={len(r['seeds'])})")
    print("\n=== Robustness: test MRR under noise ===")
    rob, rob_a2 = {}, {}
    for ds, r in res.items():
        if not r["rob"]:
            continue
        rob[ds] = [{float(k): v for k, v in d.items()} for d in r["rob"].values()]
        rob_a2[ds] = [{float(k): v for k, v in d.items()} for d in r["rob_a2"].values()]
        for name, runs in (("DS-GAT", rob[ds]), ("A2", rob_a2[ds])):
            if not runs:
                continue
            mat = np.array([[d[s] for s in NOISE_LEVELS] for d in runs])
            drop = 100 * (mat[:, 0] - mat[:, -1]) / mat[:, 0]
            print(f"{ds:6s} {name:6s} n={len(runs)}  " +
                  "  ".join(f"s={s}:{mat[:, i].mean():.4f}" for i, s in enumerate(NOISE_LEVELS)) +
                  f"  drop@0.3={drop.mean():.1f}%+-{drop.std():.1f}")
        if "42" in r["rob"] and r["rob_a2"]:
            d, a = r["rob"]["42"], next(iter(r["rob_a2"].values()))
            print(f"{ds:6s} seed 42 drop@0.3: DS-GAT {100*(d['0.0']-d['0.3'])/d['0.0']:.1f}%  "
                  f"A2 {100*(a['0.0']-a['0.3'])/a['0.0']:.1f}%")
    fig_dir = f"{OUT_DIR}/figures"
    B.plot_confidence_distributions({ds: a["conf"] for ds, a in arr.items()}, fig_dir)
    B.plot_bayesian_calibration_multi(
        {ds: [(a["conf"], list(a["mu"][i]), list(a["var"][i])) for i in range(len(a["mu"]))]
         for ds, a in arr.items()}, fig_dir)
    if rob:
        B.plot_robustness_multi(rob, rob_a2, fig_dir)
    print(f"\nFigures in {fig_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    if a.aggregate:
        aggregate()
    else:
        dev = torch.device(f"cuda:{a.gpu}" if a.gpu >= 0 and torch.cuda.is_available() else "cpu")
        t0 = time.time()
        run_dataset(a.dataset, dev, a.skip_robustness)
        print(f"elapsed {(time.time() - t0) / 60:.1f} min")
