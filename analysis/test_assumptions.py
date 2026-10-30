"""
Statistical tests for DS-GAT Bayesian assumptions.
Standalone — does NOT require torch_geometric, dgl, or torch_scatter.

A1+A3 — Test 1: Anderson-Darling normality on residuals c_ij - mu_ij
A1+A3 — Test 2: one-sample t-test: E[delta_theta(c_ij)] = 0
A2    — Test 3: Levene's test for unequal variances across relation types
A2    — Test 4: Spearman rho between learned var_scale (nu_r) and
                empirical per-relation variance of c_ij
"""

import argparse
import glob
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import anderson, levene, spearmanr, ttest_1samp, skew, kurtosis


# ── reconstruct MLPs from state_dict (no model import needed) ────────────────

def make_correction_mlp():
    return nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1), nn.Tanh())

def make_var_mlp():
    return nn.Sequential(
        nn.Linear(1, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, 1),  nn.Sigmoid()
    )

def load_conv_params(state_dict, layer=0):
    prefix = f"convs.{layer}."
    corr = make_correction_mlp()
    corr[0].weight = nn.Parameter(state_dict[prefix + "correction_mlp.0.weight"])
    corr[0].bias   = nn.Parameter(state_dict[prefix + "correction_mlp.0.bias"])
    corr[2].weight = nn.Parameter(state_dict[prefix + "correction_mlp.2.weight"])
    corr[2].bias   = nn.Parameter(state_dict[prefix + "correction_mlp.2.bias"])

    var_mlp = make_var_mlp()
    for i, j in [(0,0),(2,2),(4,4)]:
        var_mlp[i].weight = nn.Parameter(state_dict[prefix + f"edge_weight_var_mlp.{j}.weight"])
        var_mlp[i].bias   = nn.Parameter(state_dict[prefix + f"edge_weight_var_mlp.{j}.bias"])

    mean_scale = state_dict[prefix + "mean_scale"]   # [R]
    var_scale  = state_dict[prefix + "var_scale"]    # [R]
    return corr, var_mlp, mean_scale, var_scale


# ── load raw data from TSV ────────────────────────────────────────────────────

def load_tsv(path):
    """Returns (head, rel, tail, conf) — handles both numeric and string IDs."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                rows.append(parts)

    # Try numeric first
    try:
        arr   = np.array(rows, dtype=float)
        return arr[:,0].astype(int), arr[:,1].astype(int), \
               arr[:,2].astype(int), arr[:,3].astype(float)
    except ValueError:
        pass

    # String IDs: build vocab on the fly
    data_dir = os.path.dirname(path)
    h_vocab, r_vocab, t_vocab = {}, {}, {}

    # Try loading pre-built mappings if available
    rel_map_path = os.path.join(data_dir, "relation_id.csv")
    if os.path.exists(rel_map_path):
        import csv
        with open(rel_map_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                r_vocab[row.get("rel string", row.get("rel_string", ""))] = int(row["id"])

    heads, rels, tails, confs = [], [], [], []
    for parts in rows:
        h = h_vocab.setdefault(parts[0], len(h_vocab))
        r = r_vocab.get(parts[1], r_vocab.setdefault(parts[1], len(r_vocab)))
        t = t_vocab.setdefault(parts[2], len(t_vocab))
        heads.append(h); rels.append(r)
        tails.append(t); confs.append(float(parts[3]))

    return (np.array(heads), np.array(rels),
            np.array(tails), np.array(confs))


# ── compute Bayesian quantities ───────────────────────────────────────────────

@torch.no_grad()
def compute_quantities(conf_np, rel_np, corr_mlp, var_mlp, mean_scale, var_scale):
    ew  = torch.tensor(conf_np, dtype=torch.float32).unsqueeze(1)   # [N,1]
    et  = torch.tensor(rel_np,  dtype=torch.long)                    # [N]

    delta      = corr_mlp(ew)                                         # [N,1]
    mu_raw     = torch.clamp(ew + delta, 0.0, 1.0)
    mu_ij      = mu_raw * F.softplus(mean_scale[et]).unsqueeze(1)
    sigma2     = (var_mlp(ew) * F.softplus(var_scale[et]).unsqueeze(1) + 1e-6)

    return {
        "c_ij":      conf_np,
        "delta":     delta.squeeze(1).numpy(),
        "mu_ij":     mu_ij.squeeze(1).numpy(),
        "sigma2":    sigma2.squeeze(1).numpy(),
        "nu_r":      F.softplus(var_scale).numpy(),
        "edge_type": rel_np,
    }


# ── four statistical tests ────────────────────────────────────────────────────

def test1_anderson(q):
    residuals = q["c_ij"] - q["mu_ij"]
    res  = anderson(residuals, dist="norm")
    stat = res.statistic
    crit = res.critical_values[2]   # 5% level
    reject = bool(stat > crit)
    return dict(stat=stat, critical=crit, reject=reject,
                label="A1 Anderson-Darling (residuals ~ N)",
                verdict=("NOT Gaussian — A1 weakened" if reject
                         else "Consistent with Gaussian — A1 supported"))

def test2_ttest(q):
    tstat, pval = ttest_1samp(q["delta"], popmean=0.0)
    reject = bool(pval < 0.05)
    return dict(t=float(tstat), p=float(pval), mean_delta=float(q["delta"].mean()),
                reject=reject,
                label="A3 t-test E[delta]=0 (prior anchoring)",
                verdict=(f"delta mean={q['delta'].mean():.4f} != 0 — A3 weakened" if reject
                         else f"delta mean={q['delta'].mean():.4f} ~ 0 — A3 supported"))

def test3_levene(q):
    groups = [q["c_ij"][q["edge_type"] == r]
              for r in np.unique(q["edge_type"])
              if (q["edge_type"] == r).sum() >= 2]
    stat, pval = levene(*groups)
    reject = bool(pval < 0.05)
    return dict(stat=float(stat), p=float(pval), n_groups=len(groups),
                reject=reject,
                label="A2 Levene (unequal variances across relations)",
                verdict=("Variances differ by relation — A2 supported" if reject
                         else "No variance heterogeneity — A2 weakened"))

def test4_spearman(q):
    rels     = np.unique(q["edge_type"])
    emp_var  = np.array([q["c_ij"][q["edge_type"] == r].var() for r in rels])
    nu_sub   = q["nu_r"][rels]
    rho, p   = spearmanr(nu_sub, emp_var)
    sig      = bool(p < 0.05)
    reject   = bool(p < 0.05 and rho > 0)
    if sig and rho > 0:
        interp = f"rho={rho:.3f} p={p:.4f} — nu_r positively tracks variance — A2 supported"
    elif sig and rho < 0:
        interp = f"rho={rho:.3f} p={p:.4f} — nu_r inversely tracks variance (model compensates)"
    else:
        interp = f"rho={rho:.3f} p={p:.4f} — no significant correlation"
    return dict(rho=float(rho), p=float(p), reject=reject,
                label="A2 Spearman rho(nu_r, empirical var)",
                verdict=interp)

def test5_calibration(q):
    """Core: does sigma2_ij track |c_ij - mu_ij|?
    If core assumption holds, edges with larger residuals should get higher uncertainty."""
    residuals_abs = np.abs(q["c_ij"] - q["mu_ij"])
    sigma2 = q["sigma2"]
    rho, p = spearmanr(sigma2, residuals_abs)
    sig = bool(p < 0.05)
    supported = bool(sig and rho > 0)
    cv = float(sigma2.std() / (sigma2.mean() + 1e-9))
    if supported:
        interp = f"rho={rho:.3f} p={p:.4f} CV={cv:.3f} — sigma2 tracks actual noise — Core supported"
    elif sig and rho < 0:
        interp = f"rho={rho:.3f} p={p:.4f} CV={cv:.3f} — sigma2 inversely tracks noise (unexpected)"
    else:
        interp = f"rho={rho:.3f} p={p:.4f} CV={cv:.3f} — no calibration — Core weakened"
    return dict(rho=float(rho), p=float(p), cv=cv, reject=supported,
                label="Core: calibration rho(sigma2_ij, |residual_ij|)",
                verdict=interp)

def test6_gaussian_shape(q):
    """A1 shape: even if AD rejects, are skewness and kurtosis near Gaussian?
    Gaussian has skew=0, excess kurtosis=0."""
    residuals = q["c_ij"] - q["mu_ij"]
    sk = float(skew(residuals))
    ku = float(kurtosis(residuals, fisher=True))   # excess kurtosis
    approx = bool(abs(sk) < 0.5 and abs(ku) < 1.0)
    return dict(skewness=sk, kurtosis_excess=ku, reject=approx,
                label="A1 shape: skewness and excess kurtosis of residuals",
                verdict=(f"skew={sk:.3f} kurt={ku:.3f} — approximately Gaussian shape — A1 plausible"
                         if approx
                         else f"skew={sk:.3f} kurt={ku:.3f} — non-Gaussian shape (A1 is rough approximation)"))


# ── run one checkpoint ────────────────────────────────────────────────────────

def run_checkpoint(ckpt_path, data_path, layer=0):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt.get("state_dict", ckpt)
    corr_mlp, var_mlp, mean_scale, var_scale = load_conv_params(sd, layer)
    corr_mlp.eval(); var_mlp.eval()

    _, rels, _, conf = load_tsv(data_path)

    q = compute_quantities(conf, rels, corr_mlp, var_mlp, mean_scale, var_scale)

    return {
        "test1": test1_anderson(q),
        "test2": test2_ttest(q),
        "test3": test3_levene(q),
        "test4": test4_spearman(q),
        "test5": test5_calibration(q),
        "test6": test6_gaussian_shape(q),
    }


# ── aggregate and print ───────────────────────────────────────────────────────

def agg(results, key, field):
    vals = np.array([r[key][field] for r in results], dtype=float)
    return vals.mean(), vals.std()

def print_summary(results, dataset, layer):
    n = len(results)
    print(f"\n{'='*65}")
    print(f"RESULTS — {dataset}  layer={layer}  n={n} seeds")
    print(f"{'='*65}")

    for tkey in ["test1", "test2", "test3", "test4", "test5", "test6"]:
        r0 = results[0][tkey]
        print(f"\n{r0['label']}")
        rejects = sum(r[tkey]["reject"] for r in results)

        if tkey == "test1":
            m, s = agg(results, tkey, "stat")
            print(f"  stat  : {m:.4f} ± {s:.4f}")
        elif tkey == "test2":
            m, s = agg(results, tkey, "p")
            dm, ds = agg(results, tkey, "mean_delta")
            print(f"  p-val : {m:.4f} ± {s:.4f}   mean delta: {dm:.4f} ± {ds:.4f}")
        elif tkey == "test3":
            m, s = agg(results, tkey, "p")
            print(f"  p-val : {m:.4f} ± {s:.4f}   groups: {r0['n_groups']}")
        elif tkey == "test4":
            rm, rs = agg(results, tkey, "rho")
            pm, ps = agg(results, tkey, "p")
            print(f"  rho   : {rm:.4f} ± {rs:.4f}   p: {pm:.4f} ± {ps:.4f}")
        elif tkey == "test5":
            rm, rs = agg(results, tkey, "rho")
            pm, ps = agg(results, tkey, "p")
            cm, cs = agg(results, tkey, "cv")
            print(f"  rho   : {rm:.4f} ± {rs:.4f}   p: {pm:.4f} ± {ps:.4f}   CV: {cm:.3f} ± {cs:.3f}")
        elif tkey == "test6":
            sm, ss = agg(results, tkey, "skewness")
            km, ks = agg(results, tkey, "kurtosis_excess")
            print(f"  skew  : {sm:.4f} ± {ss:.4f}   excess kurt: {km:.4f} ± {ks:.4f}")

        print(f"  H0 rejected: {rejects}/{n} seeds")
        print(f"  → {r0['verdict']}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="cn15k")
    parser.add_argument("--data_dir",  default="data")
    parser.add_argument("--layer",     type=int, default=0)
    parser.add_argument("--checkpoints", nargs="+")
    args = parser.parse_args()

    data_path = f"{args.data_dir}/{args.dataset}/train.tsv"

    checkpoints = args.checkpoints or sorted(
        glob.glob(f"output/DSGAT2_complex_{args.dataset}*best_mrr_model.pth")
    )
    print(f"Found {len(checkpoints)} checkpoints:")
    for c in checkpoints:
        print(f"  {c}")

    results = []
    for ckpt in checkpoints:
        print(f"\nRunning {ckpt} ...")
        res = run_checkpoint(ckpt, data_path, args.layer)
        results.append(res)
        for v in res.values():
            print(f"  {v['label']}: {v['verdict']}")

    print_summary(results, args.dataset, args.layer)


if __name__ == "__main__":
    main()
