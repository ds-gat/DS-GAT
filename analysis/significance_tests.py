"""Significance of the DS-GAT MRR gains (Table 1 caption and Section 6.1).

Two-tailed Welch t-test on the per-run MRR, Holm correction over the 24
(dataset, baseline) comparisons, Cohen's d, and a 95% bootstrap interval of
the MRR gap. Reads results/link_prediction/<model>/metrics_*.csv.

  python analysis/significance_tests.py
"""
import csv
import glob
import os

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "link_prediction")
DATASETS = ["ppi5k", "cn15k", "nl27k"]
BASELINES = ["GATv2", "EGAT", "WSGAT", "PASSLEAF", "UPGAT", "UKGE", "BEUrRE", "ssCDL"]


def mrrs(model, ds):
    out = []
    for f in glob.glob(f"{ROOT}/{model}/metrics_*.csv"):
        if f"_{ds}_" in os.path.basename(f):
            out += [float(r["mrr"]) for r in csv.DictReader(open(f)) if r["relation"] == "overall"]
    return np.array(out)


rng = np.random.default_rng(0)
res = []
for ds in DATASETS:
    a = mrrs("DS-GAT", ds)
    for b_name in BASELINES:
        b = mrrs(b_name, ds)
        _, p = stats.ttest_ind(a, b, equal_var=False)
        boot = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(10000)]
        d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        res.append([ds, b_name, len(a), len(b), a.mean() - b.mean(), *np.percentile(boot, [2.5, 97.5]), d, p])

ps = np.array([r[-1] for r in res])
holm, running = np.empty(len(ps)), 0.0
for i, j in enumerate(np.argsort(ps)):
    running = max(running, min(1.0, (len(ps) - i) * ps[j]))
    holm[j] = running

print(f"{'dataset':7s} {'baseline':9s} nA nB  gap      95% CI               d      p        p_Holm")
for r, h in zip(res, holm):
    print(f"{r[0]:7s} {r[1]:9s} {r[2]:2d} {r[3]:2d} {r[4]:+.4f} [{r[5]:+.4f}, {r[6]:+.4f}] {r[7]:6.1f} {r[8]:.2e} {h:.2e}")
print(f"\nlargest Holm-corrected p = {holm.max():.2e}; smallest Cohen's d = {min(r[7] for r in res):.1f}")
