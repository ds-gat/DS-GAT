"""Coverage of the relational evidence on the test split, and a rule-only baseline.

For every test triple (h, r, t) we count its supporting TRAINING facts exactly as
DSGAT3 does (models_base/dsgat3.py): (h, r', t) with r' != r, or (t, r', h) for any r'.

Rule-only baseline: ranks the tails of (h, r, ?) with the evidence term alone,
    e(h, r, x) = sum_s c_s * g(r, r'_s, d_s),
where g is the empirical rule confidence measured on the training split
    g(r, r', d) = #pairs linked by r' in direction d that are also linked by r
                  / #pairs linked by r' in direction d.
No embeddings and no learning. Filtered ranking over all entities, ties broken
by the expected (mean) rank, tail prediction only, as in the paper.

Usage: python analysis/evidence_coverage.py
"""
import collections
import csv
import numpy as np

DATASETS = ['ppi5k', 'cn15k', 'nl27k']


def read(path):
    out = []
    with open(path) as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) >= 3:
                out.append((row[0], row[1], row[2], float(row[3]) if len(row) > 3 else 1.0))
    return out


for ds in DATASETS:
    train = read(f'data/{ds}/train.tsv')
    val = read(f'data/{ds}/val.tsv')
    test = read(f'data/{ds}/test.tsv')

    # supports indexed by entity: h -> list of (other, rel, dir, conf)
    by_ent = collections.defaultdict(list)
    pair_rels = collections.defaultdict(set)          # (a, b) -> relations a->b
    for h, r, t, c in train:
        by_ent[h].append((t, r, 0, c))
        by_ent[t].append((h, r, 1, c))
        pair_rels[(h, t)].add(r)

    # empirical rule confidence g(r, r', d)
    num = collections.Counter()
    den = collections.Counter()
    for (a, b), rels in pair_rels.items():
        rev = pair_rels.get((b, a), set())
        for rp in rels:                                 # r' in direction 0 on (a, b)
            den[(rp, 0)] += 1
            for r in rels:
                if r != rp:
                    num[(r, rp, 0)] += 1
        for rp in rev:                                  # r' in direction 1: (b, r', a)
            den[(rp, 1)] += 1
            for r in rels:
                num[(r, rp, 1)] += 1

    def g(r, rp, d):
        return num[(r, rp, d)] / den[(rp, d)] if den[(rp, d)] else 0.0

    entities = {e for tr in train + val + test for e in (tr[0], tr[2])}
    n_ent = len(entities)
    true_tails = collections.defaultdict(set)
    for h, r, t, _ in train + val + test:
        true_tails[(h, r)].add(t)

    has_sup = n_inv_same = 0
    rr_all, rr_sup, rr_nosup = [], [], []
    for h, r, t, _ in test:
        scores = collections.defaultdict(float)
        sup_t = False
        for other, rp, d, c in by_ent[h]:
            if d == 0 and rp == r:
                continue
            scores[other] += c * g(r, rp, d)
            if other == t:
                sup_t = True
        if (t, h) in pair_rels and r in pair_rels[(t, h)]:
            n_inv_same += 1
        has_sup += sup_t
        s_true = scores.get(t, 0.0)
        filt = true_tails[(h, r)] - {t}
        higher = sum(1 for x, s in scores.items() if s > s_true and x not in filt)
        equal_nz = sum(1 for x, s in scores.items() if s == s_true and x != t and x not in filt)
        if s_true == 0.0:                               # tie with every unscored candidate
            n_zero = n_ent - 1 - len(filt - {t}) - sum(1 for x, s in scores.items() if x != t and x not in filt and s != 0.0)
            ties = n_zero
        else:
            ties = equal_nz
        rank = 1 + higher + ties / 2.0
        rr_all.append(1.0 / rank)
        (rr_sup if sup_t else rr_nosup).append(1.0 / rank)

    n = len(test)
    print(f'{ds}: test={n}  with support={has_sup / n:.1%}  '
          f'reverse with same relation in train={n_inv_same / n:.1%}')
    print(f'   rule-only MRR overall={np.mean(rr_all):.3f}  '
          f'on supported={np.mean(rr_sup) if rr_sup else float("nan"):.3f}  '
          f'on unsupported={np.mean(rr_nosup) if rr_nosup else float("nan"):.3f}')
