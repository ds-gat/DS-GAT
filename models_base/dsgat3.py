"""DSGAT3 — final DS-GAT model (the model reported in the paper).

  1. Encoder: dual-stream GAT with Bayesian edge weights and KL term
     (inherited from DSGAT2).

  2. Ranking score = ComplEx(h, r, t) + relational evidence
         e(h, r, t) = sum_i c_i * softplus(W[r, r'_i, dir_i])
     over the TRAINING facts i that connect the two entities,
     (h, r', t) (dir 0) or (t, r', h) (dir 1); a triple never supports itself.
     c_i is the observed confidence of fact i. W (Gamma in the paper) is
     trained with its own, larger learning rate (models.py, --evidence-lr):
     with the base lr it stays close to its initial value.

  3. Confidence: a Beta(alpha, beta) per triple, trained with the Beta NLL of
     the observed confidence on positives. Its inputs are detached from the
     encoder, so confidence training never changes the ranking.
         cbar   = sum_i a_i c_i,      a_i = softmax_i(v_i), v_i = c_i softplus(W)
         lambda = sigmoid(g[r] + k[r] * log1p(sum_i v_i))    (0 if no support)
         mu     = (1 - lambda) * sigmoid(MLP_mu(x) + b_mu[r]) + lambda * cbar
         kappa  = softplus(MLP_k(x) + b_k[r]) + KAPPA_MIN + softplus(u[r]) * sum_i v_i
     x is the ComplEx interaction vector Re/Im(h * r * conj(t)).

Training loss (via score_loss_soft_v2, called by train()):
    mean_i c_i * NLL_softmax(pos_i vs negatives)  +  beta_lambda * BetaNLL
plus DS-GAT's regularisation and KL terms added in train().
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .dsgat2 import DSGAT2

W_INIT = -3.0          # softplus(-3) ~ 0.05
KAPPA_MIN = 2.0
KAPPA_MAX = 1000.0
CONF_EPS = 1e-3


class DSGAT3(DSGAT2):
    def __init__(self, node_features, num_relations, embedding_dim, dropout,
                 num_layers=2, num_heads=4, score_function="complex",
                 beta_lambda=0.1, beta_hidden=128):
        assert score_function == "complex", "DSGAT3 uses the ComplEx score"
        super().__init__(node_features, num_relations, embedding_dim, dropout,
                         num_layers=num_layers, num_heads=num_heads,
                         score_function=score_function)
        del self.score_w, self.score_b          # old calibration, never trained
        R = num_relations
        self.beta_lambda = beta_lambda
        # relational evidence: W[query relation, support relation, direction]
        self.support_w = nn.Parameter(torch.full((R, R, 2), W_INIT))
        # confidence decoder
        self.conf_mlp = nn.Sequential(nn.Linear(self.embedding_dim, beta_hidden),
                                      nn.ReLU(), nn.Linear(beta_hidden, 2))
        self.rel_mu_bias = nn.Parameter(torch.zeros(R))
        self.rel_kappa_bias = nn.Parameter(torch.zeros(R))
        self.copy_gate = nn.Parameter(torch.zeros(R))
        self.copy_slope = nn.Parameter(torch.ones(R))
        self.evid_kappa = nn.Parameter(torch.full((R,), -3.0))
        self._sup = None
        self._sup_device = None

    # ── support graph (training facts only) ─────────────────────────────────
    def set_support_graph(self, train_triplets, train_conf, num_entities):
        tr = np.asarray(train_triplets)[:, :3].astype(np.int64)
        conf = (np.ones(len(tr), dtype=np.float32) if train_conf is None
                else np.clip(np.asarray(train_conf, dtype=np.float32), 0.0, 1.0))
        n = int(num_entities)
        key = tr[:, 0] * n + tr[:, 2]                     # pair index (a, b)
        o = np.argsort(key, kind="stable")
        ent = np.concatenate([tr[:, 0], tr[:, 2]])        # entity index
        other = np.concatenate([tr[:, 2], tr[:, 0]])
        rel = np.concatenate([tr[:, 1], tr[:, 1]])
        dirn = np.concatenate([np.zeros(len(tr), np.int64), np.ones(len(tr), np.int64)])
        cf = np.concatenate([conf, conf])
        oe = np.argsort(ent, kind="stable")
        ptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(ptr, ent[oe] + 1, 1)
        self._sup = dict(n=n,
                         pair_key=torch.from_numpy(key[o]), pair_rel=torch.from_numpy(tr[o, 1]),
                         pair_conf=torch.from_numpy(conf[o]),
                         ent_ptr=torch.from_numpy(np.cumsum(ptr)), ent_other=torch.from_numpy(other[oe]),
                         ent_rel=torch.from_numpy(rel[oe]), ent_dir=torch.from_numpy(dirn[oe]),
                         ent_conf=torch.from_numpy(cf[oe]))
        self._sup_device = None

    def _support(self, device):
        if self._sup is None:
            raise RuntimeError("call set_support_graph() before using DSGAT3")
        if self._sup_device != device:
            self._sup = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in self._sup.items()}
            self._sup_device = device
        return self._sup

    @staticmethod
    def _ranges(lo, hi):
        cnt = hi - lo
        rows = torch.repeat_interleave(torch.arange(len(lo), device=lo.device), cnt)
        if rows.numel() == 0:
            return rows, rows
        start = torch.repeat_interleave(lo - torch.cumsum(cnt, 0) + cnt, cnt)
        return rows, start + torch.arange(rows.numel(), device=lo.device)

    def _supports(self, triplets):
        """Supporting facts of a batch: (row, value v_i, confidence c_i)."""
        S = self._support(triplets.device)
        h, r, t = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        sw = F.softplus(self.support_w)
        rows_l, v_l, c_l = [], [], []
        for d, (a, b) in enumerate([(h, t), (t, h)]):
            q = a * S["n"] + b
            lo = torch.searchsorted(S["pair_key"], q, right=False)
            hi = torch.searchsorted(S["pair_key"], q, right=True)
            rows, idx = self._ranges(lo, hi)
            if rows.numel() == 0:
                continue
            r_sup, c = S["pair_rel"][idx], S["pair_conf"][idx]
            if d == 0:                      # the triple itself is not support
                keep = r_sup != r[rows]
                rows, r_sup, c = rows[keep], r_sup[keep], c[keep]
            rows_l.append(rows); c_l.append(c); v_l.append(sw[r[rows], r_sup, d] * c)
        if not rows_l:
            z = torch.zeros(0, device=triplets.device)
            return z.long(), z, z
        return torch.cat(rows_l), torch.cat(v_l), torch.cat(c_l)

    # ── ranking ─────────────────────────────────────────────────────────────
    def evidence(self, triplets):
        rows, v, _ = self._supports(triplets)
        return torch.zeros(len(triplets), device=triplets.device).index_add(0, rows, v)

    def evidence_all_tails(self, h, r):
        """Evidence for (h, r, x) for every entity x (dense, for evaluation)."""
        S = self._support(self.support_w.device)
        lo, hi = int(S["ent_ptr"][h]), int(S["ent_ptr"][h + 1])
        e = torch.zeros(S["n"], device=self.support_w.device)
        if hi > lo:
            r_sup, d = S["ent_rel"][lo:hi], S["ent_dir"][lo:hi]
            w = F.softplus(self.support_w)[r, r_sup, d] * S["ent_conf"][lo:hi]
            w = w * ~((d == 0) & (r_sup == r))
            e = e.index_add(0, S["ent_other"][lo:hi], w)
        return e

    def _raw_score(self, embedding, triplets):
        return self._complex_raw(embedding, triplets) + self.evidence(triplets)

    def _score_for_eval_ids(self, h_emb, r_idx, t_embs, subject, cand):
        base = self._score_for_eval(h_emb, r_idx, t_embs)
        e = self.evidence_all_tails(int(subject), int(r_idx))
        return base + e[cand.to(e.device)]

    # ── confidence ──────────────────────────────────────────────────────────
    def _interaction(self, embedding, triplets):
        h = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        t = embedding[triplets[:, 2]]
        re_h, im_h = torch.chunk(h, 2, dim=-1)
        re_r, im_r = torch.chunk(r, 2, dim=-1)
        re_t, im_t = torch.chunk(t, 2, dim=-1)
        q_re = re_h * re_r - im_h * im_r
        q_im = im_h * re_r + re_h * im_r
        return torch.cat([q_re * re_t + q_im * im_t, q_im * re_t - q_re * im_t], dim=-1)

    def _support_confidence(self, triplets):
        """cbar (attention-weighted support confidence), total support, has-support."""
        n = len(triplets)
        dev = triplets.device
        cbar = torch.zeros(n, device=dev)
        total = torch.zeros(n, device=dev)
        has = torch.zeros(n, dtype=torch.bool, device=dev)
        rows, v, c = self._supports(triplets)
        if rows.numel() == 0:
            return cbar, total, has
        m = torch.zeros(n, device=dev).scatter_reduce(0, rows, v, reduce="amax", include_self=False)
        ex = torch.exp(v - m[rows])
        Z = torch.zeros(n, device=dev).index_add(0, rows, ex)
        cbar = cbar.index_add(0, rows, (ex / Z[rows]) * c)
        total = total.index_add(0, rows, v)
        has[rows] = True
        return cbar, total, has

    def beta_params(self, embedding, triplets):
        x = self._interaction(embedding, triplets).detach()
        out = self.conf_mlp(x)
        rel = triplets[:, 1]
        with torch.no_grad():
            cbar, total, has = self._support_confidence(triplets)
        mu_mlp = torch.sigmoid(out[:, 0] + self.rel_mu_bias[rel])
        lam = torch.sigmoid(self.copy_gate[rel] + self.copy_slope[rel] * torch.log1p(total)) * has.float()
        mu = ((1 - lam) * mu_mlp + lam * cbar).clamp(1e-4, 1 - 1e-4)
        kappa = (F.softplus(out[:, 1] + self.rel_kappa_bias[rel]) + KAPPA_MIN
                 + F.softplus(self.evid_kappa[rel]) * total).clamp(max=KAPPA_MAX)
        return mu * kappa, (1 - mu) * kappa, mu, kappa

    def _calibrated_score(self, embedding, triplets):
        """Predicted confidence (Beta mean), used by calc_confidence_metrics."""
        return self.beta_params(embedding, triplets)[2]

    # ── loss (called by train()) ────────────────────────────────────────────
    def score_loss_soft_v2(self, embedding, triplets, labels, sample_conf, relation_types=None):
        raw_pos, raw_neg, n_pos = self._get_pos_neg(embedding, triplets, labels)
        conf_pos = sample_conf[:n_pos]
        rank = (conf_pos.clamp(1e-6, 1.0) * self._stable_nll(raw_pos, raw_neg)).mean()
        alpha, beta, _, _ = self.beta_params(embedding, triplets[:n_pos])
        target = conf_pos.clamp(CONF_EPS, 1 - CONF_EPS)
        beta_nll = -torch.distributions.Beta(alpha, beta).log_prob(target).mean()
        return rank + self.beta_lambda * beta_nll
