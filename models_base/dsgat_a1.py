"""DSGATA1 — Ablation: No Stream B, no Bayesian (attention only).

Identical to DSGAT5 except EGATConv28 is replaced with EGATConv_A1
which removes Stream B and Bayesian entirely:
  - No W_B projection
  - No lambda_raw per-relation weights
  - No lam_w weighted aggregation
  - No correction_mlp, variance_mlp, mean_scale, var_scale
  - No KL loss (returns zero)
  - Only Stream A (softmax attention) aggregates neighbor features
  - edge_project(efeats) — relation only, no confidence

Rationale: Stream B is the only channel through which eff_w enters the
graph. Removing Stream B without removing Bayesian would compute KL loss
on parameters that have no path to the node embeddings — meaningless.
A1 is therefore a pure attention baseline with no confidence signal.

Purpose: validates contribution of dual-stream + Bayesian design.
Compare DS-GAT >> A4 >> A1 for progressive ablation story:
  DS-GAT: full model (Stream A + Stream B + Bayesian)
  A4:     Stream A + Stream B deterministic (no Bayesian)
  A1:     Stream A only (no Stream B, no Bayesian)
"""

import torch
from torch import nn
import torch.nn.functional as F
import dgl
from torch.nn import init


class DSGATA1(nn.Module):
    def __init__(self, node_features, num_relations, embedding_dim, dropout,
                 num_layers=2, num_heads=4, score_function="dismult"):
        super().__init__()

        if score_function == "complex":
            embedding_dim = embedding_dim * 2

        self.embedding_dim  = embedding_dim
        self.num_layers     = num_layers
        self.num_heads      = num_heads
        self.dropout_ratio  = dropout
        self.score_function = score_function
        self.num_relations  = num_relations

        entity_proj_dim = embedding_dim * 2 if score_function == "rotate" else embedding_dim

        self.entity_embedding   = nn.Embedding.from_pretrained(node_features, freeze=False)
        self.project            = nn.Linear(node_features.shape[1], entity_proj_dim)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.relation_embedding)

        self.score_w = nn.Parameter(torch.tensor(1.0))
        self.score_b = nn.Parameter(torch.tensor(0.0))

        if score_function == "rotate":
            self.epsilon = 2.0
            self.rotate_margin = nn.Parameter(
                torch.Tensor([9.0]), requires_grad=False)
            self.embedding_range = nn.Parameter(
                torch.Tensor([(9.0 + self.epsilon) / embedding_dim]),
                requires_grad=False)
            nn.init.uniform_(self.relation_embedding,
                             -self.embedding_range.item(),
                              self.embedding_range.item())

        self.log_tau = nn.Parameter(torch.tensor(0.0))

        node_dim = entity_proj_dim
        self.convs = nn.ModuleList([
            EGATConv_A1(node_dim, node_dim, node_dim, node_dim,
                        num_heads, num_relations)
            for _ in range(num_layers)
        ])

        # Empty — no Bayesian stream, kept for interface compatibility
        self._conf_dist_cache: list = []

    def forward(self, entity, edge_index, edge_type=None, edge_weight=None):
        x      = self.project(self.entity_embedding(entity))
        efeats = self.relation_embedding[edge_type]

        if self.score_function == "rotate" and efeats.shape[1] != x.shape[1]:
            pad    = torch.zeros(efeats.shape[0], x.shape[1] - efeats.shape[1],
                                 device=efeats.device)
            efeats = torch.cat([efeats, pad], dim=1)

        src, dst = edge_index
        g = dgl.graph((src, dst), num_nodes=x.size(0))
        g.ndata['h'] = x
        g.edata['f'] = efeats

        self._conf_dist_cache = []

        for i, conv in enumerate(self.convs):
            x, efeats = conv(g, x, efeats, edge_type, edge_weight)
            x      = x.mean(dim=1)
            efeats = efeats.mean(dim=1)

            g.ndata['h'] = x
            g.edata['f'] = efeats

            if i != len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_ratio, training=self.training)
                g.ndata['h'] = x

        return x

    def confidence_kl_loss(self, edge_weight: torch.Tensor,
                           sigma: float = 0.3) -> torch.Tensor:
        """No-op — no Bayesian stream in A1."""
        return torch.tensor(0.0, requires_grad=True, device=edge_weight.device)

    # ── Scoring (identical to DSGAT5) ─────────────────────────────────────────
    def _distmult_direct(self, h, r, t):
        return torch.sum(h * r * t, dim=-1)

    def _complex_direct(self, h, r, t):
        re_h, im_h = torch.chunk(h, 2, dim=-1)
        re_r, im_r = torch.chunk(r, 2, dim=-1)
        re_t, im_t = torch.chunk(t, 2, dim=-1)
        return torch.sum(
            re_h * re_t * re_r + im_h * im_t * re_r
            + re_h * im_t * im_r - im_h * re_t * im_r, dim=-1)

    def _rotate_direct(self, h, r, t):
        pi = 3.14159265358979323846
        re_h, im_h = torch.chunk(h, 2, dim=-1)
        re_t, im_t = torch.chunk(t, 2, dim=-1)
        phase = r / (self.embedding_range.item() / pi)
        re_r  = torch.cos(phase)
        im_r  = torch.sin(phase)
        re_sc = re_h * re_r - im_h * im_r - re_t
        im_sc = re_h * im_r + im_h * re_r - im_t
        score = torch.stack([re_sc, im_sc], dim=0).norm(dim=0)
        return self.rotate_margin.item() - score.sum(dim=-1)

    def _raw_score(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        if self.score_function == "complex":
            return self._complex_direct(s, r, o)
        elif self.score_function == "rotate":
            return self._rotate_direct(s, r, o)
        return self._distmult_direct(s, r, o)

    def _calibrated_score(self, embedding, triplets):
        return torch.sigmoid(self.score_w * self._raw_score(embedding, triplets)
                             + self.score_b)

    def distmult(self, embedding, triplets):
        return self._raw_score(embedding, triplets)

    def _score_for_eval(self, h_emb, r_idx, t_embs):
        n   = t_embs.shape[0]
        r   = self.relation_embedding[r_idx]
        h   = h_emb.unsqueeze(0).expand(n, -1)
        r_b = r.unsqueeze(0).expand(n, -1)
        if self.score_function == "complex":
            return self._complex_direct(h, r_b, t_embs)
        elif self.score_function == "rotate":
            return self._rotate_direct(h, r_b, t_embs)
        return self._distmult_direct(h, r_b, t_embs)

    def _get_pos_neg(self, embedding, triplets, labels):
        raw      = self._raw_score(embedding, triplets)
        n_pos    = int((labels > 0.5).sum().item())
        n_neg    = len(labels) - n_pos
        neg_rate = max(n_neg // max(n_pos, 1), 1)
        raw_pos  = raw[:n_pos]
        raw_neg  = raw[n_pos:n_pos + neg_rate * n_pos].view(neg_rate, n_pos).t()
        return raw_pos, raw_neg, n_pos

    def _stable_nll(self, raw_pos, raw_neg):
        all_scores  = torch.cat([raw_pos.unsqueeze(1), raw_neg], dim=1)
        logsumexp   = torch.logsumexp(all_scores, dim=1)
        return logsumexp - raw_pos

    def score_loss_soft_v2(self, embedding, triplets, labels, sample_conf,
                           relation_types=None):
        raw_pos, raw_neg, n_pos = self._get_pos_neg(embedding, triplets, labels)
        conf_pos    = sample_conf[:n_pos].clamp(1e-6, 1.0)
        nll_per_pos = self._stable_nll(raw_pos, raw_neg)
        return (conf_pos * nll_per_pos).mean()

    def score_loss(self, embedding, triplets, labels):
        raw_pos, raw_neg, _ = self._get_pos_neg(embedding, triplets, labels)
        return self._stable_nll(raw_pos, raw_neg).mean()

    def reg_loss(self, embedding):
        return (torch.mean(embedding.pow(2))
                + torch.mean(self.relation_embedding.pow(2)))


# ─────────────────────────────────────────────────────────────────────────────
# EGATConv_A1 — Stream A only, no Stream B, no Bayesian

class EGATConv_A1(nn.Module):
    """Pure softmax attention conv — no confidence signal whatsoever.

    Removed vs DSGAT5:
      - W_B, lambda_raw (Stream B)
      - correction_mlp, variance_mlp, mean_scale, var_scale (Bayesian)
      - eff_w never computed or used

    Kept:
      - Stream A: softmax attention aggregation (unchanged)
      - edge_project(efeats): relation embedding only
    """

    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats,
                 num_heads, num_relations, default_edge_weight=1.0, **kwargs):
        super().__init__()
        self._num_heads      = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.num_relations   = num_relations
        self.default_edge_weight = default_edge_weight

        self.edge_project = nn.Linear(in_edge_feats, in_edge_feats)
        self.fc_nodes     = nn.Linear(in_node_feats, out_node_feats * num_heads, bias=True)
        self.fc_edges     = nn.Linear(in_edge_feats + 2 * in_node_feats,
                                      out_edge_feats * num_heads, bias=False)
        self.fc_attn      = nn.Linear(out_edge_feats, 1, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        gain = init.calculate_gain('relu')
        for m in [self.fc_nodes, self.fc_edges, self.fc_attn, self.edge_project]:
            init.xavier_normal_(m.weight, gain=gain)
        init.zeros_(self.edge_project.bias)

    def edge_attention(self, edges):
        h_src = edges.src['h'].mean(dim=1)
        h_dst = edges.dst['h'].mean(dim=1)
        stack = torch.cat([h_src, edges.data['f'], h_dst], dim=-1)
        f_out = F.leaky_relu(self.fc_edges(stack)).view(
            -1, self._num_heads, self._out_edge_feats)
        return {'a': self.fc_attn(f_out), 'f': f_out}

    def message_func(self, edges):
        return {'h': edges.src['h'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['a'], dim=1)
        return {'h': torch.sum(alpha * nodes.mailbox['h'], dim=1)}

    def forward(self, graph, nfeats, efeats, edge_type, edge_weight=None):
        """Returns (node_feats, edge_feats) — no Bayesian outputs."""
        with graph.local_scope():
            ef = self.edge_project(efeats)

            graph.ndata['h'] = self.fc_nodes(nfeats).view(
                -1, self._num_heads, self._out_node_feats)
            graph.edata['f'] = ef

            graph.apply_edges(self.edge_attention)
            graph.update_all(self.message_func, self.reduce_func)

            return graph.ndata.pop('h'), graph.edata.pop('f')