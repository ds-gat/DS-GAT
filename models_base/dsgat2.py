"""DSGAT2 — DS-GAT with Option 2 stream separation and eval-only cache.

Option 2: separate edge projections per stream:
  - Stream A: edge_project(efeats)                     — relation only, no confidence
  - Stream B: edge_project_b(cat([efeats, eff_w]))     — relation + Bayesian confidence

Performance fix: stream cache (alpha, lam_w) computed only at eval time,
avoiding double edge_softmax and GPU->CPU transfer during training.
"""

import torch
from torch import nn
import torch.nn.functional as F
import dgl
from torch.nn import init


class DSGAT2(nn.Module):
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
            EGATConv28(node_dim, node_dim, node_dim, node_dim,
                       num_heads, num_relations)
            for _ in range(num_layers)
        ])

        self._conf_dist_cache: list = []
        self._stream_cache:    list = []

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
        self._stream_cache    = []

        for i, conv in enumerate(self.convs):
            x, efeats, w_mean, w_var = conv(g, x, efeats, edge_type, edge_weight)
            x      = x.mean(dim=1)
            efeats = efeats.mean(dim=1)

            self._conf_dist_cache.append((w_mean, w_var))

            # Stream cache populated by conv only at eval time (see EGATConv28.forward)
            layer_cache = {}
            if hasattr(conv, '_cached_lam_w'):
                layer_cache['lam_w'] = conv._cached_lam_w
            if hasattr(conv, '_cached_alpha'):
                layer_cache['alpha'] = conv._cached_alpha
            self._stream_cache.append(layer_cache)

            g.ndata['h'] = x
            g.edata['f'] = efeats

            if i != len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_ratio, training=self.training)
                g.ndata['h'] = x

        return x

    def confidence_kl_loss(self, edge_weight: torch.Tensor,
                           sigma: float = 0.3) -> torch.Tensor:
        if not self._conf_dist_cache:
            raise RuntimeError(
                "confidence_kl_loss() must be called after forward().")

        gt     = edge_weight.float().view(-1, 1).clamp(1e-6, 1.0)
        sigma2 = sigma ** 2
        kl_terms = []

        for w_mean, w_var in self._conf_dist_cache:
            kl = (0.5 * torch.log(torch.tensor(sigma2, device=gt.device) / (w_var + 1e-8))
                  + (w_var + (w_mean - gt) ** 2) / (2.0 * sigma2)
                  - 0.5)
            kl_terms.append(kl.mean())

        return torch.stack(kl_terms).mean()

    # ── Scoring ───────────────────────────────────────────────────────────────
    def _distmult_direct(self, h, r, t):
        return torch.sum(h * r * t, dim=-1)

    def _complex_direct(self, h, r, t):
        re_h, im_h = torch.chunk(h, 2, dim=-1)
        re_r, im_r = torch.chunk(r, 2, dim=-1)
        re_t, im_t = torch.chunk(t, 2, dim=-1)
        return torch.sum(
            re_h * re_t * re_r
            + im_h * im_t * re_r
            + re_h * im_t * im_r
            - im_h * re_t * im_r, dim=-1)

    def _rotate_direct(self, h, r, t):
        pi    = 3.14159265358979323846
        re_h, im_h = torch.chunk(h, 2, dim=-1)
        re_t, im_t = torch.chunk(t, 2, dim=-1)
        phase = r / (self.embedding_range.item() / pi)
        re_r  = torch.cos(phase)
        im_r  = torch.sin(phase)
        re_sc = re_h * re_r - im_h * im_r - re_t
        im_sc = re_h * im_r + im_h * re_r - im_t
        score = torch.stack([re_sc, im_sc], dim=0).norm(dim=0)
        return self.rotate_margin.item() - score.sum(dim=-1)

    def _distmult_raw(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        return self._distmult_direct(s, r, o)

    def _complex_raw(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        return self._complex_direct(s, r, o)

    def _rotate_raw(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        o = embedding[triplets[:, 2]]
        r = self.relation_embedding[triplets[:, 1]]
        return self._rotate_direct(s, r, o)

    def _raw_score(self, embedding, triplets):
        if self.score_function == "complex":
            return self._complex_raw(embedding, triplets)
        elif self.score_function == "rotate":
            return self._rotate_raw(embedding, triplets)
        return self._distmult_raw(embedding, triplets)

    def _calibrated_score(self, embedding, triplets):
        return torch.sigmoid(self.score_w * self._raw_score(embedding, triplets)
                             + self.score_b)

    def distmult(self, embedding, triplets):
        return self._distmult_raw(embedding, triplets)

    def complex_score(self, embedding, triplets):
        return self._complex_raw(embedding, triplets)

    def _score_for_eval(self, h_emb: torch.Tensor, r_idx: torch.Tensor,
                        t_embs: torch.Tensor) -> torch.Tensor:
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

    def score_loss_soft(self, embedding, triplets, labels, sample_conf):
        raw_pos, raw_neg, n_pos = self._get_pos_neg(embedding, triplets, labels)
        conf_pos    = sample_conf[:n_pos].clamp(1e-6, 1.0)
        nll_per_pos = self._stable_nll(raw_pos, raw_neg)
        return (conf_pos * nll_per_pos).mean()

    def score_loss(self, embedding, triplets, labels):
        raw_pos, raw_neg, _ = self._get_pos_neg(embedding, triplets, labels)
        return self._stable_nll(raw_pos, raw_neg).mean()

    def score_loss_soft_v2(self, embedding, triplets, labels, sample_conf,
                           relation_types=None):
        raw_pos, raw_neg, n_pos = self._get_pos_neg(embedding, triplets, labels)
        conf_pos    = sample_conf[:n_pos].clamp(1e-6, 1.0)
        nll_per_pos = self._stable_nll(raw_pos, raw_neg)
        return (conf_pos * nll_per_pos).mean()

    def reg_loss(self, embedding):
        return (torch.mean(embedding.pow(2))
                + torch.mean(self.relation_embedding.pow(2)))


# ─────────────────────────────────────────────────────────────────────────────
# EGATConv28 — Option 2: separate projections for Stream A and Stream B

class EGATConv28(nn.Module):

    def __init__(self, in_node_feats, in_edge_feats, out_node_feats, out_edge_feats,
                 num_heads, num_relations, default_edge_weight=1.0, **kwargs):
        super().__init__()
        self._num_heads      = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.num_relations   = num_relations
        self.default_edge_weight = default_edge_weight

        # Option 2: separate projections
        # Stream A: relation embedding only — topologically pure
        self.edge_project   = nn.Linear(in_edge_feats,     in_edge_feats)
        # Stream B: relation embedding + Bayesian confidence weight
        self.edge_project_b = nn.Linear(in_edge_feats + 1, in_edge_feats)

        self.fc_nodes = nn.Linear(in_node_feats, out_node_feats * num_heads, bias=True)
        self.fc_edges = nn.Linear(in_edge_feats + 2 * in_node_feats,
                                  out_edge_feats * num_heads, bias=False)
        self.fc_attn  = nn.Linear(out_edge_feats, 1, bias=False)
        self.W_B      = nn.Linear(in_node_feats, out_node_feats, bias=True)

        self.lambda_raw = nn.Parameter(torch.full((num_relations,), -1.0))
        self.mean_scale = nn.Parameter(torch.full((num_relations,), 0.54))
        self.var_scale  = nn.Parameter(torch.ones(num_relations))

        self.correction_mlp = nn.Sequential(
            nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1), nn.Tanh())
        self.edge_weight_var_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid())

        self._reset_parameters()

    def _reset_parameters(self):
        gain = init.calculate_gain('relu')
        for m in [self.fc_nodes, self.fc_edges, self.fc_attn,
                  self.edge_project, self.edge_project_b, self.W_B]:
            init.xavier_normal_(m.weight, gain=gain)
        for m in [self.edge_project, self.edge_project_b, self.W_B]:
            init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.correction_mlp[0].weight)
        nn.init.zeros_(self.correction_mlp[0].bias)
        with torch.no_grad():
            self.correction_mlp[2].weight.fill_(0.0)
            self.correction_mlp[2].bias.fill_(0.0)
        nn.init.xavier_uniform_(self.edge_weight_var_mlp[0].weight)
        nn.init.zeros_(self.edge_weight_var_mlp[0].bias)
        nn.init.xavier_uniform_(self.edge_weight_var_mlp[2].weight)
        nn.init.zeros_(self.edge_weight_var_mlp[2].bias)
        with torch.no_grad():
            self.edge_weight_var_mlp[4].weight.fill_(-0.5)
            self.edge_weight_var_mlp[4].bias.fill_(0.5)

    def _compute_eff_w(self, ew, edge_type):
        delta       = self.correction_mlp(ew)
        weight_mean = torch.clamp(ew + delta, 0.0, 1.0)
        weight_mean = weight_mean * F.softplus(
            self.mean_scale[edge_type]).view(-1, 1)
        weight_var  = (self.edge_weight_var_mlp(ew)
                       * F.softplus(self.var_scale[edge_type]).view(-1, 1)
                       + 1e-6)
        if self.training:
            eff_w = torch.clamp(
                weight_mean + torch.randn_like(weight_mean) * weight_var.sqrt(),
                1e-3, 1.0)
        else:
            eff_w = torch.clamp(weight_mean / (1.0 + weight_var), 1e-3, 1.0)
        return eff_w, weight_mean, weight_var

    def edge_attention(self, edges):
        h_src = edges.src['h'].mean(dim=1)
        h_dst = edges.dst['h'].mean(dim=1)
        # Stream A uses ef (relation only) for attention — topologically pure
        stack = torch.cat([h_src, edges.data['f'], h_dst], dim=-1)
        f_out = F.leaky_relu(self.fc_edges(stack)).view(
            -1, self._num_heads, self._out_edge_feats)
        attn_logits = self.fc_attn(f_out)
        return {'a': attn_logits, 'f': f_out}

    def message_func(self, edges):
        return {
            'h':     edges.src['h'],
            'a':     edges.data['a'],
            'm_b':   edges.src['h_raw'] * edges.data['f_b'],
            'lam_w': edges.data['lam_w'],
        }

    def reduce_func(self, nodes):
        alpha  = F.softmax(nodes.mailbox['a'], dim=1)
        h_topo = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        lam_w  = nodes.mailbox['lam_w']
        agg    = ((lam_w * nodes.mailbox['m_b']).sum(dim=1)
                / (lam_w.sum(dim=1) + 1e-8))
        h_out  = h_topo + self.W_B(agg).unsqueeze(1).expand(
            -1, self._num_heads, -1)
        return {'h': h_out}

    def forward(self, graph, nfeats, efeats, edge_type, edge_weight=None):
        """Returns (node_feats, edge_feats, weight_mean, weight_var)."""
        with graph.local_scope():
            ew = (edge_weight.float().view(-1, 1) if edge_weight is not None
                  else torch.full((graph.num_edges(), 1),
                                  self.default_edge_weight,
                                  device=nfeats.device))

            eff_w, weight_mean, weight_var = self._compute_eff_w(ew, edge_type)

            lam_w = F.softplus(
                self.lambda_raw[edge_type]).view(-1, 1) * eff_w

            # Option 2: separate projections per stream
            ef   = self.edge_project(efeats)                                # Stream A: relation only
            ef_b = self.edge_project_b(torch.cat([efeats, eff_w], dim=1))  # Stream B: relation + confidence

            graph.ndata.update({'h': nfeats, 'h_raw': nfeats})
            graph.edata.update({'f': ef, 'f_b': ef_b, 'lam_w': lam_w})

            graph.ndata['h'] = self.fc_nodes(nfeats).view(
                -1, self._num_heads, self._out_node_feats)

            graph.apply_edges(self.edge_attention)
            graph.update_all(self.message_func, self.reduce_func)

            # Stream cache: only at eval time to avoid GPU->CPU sync during training
            if not self.training:
                attn_soft = dgl.ops.edge_softmax(graph, graph.edata['a'])
                self._cached_alpha = attn_soft.mean(dim=[1, 2]).detach().cpu()  # [n_edges]
                self._cached_lam_w = lam_w.squeeze(-1).detach().cpu()          # [n_edges]

            return graph.ndata.pop('h'), graph.edata.pop('f'), weight_mean, weight_var