"""Ablations of the final model DSGAT3 (same evidence, decoder, loss and
separate evidence learning rate; only the removed part differs).

  DSGAT3A1: neighborhood stream only. GAT layers are EGATConv_A1 (no
            confidence stream, no Bayesian weights, no KL term).
  DSGAT3A2: confidence stream without Bayesian learning. GAT layers are
            EGATConv_A2 (deterministic edge weights, no posterior, no KL).
  DSGAT3NE: no relational evidence. The ranking score is ComplEx only and
            the confidence decoder sees no supports (lambda = 0, no
            evidence term in kappa). Encoder unchanged.
"""

import torch
import torch.nn.functional as F
from torch import nn
import dgl

from .dsgat3 import DSGAT3
from .dsgat_a1 import EGATConv_A1
from .dsgat_a2 import EGATConv_A2


class _DSGAT3NoBayes(DSGAT3):
    """Shared forward for the ablations whose GAT layers have no Bayesian stream."""
    conv_class = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        node_dim = self.project.out_features
        self.convs = nn.ModuleList([
            self.conv_class(node_dim, node_dim, node_dim, node_dim,
                            self.num_heads, self.num_relations)
            for _ in range(self.num_layers)])

    def forward(self, entity, edge_index, edge_type=None, edge_weight=None):
        x = self.project(self.entity_embedding(entity))
        efeats = self.relation_embedding[edge_type]
        src, dst = edge_index
        g = dgl.graph((src, dst), num_nodes=x.size(0))
        g.ndata['h'] = x
        g.edata['f'] = efeats
        self._conf_dist_cache = []
        self._stream_cache = []
        for i, conv in enumerate(self.convs):
            out = conv(g, x, efeats, edge_type, edge_weight)
            x, efeats = out[0].mean(dim=1), out[1].mean(dim=1)
            g.ndata['h'] = x
            g.edata['f'] = efeats
            if i != len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_ratio, training=self.training)
                g.ndata['h'] = x
        return x

    def confidence_kl_loss(self, edge_weight, sigma=0.3):
        """No Bayesian stream: no KL term."""
        return torch.zeros((), device=edge_weight.device)


class DSGAT3A1(_DSGAT3NoBayes):
    conv_class = EGATConv_A1


class DSGAT3A2(_DSGAT3NoBayes):
    conv_class = EGATConv_A2


class DSGAT3NE(DSGAT3):
    def evidence(self, triplets):
        return torch.zeros(len(triplets), device=triplets.device)

    def evidence_all_tails(self, h, r):
        return torch.zeros(self._support(self.support_w.device)["n"],
                           device=self.support_w.device)

    def _support_confidence(self, triplets):
        n, dev = len(triplets), triplets.device
        return (torch.zeros(n, device=dev), torch.zeros(n, device=dev),
                torch.zeros(n, dtype=torch.bool, device=dev))
