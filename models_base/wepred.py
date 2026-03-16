"""WePred: Edge Weight-Guided Contrastive Learning for Bipartite Link Prediction.

Adapted for knowledge graph link prediction, combining:
  - Weight-guided edge attention convolution (WePredConv)
  - Dual-level contrastive learning: edge-level InfoNCE + node-level SimCSE-style
  - DistMult decoder for KG-compatible scoring

Reference:
    Chen et al., "WePred: Edge Weight-Guided Contrastive Learning for
    Bipartite Link Prediction", Electronics 14(1), 2025.
    https://www.mdpi.com/2079-9292/14/1/20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter


class WePredConv(nn.Module):
    """Weight-guided edge attention convolution layer.

    For each edge (i→j) with weight w_ij and relation embedding e_ij the
    attention score is computed as:

        alpha_raw = LeakyReLU( (q_i + k_j + proj(e_ij)) / sqrt(C)
                               + gate * w_ij )
        alpha_ij  = softmax_{j in N(i)}( alpha_raw )
        h_i'      = sum_j  alpha_ij  *  v_j

    where q, k, v are per-head query / key / value projections of the node
    features, gate is a learnable per-head scalar that controls how strongly
    edge weights steer the attention, and e_ij is the (projected) relation
    embedding for edge (i, j).
    """

    def __init__(self, in_channels, out_channels, heads=4, dropout=0.0,
                 edge_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim

        # Query / key / value projections
        self.lin_q = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_k = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_v = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Learnable per-head scalar gate for the edge-weight term
        self.weight_gate = nn.Parameter(torch.ones(heads))

        # Optional projection of relation (edge) embeddings
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels,
                                      bias=False)
        else:
            self.lin_edge = None

        self.bias = nn.Parameter(torch.zeros(heads * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.ones_(self.weight_gate)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        """
        Parameters
        ----------
        x           : [N, in_channels]
        edge_index  : [2, E]  (src, dst)
        edge_weight : [E]     edge confidence / weight  (optional)
        edge_attr   : [E, edge_dim]  relation embeddings  (optional)

        Returns
        -------
        out : [N, heads * out_channels]
        """
        N = x.size(0)
        H, C = self.heads, self.out_channels
        src_idx, dst_idx = edge_index[0], edge_index[1]

        # Project node features
        q = self.lin_q(x).view(N, H, C)   # [N, H, C]
        k = self.lin_k(x).view(N, H, C)   # [N, H, C]
        v = self.lin_v(x).view(N, H, C)   # [N, H, C]

        # Gather per-edge query (destination) and key (source) vectors
        q_i = q[dst_idx]   # [E, H, C]
        k_j = k[src_idx]   # [E, H, C]
        v_j = v[src_idx]   # [E, H, C]

        # Weight-guided attention logits
        if edge_attr is not None and self.lin_edge is not None:
            e_ij = self.lin_edge(edge_attr).view(-1, H, C)   # [E, H, C]
            alpha = (q_i + k_j + e_ij).sum(dim=-1) / (C ** 0.5)  # [E, H]
        else:
            alpha = (q_i + k_j).sum(dim=-1) / (C ** 0.5)          # [E, H]

        if edge_weight is not None:
            # weight_gate: [H] → broadcast over edges
            alpha = alpha + self.weight_gate.unsqueeze(0) * edge_weight.unsqueeze(-1)

        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Normalise: softmax over all incoming edges of each destination node
        alpha = pyg_softmax(alpha, dst_idx, num_nodes=N)   # [E, H]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted aggregation → destination nodes
        msg = v_j * alpha.unsqueeze(-1)                              # [E, H, C]
        out = scatter(msg, dst_idx, dim=0, dim_size=N, reduce='sum') # [N, H, C]
        out = out.view(N, H * C)
        out = out + self.bias

        return out


class WePred(nn.Module):
    """WePred: Edge Weight-Guided Contrastive Learning for Bipartite Link
    Prediction — adapted to knowledge-graph link prediction.

    Architecture
    ------------
    - Pre-trained node embeddings (fine-tuned), projected to embedding_dim
    - num_layers × WePredConv with ELU + Dropout between hidden layers
    - DistMult decoder: f(s,r,o) = e_s ⊙ r ⊙ e_o
    - Score loss: BCE (classification) + contrastive_weight × (edge CL + node CL)

    The edge-level contrastive loss uses the negative samples generated by the
    training loop as contrastive negatives, bringing positive-pair embeddings
    together and pushing negative-pair embeddings apart (InfoNCE-style).

    The node-level contrastive loss applies SimCSE-style augmentation: two
    independently-noised views of the same node form a positive pair; all other
    nodes in the batch serve as negatives.

    Parameters
    ----------
    num_entities        : total number of entities
    num_relations       : number of relation types (including inverses)
    dropout             : dropout rate for intermediate layers
    node_features       : pre-trained feature matrix [num_entities, feat_dim]
    embedding_dim       : latent dimension (must be divisible by heads)
    num_layers          : number of WePredConv message-passing layers
    heads               : number of attention heads
    contrastive_temp    : softmax temperature τ for contrastive losses
    contrastive_weight  : λ multiplying the contrastive term in total loss

    Usage in models.py
    ------------------
    Set --weighted flag so that models.py passes edge_weight to forward().
    """

    def __init__(self, num_entities, num_relations, dropout, node_features,
                 embedding_dim=100, num_layers=2, heads=4,
                 contrastive_temp=0.1, contrastive_weight=0.1):
        super().__init__()

        assert embedding_dim % heads == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by heads ({heads})"

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_ratio = dropout
        self.contrastive_temp = contrastive_temp
        self.contrastive_weight = contrastive_weight

        # Pre-trained node embeddings (fine-tuned)
        self.entity_embedding = nn.Embedding.from_pretrained(
            node_features, freeze=False
        )
        self.project = nn.Linear(node_features.shape[1], embedding_dim)

        # Relation embeddings: used for DistMult scoring and as edge features
        self.relation_embedding = nn.Parameter(
            torch.Tensor(num_relations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.relation_embedding)

        # WePred attention layers
        self.convs = nn.ModuleList([
            WePredConv(
                in_channels=embedding_dim,
                out_channels=embedding_dim // heads,
                heads=heads,
                dropout=dropout,
                edge_dim=embedding_dim,
            )
            for _ in range(num_layers)
        ])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, entity, edge_index, edge_type, edge_weight=None,
                edge_norm=None):
        """
        Parameters
        ----------
        entity       : [N]     entity indices in the current subgraph
        edge_index   : [2, E]
        edge_type    : [E]     relation type per edge
        edge_weight  : [E]     confidence / weight per edge  (recommended)
        edge_norm    : [E]     unused (kept for interface compatibility)

        Returns
        -------
        x : [N, embedding_dim]  updated node embeddings
        """
        x = self.entity_embedding(entity)
        x = self.project(x)

        # Relation embeddings as per-edge features
        edge_attr = self.relation_embedding[edge_type]   # [E, embedding_dim]

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
            if i != len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        return x

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        return torch.sum(s * r * o, dim=1)

    def score_loss(self, embedding, triplets, target):
        """Combined BCE + dual-level contrastive loss.

        During eval mode only the classification loss is returned (consistent
        with how the training loop calls this function during validation).
        """
        score = self.distmult(embedding, triplets)
        cls_loss = F.binary_cross_entropy_with_logits(score, target)

        if not self.training:
            return cls_loss

        pos_mask = target > 0.5
        neg_mask = ~pos_mask

        if pos_mask.sum() > 1 and neg_mask.sum() > 1:
            edge_cl = self._edge_contrastive_loss(
                embedding, triplets, pos_mask, neg_mask
            )
            node_cl = self._node_contrastive_loss(embedding)
            return cls_loss + self.contrastive_weight * (edge_cl + node_cl)

        return cls_loss

    def reg_loss(self, embedding):
        return (torch.mean(embedding.pow(2)) +
                torch.mean(self.relation_embedding.pow(2)))

    # ------------------------------------------------------------------
    # Contrastive helpers
    # ------------------------------------------------------------------

    def _edge_contrastive_loss(self, embedding, triplets, pos_mask, neg_mask):
        """Edge-level contrastive loss (local interaction patterns).

        Positive pairs  : source / destination of *real* edges (labels == 1).
        Negative pairs  : source / destination of *corrupted* edges from the
                          negative sampling already done by the training loop.

        Uses a binary InfoNCE formulation: for each positive pair, the loss
        maximises the cosine similarity relative to the paired negative.
        """
        pos_s = embedding[triplets[pos_mask, 0]]   # [P, d]
        pos_o = embedding[triplets[pos_mask, 2]]   # [P, d]

        n_neg = min(neg_mask.sum().item(), pos_mask.sum().item())
        neg_idx = torch.where(neg_mask)[0][:n_neg]
        neg_s = embedding[triplets[neg_idx, 0]]    # [Q, d]
        neg_o = embedding[triplets[neg_idx, 2]]    # [Q, d]

        pos_sim = (F.cosine_similarity(pos_s, pos_o, dim=-1)
                   / self.contrastive_temp)          # [P]
        neg_sim = (F.cosine_similarity(neg_s, neg_o, dim=-1)
                   / self.contrastive_temp)          # [Q]

        n = min(pos_sim.size(0), neg_sim.size(0))
        # logits[:, 0] = positive score, logits[:, 1] = negative score
        logits = torch.stack([pos_sim[:n], neg_sim[:n]], dim=1)   # [n, 2]
        labels = torch.zeros(n, dtype=torch.long, device=embedding.device)

        return F.cross_entropy(logits, labels)

    def _node_contrastive_loss(self, embedding, num_samples=256):
        """Node-level contrastive loss (global structural patterns).

        Implements a SimCSE-style loss: two independently noise-perturbed views
        of the same node embedding are treated as a positive pair; all other
        nodes in the sampled batch serve as negatives.  This encourages the
        model to produce representations that are robust to minor perturbations
        and well-separated across nodes.
        """
        N = embedding.size(0)
        if N > num_samples:
            idx = torch.randperm(N, device=embedding.device)[:num_samples]
            emb = embedding[idx]
        else:
            emb = embedding

        n = emb.size(0)
        noise_scale = 0.01 * emb.detach().std().clamp(min=1e-6)

        z1 = F.normalize(emb + torch.randn_like(emb) * noise_scale, p=2, dim=-1)
        z2 = F.normalize(emb + torch.randn_like(emb) * noise_scale, p=2, dim=-1)

        # [n, n] cosine similarity matrix (scaled by temperature)
        sim = torch.mm(z1, z2.t()) / self.contrastive_temp

        # Positive pairs are on the diagonal (same node, two views)
        labels = torch.arange(n, device=embedding.device)
        return F.cross_entropy(sim, labels)
