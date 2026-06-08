import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GATv2Conv
from utils import uniform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATv2Rel(nn.Module):

    def __init__(self,
                 num_entities,
                 num_relations,
                 dropout,
                 node_features,
                 embedding_dim=300,
                 num_layers=2,
                 heads=4,
                 weights=False,
                 score_function="dismult"):

        super().__init__()

        if score_function == "complex":
            embedding_dim = embedding_dim * 2
        self.score_function = score_function
        self.embedding_dim = embedding_dim
        self.dropout_ratio = dropout
        self.num_layers = num_layers
        self.heads = heads
        self.weights = weights

        # Node embeddings
        self.entity_embedding = nn.Embedding.from_pretrained(
            node_features,   # shape: [num_entities, embedding_dim]
            freeze=False     # IMPORTANT: allow training
        )

        # Relation embeddings (used BOTH for attention and DistMult)
        self.relation_embedding = nn.Parameter(
            torch.Tensor(num_relations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.relation_embedding)

        # Project node features
        self.project = nn.Linear(
            node_features.shape[1],
            embedding_dim
        )

        # GATv2 layers with edge features
        self.convs = nn.ModuleList()

        for _ in range(num_layers):

            self.convs.append(
                GATv2Conv(
                    in_channels=embedding_dim,
                    out_channels=embedding_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=embedding_dim   
                )
            )

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------
    def forward(self, entity, edge_index, edge_type, edge_norm=None,  edge_weight=None):

        x = self.entity_embedding(entity)
        x = self.project(x)  # projects pretrained embeddings to latent space

        # Convert edge_type → edge_attr using relation embeddings
        rel_emb = self.relation_embedding[edge_type]

        if edge_weight is not None and self.weights:
            rel_emb = rel_emb * edge_weight.unsqueeze(1)
        edge_attr = rel_emb

        for i, conv in enumerate(self.convs):

            x = conv(x, edge_index, edge_attr)

            if i != len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(
                    x,
                    p=self.dropout_ratio,
                    training=self.training
                )

        return x

    # ------------------------------------------------
    # Scoring
    # ------------------------------------------------
    def _distmult_direct(self, h, r, t):
        return torch.sum(h * r * t, dim=-1)

    def _complex_direct(self, h, r, t):
        re_h, im_h = torch.chunk(h, 2, dim=-1)
        re_r, im_r = torch.chunk(r, 2, dim=-1)
        re_t, im_t = torch.chunk(t, 2, dim=-1)
        return torch.sum(
            re_h * re_t * re_r + im_h * im_t * re_r
            + re_h * im_t * im_r - im_h * re_t * im_r, dim=-1)

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        return self._distmult_direct(s, r, o)

    def score_loss(self, embedding, triplets, target):
        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        if self.score_function == "complex":
            score = self._complex_direct(s, r, o)
        else:
            score = self._distmult_direct(s, r, o)
        return F.binary_cross_entropy_with_logits(score, target)

    def _score_for_eval(self, h_emb, r_idx, t_embs):
        n   = t_embs.shape[0]
        r   = self.relation_embedding[r_idx]
        h   = h_emb.unsqueeze(0).expand(n, -1)
        r_b = r.unsqueeze(0).expand(n, -1)
        if self.score_function == "complex":
            return self._complex_direct(h, r_b, t_embs)
        return self._distmult_direct(h, r_b, t_embs)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + \
               torch.mean(self.relation_embedding.pow(2))
