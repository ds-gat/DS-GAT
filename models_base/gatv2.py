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
            weights=False):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout_ratio = dropout
        self.num_layers = num_layers
        self.heads = heads
        self.weights= weights

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
    # DistMult scoring
    # ------------------------------------------------
    def distmult(self, embedding, triplets):

        s = embedding[triplets[:, 0]]
        r = self.relation_embedding[triplets[:, 1]]
        o = embedding[triplets[:, 2]]

        score = torch.sum(s * r * o, dim=1)
        return score

    def score_loss(self, embedding, triplets, target):

        score = self.distmult(embedding, triplets)
        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):

        return torch.mean(embedding.pow(2)) + \
               torch.mean(self.relation_embedding.pow(2))
