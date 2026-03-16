import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add

from utils import uniform
import torch.nn.functional as F


class RGCNW(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout, node_features, embedding_dim,
                 num_layers=2, edge_weight_mode="normalize"):
        super(RGCNW, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.entity_embedding = nn.Embedding.from_pretrained(
            node_features,   # shape: [num_entities, embedding_dim]
            freeze=False     # IMPORTANT: allow training
        )
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, embedding_dim))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.project = nn.Linear(node_features.shape[1], embedding_dim)  # project to n-dim latent space


        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(RGCNConv(embedding_dim,embedding_dim, num_relations, num_bases=num_bases, edge_weight_mode=edge_weight_mode))
        
        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type,  edge_weight=None):
        x = self.entity_embedding(entity)
        x = self.project(x)  # projects pretrained embeddings to latent space

        for i, conv in enumerate(self.convs):
            if(i!=len(self.convs)-1):
                x=F.relu(conv(x,edge_index,edge_type,edge_weight))
                x=F.dropout(x,p=self.dropout_ratio, training = self.training)
            else:
                x=conv(x,edge_index,edge_type,edge_weight)
        return x

    def distmult(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.relation_embedding[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))


class RGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, edge_weight_mode="none", **kwargs):
        super().__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.edge_weight_mode = edge_weight_mode

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if self.edge_weight_mode == "concat":
            # MLP that takes concatenated features (x_j + edge_weight) and outputs transformed features
            self.edge_weight_mlp = nn.Sequential(
                nn.Linear(in_channels + 1, out_channels),
                nn.ReLU()
            )
        elif self.edge_weight_mode == "learnable":
            # Simple 2-layer MLP transforming scalar edge weight to scalar
            self.edge_weight_mlp = nn.Sequential(
                nn.Linear(1, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()  # Keep weights in (0,1)
            )
        elif self.edge_weight_mode == "bayesian":
            # Bayesian edge weights: model weight as w ~ N(μ, σ²)
            # Learn mean and variance from input confidence scores
            # Mean network: refines input confidence
            self.edge_weight_mean_mlp = nn.Sequential(
                nn.Linear(1, 16),  # Input: confidence score
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()  # Mean in [0, 1]
            )
            # Variance network: learns uncertainty from confidence
            # Lower confidence -> higher variance (more uncertainty)
            self.edge_weight_var_mlp = nn.Sequential(
                nn.Linear(1, 16),  # Input: confidence score
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()  # Variance in [0, 1], will be scaled
            )
            # Scale factor for variance (learnable)
            #self.var_scale = nn.Parameter(torch.tensor(1.0))  # Initialize small variance
            self.var_scale = nn.Parameter(torch.ones(num_relations))  # one learnable variance scale per relation
            self.mean_scale = nn.Parameter(torch.ones(num_relations))  # scale for mean per relation
            # Set edge_weight_mlp to None since we use separate mean/var networks
            self.edge_weight_mlp = None
        else:
            self.edge_weight_mlp = None

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

        # Initialize edge_weight_mlp if it exists (for concat/learnable modes)
        if self.edge_weight_mlp is not None:
            if self.edge_weight_mode == "concat":
                nn.init.xavier_uniform_(self.edge_weight_mlp[0].weight)
                nn.init.zeros_(self.edge_weight_mlp[0].bias)
            elif self.edge_weight_mode == "learnable":
                nn.init.xavier_uniform_(self.edge_weight_mlp[0].weight)
                nn.init.zeros_(self.edge_weight_mlp[0].bias)
                nn.init.xavier_uniform_(self.edge_weight_mlp[2].weight)
                nn.init.zeros_(self.edge_weight_mlp[2].bias)
        
        # Initialize Bayesian networks (mean and variance)
        if self.edge_weight_mode == "bayesian":
            # Initialize mean network to pass through confidence (identity-like)
            nn.init.xavier_uniform_(self.edge_weight_mean_mlp[0].weight)
            nn.init.zeros_(self.edge_weight_mean_mlp[0].bias)
            nn.init.xavier_uniform_(self.edge_weight_mean_mlp[2].weight)
            nn.init.zeros_(self.edge_weight_mean_mlp[2].bias)
            # Initialize last layer to be close to identity
            with torch.no_grad():
                self.edge_weight_mean_mlp[4].weight.fill_(0.1)
                self.edge_weight_mean_mlp[4].bias.fill_(0.0)
            
            # Initialize variance network: higher variance for lower confidence
            nn.init.xavier_uniform_(self.edge_weight_var_mlp[0].weight)
            nn.init.zeros_(self.edge_weight_var_mlp[0].bias)
            nn.init.xavier_uniform_(self.edge_weight_var_mlp[2].weight)
            nn.init.zeros_(self.edge_weight_var_mlp[2].bias)
            # Initialize to produce higher variance for lower confidence
            with torch.no_grad():
                self.edge_weight_var_mlp[4].weight.fill_(-0.5)  # Negative to invert confidence
                self.edge_weight_var_mlp[4].bias.fill_(0.5)  # Bias to ensure some variance

    def forward(self, x, edge_index, edge_type, edge_norm=None, edge_weight=None, size=None):
        if edge_weight is None and edge_norm is not None:
            edge_weight = edge_norm
            
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type, edge_weight=edge_weight)

    def message(self, x_j, edge_index_j, edge_type, edge_weight):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        if x_j is None:
            print("x_j is none")
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            if self.edge_weight_mode == "concat":
                assert edge_weight is not None, "edge_weight must be provided for 'concat' mode"
                edge_weight = edge_weight.view(-1, 1)
                x_j_w_concat = torch.cat([x_j, edge_weight], dim=-1)
                out = self.edge_weight_mlp(x_j_w_concat)

            elif self.edge_weight_mode == "learnable":
                assert edge_weight is not None, "edge_weight must be provided for 'learnable' mode"
                # Pass raw edge weight through MLP to get transformed scalar weight
                edge_weight = edge_weight.view(-1, 1)
                transformed_weight = self.edge_weight_mlp(edge_weight)  # Output shape [num_edges, 1]

                w = w.view(self.num_relations, self.in_channels, self.out_channels)
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

                # Scale messages by transformed weights
                out = out * transformed_weight.view(-1, 1)

            elif self.edge_weight_mode == "bayesian":
                assert edge_weight is not None, "edge_weight must be provided for 'bayesian' mode"
                edge_weight = edge_weight.view(-1, 1)  # Shape: [num_edges, 1]
                
                # Compute mean and variance of edge weight distribution
                # μ: refined mean from input confidence
                weight_mean = self.edge_weight_mean_mlp(edge_weight)  # [num_edges, 1]
                weight_mean = weight_mean * self.mean_scale[edge_type].view(-1, 1)  # [num_edges, 1]
                
                # σ²: variance (uncertainty) - higher for lower confidence
                weight_var_raw = self.edge_weight_var_mlp(edge_weight)  # [num_edges, 1]
                #weight_var = weight_var_raw * torch.abs(self.var_scale) + 1e-6  # Ensure positive, scaled
                #weight_var = weight_var_raw * F.softplus(self.var_scale) + 1e-6
                weight_var = weight_var_raw * F.softplus(self.var_scale[edge_type]).view(-1, 1) + 1e-6
                
                # Bayesian uncertainty-weighted aggregation
                # During training: can sample or use mean with uncertainty penalty
                # During inference: use mean / (1 + variance) for uncertainty-weighted aggregation
                if self.training:
                    eps = torch.randn_like(weight_mean)
                    effective_weight = weight_mean + eps * torch.sqrt(weight_var)
                    effective_weight = torch.clamp(effective_weight, 0, 1)
                else:
                    effective_weight = weight_mean / (1.0 + weight_var)
                
                # Apply relation-specific transformation
                w = w.view(self.num_relations, self.in_channels, self.out_channels)
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)
                
                # Scale messages by Bayesian uncertainty-weighted edge weights
                out = out * effective_weight.view(-1, 1)

            else:
                w = w.view(self.num_relations, self.in_channels, self.out_channels)
                w = torch.index_select(w, 0, edge_type)
                out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

                if edge_weight is not None:
                    if self.edge_weight_mode == "normalize":
                        dst_nodes = edge_index_j
                        key = dst_nodes * self.num_relations + edge_type
                        num_nodes = x_j.size(0)
                        norm_denom = scatter(edge_weight, key, dim=0,
                                            dim_size=num_nodes * self.num_relations,
                                            reduce='sum')
                        normed_weights = edge_weight / (norm_denom[key] + 1e-8)
                        out = out * normed_weights.view(-1, 1)
                    elif self.edge_weight_mode == "none":
                        out = out * edge_weight.view(-1, 1)
                    else:
                        raise ValueError(f"Invalid edge_weight_mode: {self.edge_weight_mode}")

        return out

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_channels}, {self.out_channels}, num_relations={self.num_relations}, edge_weight_mode={self.edge_weight_mode})'