"""Torch modules for graph attention networks with fully valuable edges (EGAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from utils import uniform


class EGATWrapper(nn.Module):
    def __init__(self, node_features,num_relations, embedding_dim, dropout, num_layers=2, num_heads=4,
             use_bayesian=True
):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_bayesian = use_bayesian

        # Node embeddings
        self.entity_embedding = nn.Embedding.from_pretrained(
            node_features, freeze=False
        )
        self.project = nn.Linear(node_features.shape[1], embedding_dim)
        # Project scalar edge weight (confidence) to embedding_dim
        
        if self.use_bayesian==False:
            self.edge_project = nn.Linear(embedding_dim + 1, embedding_dim)

        # Relation embeddings (optional, used to produce edge features)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, embedding_dim))
        nn.init.xavier_uniform_(self.relation_embedding)

        self.dropout_ratio = dropout

        # EGAT layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                EGATConv(
                    in_node_feats=embedding_dim,
                    in_edge_feats=embedding_dim,
                    out_node_feats=embedding_dim,
                    out_edge_feats=embedding_dim,
                    num_heads=num_heads,
                    num_relations=num_relations,
    use_bayesian=use_bayesian,
    default_edge_weight=1.0
                )
            )

    def forward(self, entity, edge_index, edge_type=None, edge_weight=None):
        x = self.entity_embedding(entity)
        x = self.project(x)

        # ------------------------------------------------
        # Edge features
        # ------------------------------------------------
        rel_emb = self.relation_embedding[edge_type]  # [E, embedding_dim]

        if edge_weight is not None:
            edge_weight = edge_weight.unsqueeze(1)  # [E, 1]
        else:
            edge_weight = torch.ones(rel_emb.size(0), 1, device=x.device)

        # Concatenate relation + weight
        # Project back to embedding_dim
        if self.use_bayesian:
            efeats = rel_emb
        else:
            efeats = torch.cat([rel_emb, edge_weight], dim=1)  # [E, embedding_dim+1]
            efeats = self.edge_project(efeats)


        # DGL expects graph object
        src, dst = edge_index
        graph = dgl.graph((src, dst), num_nodes=x.size(0))
        graph.ndata['h'] = x
        graph.edata['f'] = efeats

        for i, conv in enumerate(self.convs):
            x, efeats = conv(graph, x, efeats, edge_type=edge_type, edge_weight=edge_weight)
            x = x.mean(dim=1)
            efeats = efeats.mean(dim=1)
            if i != len(self.convs)-1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout_ratio, training=self.training)

        return x
    

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
    

# pylint: enable=W0235
class EGATConv(nn.Module):
    r"""
    
    Description
    -----------
    Apply Graph Attention Layer over input graph. EGAT is an extension
    of regular `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__ 
    handling edge features, detailed description is available in
    `Rossmann-Toolbox <https://pubmed.ncbi.nlm.nih.gov/34571541/>`__ (see supplementary data).
     The difference appears in the method how unnormalized attention scores :math:`e_{ij}`
     are obtain:
        
    .. math::
        e_{ij} &= \vec{F} (f_{ij}^{\prime})

        f_{ij}^{\prim} &= \mathrm{LeakyReLU}\left(A [ h_{i} \| f_{ij} \| h_{j}]\right)

    where :math:`f_{ij}^{\prim}` are edge features, :math:`\mathrm{A}` is weight matrix and 
    :math: `\vec{F}` is weight vector. After that resulting node features 
    :math:`h_{i}^{\prim}` are updated in the same way as in regular GAT. 
   
    Parameters
    ----------
    in_node_feats : int
        Input node feature size :math:`h_{i}`.
    in_edge_feats : int
        Input edge feature size :math:`f_{ij}`.
    out_node_feats : int
        Output nodes feature size.
    out_edge_feats : int
        Output edge feature size.
    num_heads : int
        Number of attention heads.
        
    Examples
    ----------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import EGATConv
    >>> 
    >>> num_nodes, num_edges = 8, 30
    >>>#define connections
    >>> u, v = th.randint(num_nodes, num_edges), th.randint(num_nodes, num_edges) 
    >>> graph = dgl.graph((u,v))    

    >>> node_feats = th.rand((num_nodes, 20)) 
    >>> edge_feats = th.rand((num_edges, 12))
    >>> egat = EGATConv(in_node_feats=20,
                          in_edge_feats=12,
                          out_node_feats=15,
                          out_edge_feats=10,
                          num_heads=3)
    >>> #forward pass                    
    >>> new_node_feats, new_edge_feats = egat(graph, node_feats, edge_feats)
    >>> new_node_feats.shape, new_edge_feats.shape
    ((8, 3, 12), (30, 3, 10))
    """
    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,
                    num_relations,
    use_bayesian=True,
    default_edge_weight=1.0,
                 **kw_args):
        
        super().__init__()
        self._num_heads = num_heads
        self._out_node_feats = out_node_feats
        self._out_edge_feats = out_edge_feats
        self.fc_nodes = nn.Linear(in_node_feats, out_node_feats*num_heads, bias=True)
        self.fc_edges = nn.Linear(in_edge_feats + 2*in_node_feats, out_edge_feats*num_heads, bias=False)
        self.fc_attn = nn.Linear(out_edge_feats, num_heads, bias=False)
        self.use_bayesian=use_bayesian
        if use_bayesian:
            self.edge_project = nn.Linear(in_edge_feats + 1, out_edge_feats)
            self.edge_weight_mean_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )

            self.edge_weight_var_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                 nn.Sigmoid()
            )

            self.var_scale = nn.Parameter(torch.ones(num_relations))
            self.mean_scale = nn.Parameter(torch.ones(num_relations))

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_nodes.weight, gain=gain)
        init.xavier_normal_(self.fc_edges.weight, gain=gain)
        init.xavier_normal_(self.fc_attn.weight, gain=gain)

        if self.use_bayesian:
            # Initialize mean network to pass through confidence (identity-like)
            nn.init.xavier_uniform_(self.edge_weight_mean_mlp[0].weight)
            nn.init.zeros_(self.edge_weight_mean_mlp[0].bias)
            nn.init.xavier_uniform_(self.edge_weight_mean_mlp[2].weight)
            nn.init.zeros_(self.edge_weight_mean_mlp[2].bias)
            with torch.no_grad():
                self.edge_weight_mean_mlp[4].weight.fill_(0.1)
                self.edge_weight_mean_mlp[4].bias.fill_(0.0)

            # Initialize variance network: higher variance for lower confidence
            nn.init.xavier_uniform_(self.edge_weight_var_mlp[0].weight)
            nn.init.zeros_(self.edge_weight_var_mlp[0].bias)
            nn.init.xavier_uniform_(self.edge_weight_var_mlp[2].weight)
            nn.init.zeros_(self.edge_weight_var_mlp[2].bias)
            with torch.no_grad():
                self.edge_weight_var_mlp[4].weight.fill_(-0.5)
                self.edge_weight_var_mlp[4].bias.fill_(0.5)


    def edge_attention(self, edges):
        #extract features
        h_src = edges.src['h']
        h_dst = edges.dst['h']
        f = edges.data['f']
        #stack h_i | f_ij | h_j
        stack = th.cat([h_src, f, h_dst], dim=-1)
        # apply FC and activation
        f_out = self.fc_edges(stack)
        f_out = nn.functional.leaky_relu(f_out)
        f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)
        # apply FC to reduce edge_feats to scalar
        a = self.fc_attn(f_out).sum(-1).unsqueeze(-1)

        return {'a': a, 'f' : f_out}

    def message_func(self, edges):
        return {'h': edges.src['h'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        alpha = nn.functional.softmax(nodes.mailbox['a'], dim=1)
        h = th.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}

    def forward(self, graph, nfeats, efeats,edge_type=None, edge_weight=None):
        r"""
        Compute new node and edge features.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        nfeats : torch.Tensor
            The input node feature of shape :math:`(*, D_{in})`
            where:
                :math:`D_{in}` is size of input node feature,
                :math:`*` is the number of nodes.
        efeats: torch.Tensor
             The input edge feature of shape :math:`(*, F_{in})`
             where:
                 :math:`F_{in}` is size of input node feauture,
                 :math:`*` is the number of edges.
       
            
        Returns
        -------
        pair of torch.Tensor
            node output features followed by edge output features
            The node output feature of shape :math:`(*, H, D_{out})` 
            The edge output feature of shape :math:`(*, H, F_{out})`
            where:
                :math:`H` is the number of heads,
                :math:`D_{out}` is size of output node feature,
                :math:`F_{out}` is size of output edge feature.            
        """
        
        with graph.local_scope():
        ##TODO allow node src and dst feats
            if(self.use_bayesian):
                rel_emb = efeats

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


                efeats = torch.cat([rel_emb, edge_weight], dim=1)  # [E, embedding_dim+1]

                # Project back to embedding_dim
                efeats = self.edge_project(efeats)          


            graph.edata['f'] = efeats
            graph.ndata['h'] = nfeats

            graph.apply_edges(self.edge_attention)

            nfeats_ = self.fc_nodes(nfeats)
            nfeats_ = nfeats_.view(-1, self._num_heads, self._out_node_feats)

            graph.ndata['h'] = nfeats_
            graph.update_all(message_func = self.message_func,
                         reduce_func = self.reduce_func)

            return graph.ndata.pop('h'), graph.edata.pop('f')
    