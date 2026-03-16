from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ParameterList
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing


class WSGATWrapper(nn.Module):

    def __init__(self,
                 node_features,
                 embedding_dim,
                 num_relations,
                 dropout,
                 num_layers=2,
                 heads=4):

        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout_ratio = dropout
        self.num_layers = num_layers
        self.heads = heads

        # -----------------------------------
        # Node embeddings (pretrained)
        # -----------------------------------
        self.entity_embedding = nn.Embedding.from_pretrained(
            node_features,
            freeze=False
        )

        self.project = nn.Linear(
            node_features.shape[1],
            embedding_dim
        )

        # -----------------------------------
        # Relation embeddings (edge features)
        # -----------------------------------
        self.relation_embedding = nn.Parameter(
            torch.Tensor(num_relations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.relation_embedding)

        self.edge_weight_proj = nn.Linear(1, embedding_dim)

        # -----------------------------------
        # wsGAT layers
        # -----------------------------------
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                wsGATConv(
                    in_channels=embedding_dim,
                    out_channels=embedding_dim // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=embedding_dim   # usamos relation_embedding como edge_attr
                )
            )

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------
    def forward(self, entity, edge_index, edge_type=None, edge_weight=None):

        # Node features
        x = self.entity_embedding(entity)
        x = self.project(x)

        # Edge features
        if edge_weight is not None:
            edge_attr = edge_weight.view(-1, 1)           # [num_edges, 1]
            edge_attr = self.edge_weight_proj(edge_attr)  # Linear(1, edge_dim) → [num_edges, 100]
        elif edge_type is not None:
            edge_attr = self.relation_embedding[edge_type]
            edge_attr = self.edge_weight_proj(edge_attr)
        else:
            edge_attr = None

        # Pass through wsGAT layers
        for i, conv in enumerate(self.convs):

            x = conv(x, edge_index, edge_attr=edge_attr)

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
    


    
class wsGATConv(MessagePassing):
    r"""The wsGAT graph attentional operator from the `"wsGAT: Weighted and Signed Graph Attention Networks for Link Prediction"
    <https://doi.org/10.1007/978-3-030-93409-5_31>`_ paper.

    ```
    Grassia, M., Mangioni, G. (2022). wsGAT: Weighted and Signed Graph Attention Networks for Link Prediction.
    In: Benito, R.M., Cherifi, C., Cherifi, H., Moro, E., Rocha, L.M., Sales-Pardo, M. (eds) Complex Networks & Their Applications X.
    COMPLEX NETWORKS 2021. Studies in Computational Intelligence, vol 1072. Springer, Cham. https://doi.org/10.1007/978-3-030-93409-5_31
    ```

    """

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            attention_layers: int = 1,
            concat: bool = True,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.attention_layers = ParameterList()
        for i in range(attention_layers - 1):
            self.attention_layers.append(
                Parameter(torch.empty(heads, 3 * out_channels, 3 * out_channels))
            )

        self.attention_layers.append(
            Parameter(torch.empty(heads, 1, 3 * out_channels))
        )

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()

        for layer in self.attention_layers:
            glorot(layer)

        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
            edge_attr: OptTensor = None, size: Size = None,
            return_attention_weights=None):

        H, C = self.heads, self.out_channels

        # 1. Node feature transformation
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'wsGATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'wsGATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        # 2. Add self-loops if needed            
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0) if x_dst is None else min(x_src.size(0), x_dst.size(0))

                # Remove existing self-loops
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

                # Add self-loops
                edge_index, tmp_edge_attr = add_self_loops(
                    edge_index,
                    edge_attr=None,  # pass None to control manually
                    fill_value=None, # avoid broadcasting
                    num_nodes=num_nodes
                )

                # Build edge_attr manually
                if edge_attr is None:
                    # original edges
                    num_orig_edges = edge_index.size(1) - num_nodes  # edges before self-loops
                    edge_attr = torch.zeros(num_orig_edges, self.edge_dim, device=x_src.device)
                else:
                    num_orig_edges = edge_attr.size(0)
                    edge_attr = edge_attr[:num_orig_edges]  # keep original edges

                # Add self-loop features (zeros)
                loop_attr = torch.zeros(num_nodes, self.edge_dim, device=x_src.device)
                edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "Adding self-loops with edge_attr in SparseTensor is not supported"
                    )

        x = (x_src, x_dst)

        # 3. Compute attention scores
        # x_src, x_dst already [num_nodes, heads, out_channels]
        x_i_edge = x_dst[edge_index[1]]  # target nodes per edge -> [E, H, C]
        x_j_edge = x_src[edge_index[0]]  # source nodes per edge -> [E, H, C]




        # compute alpha per edge
        alpha = self.edge_update(
            x_i=x_i_edge,
            x_j=x_j_edge,
            edge_attr=edge_attr,
            index=edge_index[1],
            ptr=None,
            size_i=x_dst.size(0)
        )


        alpha = alpha.unsqueeze(-1)  # -> [E, H, 1]

        # compute messages
        msg = alpha * x_j_edge  # -> [E, H, C]

        # aggregate messages to target nodes
        from torch_scatter import scatter
        out = scatter(msg, edge_index[1], dim=0, dim_size=x_dst.size(0), reduce='sum')

        # combine heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # add bias
        if self.bias is not None:
            out = out + self.bias

        # 6. Return attention weights if requested
        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def edge_update(self, x_i: Tensor, x_j: Tensor, edge_attr: Optional[Tensor] = None,
                    index: Tensor = None, ptr: OptTensor = None, size_i: Optional[int] = None):
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            # edge_attr: [num_edges, edge_dim]
            edge_attr = self.lin_edge(edge_attr)           # -> [num_edges, heads*out_channels]
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)  # -> [num_edges, heads, out_channels]
            alpha = torch.cat([x_i, x_j, edge_attr], dim=2)  # -> [num_edges, heads, 3*out_channels]
        else:
            alpha = torch.cat([x_i, x_j], dim=2)

        for i, layer in enumerate(self.attention_layers):
            # matmul safely and remove extra dims
            alpha = torch.matmul(layer, alpha.unsqueeze(-1))  # [num_edges, heads, 1, 1]
            alpha = alpha[:, :, 0, 0]  # -> [num_edges, heads]
            alpha = alpha.tanh()

        alpha = alpha.sign() * softmax(alpha.abs(), index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'attention_layers={len(self.attention_layers)}')