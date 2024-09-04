from typing import Optional

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import OptTensor, Adj, SparseTensor
from .topk import topk


class NodeInformationScore(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        ratio: float = 0.8,
    ) -> Tensor:

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index=edge_index, edge_weight=edge_weight,
                num_nodes=x.size(self.node_dim),
            )

        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index=edge_index, edge_weight=edge_weight,
                num_nodes=x.size(self.node_dim),
            )

        prop = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        out = x - prop
        out = torch.linalg.norm(out, ord=1, dim=1)

        perm = topk(out, ratio, batch)
        x = x[perm]
        batch = batch[perm]
        return x, batch

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1,1) * x_j
