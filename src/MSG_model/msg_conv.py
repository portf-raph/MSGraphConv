from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor, Adj, SparseTensor
from torch_geometric.nn import global_mean_pool


class MSGConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        scales: list[float],
        improved: bool = False,
        add_self_loops: Optional[bool] = None,
        vec_scaling: bool = True,
        readout: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr','add')
        super().__init__(**kwargs)

        assert K>0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.scales = scales
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                  weight_initializer='glorot') for _ in range(K)
        ])

        self.improved = improved
        self.add_self_loops = add_self_loops

        self.vec_scaling = vec_scaling
        self.readout = readout
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
          lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
    ) -> Tensor:

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index=edge_index, edge_weight=edge_weight,
                improved=self.improved, num_nodes=x.size(self.node_dim),
                add_self_loops=self.add_self_loops,
            )


        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index=edge_index, edge_weight=edge_weight,
                improved=self.improved, num_nodes=x.size(self.node_dim),
                add_self_loops=self.add_self_loops,
            )

        # polynomial
        convs = [self.lins[0](x)]                 # [xW_0, SxW_1, S^{2}xW_2, ..., S^{K-1}xW_{K-1}]
        P = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        convs.append(self.lins[1](P))

        for i in range(2, self.K):
            P = self.propagate(edge_index, x=P, edge_weight=edge_weight)
            convs.append(self.lins[i](P))

        if not self.vec_scaling:
            out = []
            for t in self.scales:
              out_t = convs[self.K-1]             # K-1
              for i in range(self.K-2, -1, -1):   # K-2 -> 0
                out_t = t*out_t + convs[i]        # \sum_{k=0}^{K-1} (tS)^k x W_k
              out.append(out_t)
            out = torch.stack(out)

        else:
            # vectorized scaling                                              # \sum_{k=0}^{K-1} (tS)^k x W_k
            scales = torch.Tensor(self.scales)
            scales = torch.stack([torch.pow(scales, i) for i in range(self.K)])
            scales = torch.transpose(scales, 0, 1)                            # (len(self.scales), K)

            convs_stack = torch.stack(convs)                                  # (K, num_nodes, out_channels)
            convs_stack = convs_stack.unsqueeze(0)                            # (1, K, num_nodes, out_channels)
            scales = scales.view(len(self.scales), self.K, 1, 1)              # (len(self.scales), K, 1, 1)

            out = (convs_stack * scales).sum(dim=1)                           # (len(self.scales), num_nodes, out_channels)

        if self.readout:
            if batch is None:
                batch = torch.zeros(x.size(self.node_dim), dtype=torch.long)

            batch_size = batch.max().item() + 1
            batched_img = []
            for i in range(len(self.scales)):
                batched_img.append(global_mean_pool(out[i], batch=batch, size=batch_size)) # (batch_size, out_channels)

            batched_img = torch.stack(batched_img)                                         # (scale, (batch_size, out_channels))
            return torch.permute(batched_img, (1, 0, 2))                                   # (batch_size, (scale, out_channels))

        return out


    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1,1) * x_j


