from .msg_conv import MSGConv
from .nis import NodeInformationScore

import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict
from torch import Tensor
from torch_geometric.typing import OptTensor, Adj, SparseTensor
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class MSGConvNet(nn.Module):
    def __init__(self,
                 MSGConv_cfg,
                 NeuralNet_cfg):
        super(MSGConvNet, self).__init__()

        if not MSGConv_cfg or not NeuralNet_cfg:
            raise ValueError("MSGConv or ConvNet config file is missing")
        elif not isinstance(MSGConv_cfg, edict) or not isinstance(NeuralNet_cfg, edict):
            raise TypeError("MSGConv or ConvNet config file is not in EasyDict format")


        self.MSGConv = MSGConv(
            in_channels = MSGConv_cfg.in_channels,
            out_channels = MSGConv_cfg.out_channels,
            K = MSGConv_cfg.K,
            scales = MSGConv_cfg.scales,
            improved = MSGConv_cfg.improved,
            add_self_loops = MSGConv_cfg.add_self_loops,
            vec_scaling = MSGConv_cfg.vec_scaling,
            readout = MSGConv_cfg.readout,
        )
        self.NIS = NodeInformationScore()

        self.num_classes = NeuralNet_cfg.num_classes
        self.out_channels = MSGConv_cfg.out_channels
        self.len_scales = len(MSGConv_cfg.scales)
        self.lin1 = torch.nn.Linear(self.out_channels * 2, self.out_channels)
        self.lin2 = torch.nn.Linear(self.out_channels, self.out_channels // 2)
        self.lin3 = torch.nn.Linear(self.out_channels // 2, self.num_classes)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
    ) -> Tensor:

        rout = 0
        out = self.MSGConv(x=x, edge_index=edge_index,
                          edge_weight=edge_weight, batch=batch)
        for i in range(self.len_scales):
            pool_x, pool_batch = self.NIS(out[i], edge_index=edge_index, 
                                          edge_weight=edge_weight, batch=batch, ratio=0.8**(i+1))
            rout += F.relu(torch.cat([gmp(pool_x, pool_batch), gap(pool_x, pool_batch)], dim=1))

        rout = F.relu(self.lin1(rout))
        rout = F.relu(self.lin2(rout))
        rout = F.log_softmax(self.lin3(rout), dim=-1)

        return rout
