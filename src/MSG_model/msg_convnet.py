from .msg_conv import MSGConv

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import OptTensor, Adj, SparseTensor

class ConvNet(nn.Module):
    def __init__(self,
                 num_classes):
        super(ConvNet, self).__init__()

        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.BatchNorm2d(8),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        # 128x128 -> 8x8
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 64, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x

class MSGConvNet(nn.Module):
    def __init__(self,
                 MSGConv_cfg,                    
                 ConvNet_cfg):
        super(MSGConvNet, self).__init__()

        if not MSGConv_cfg or not ConvNet_cfg:
            raise ValueError("MSGConv or ConvNet config file is missing")
        elif not isinstance(MSGConv_cfg, dict) or not isinstance(ConvNet_cfg, dict):
            raise TypeError("MSGConv or ConvNet config file is not in dictionary format")

        self.MSGConv = MSGConv(
            in_channels = MSGConv_cfg['in_channels'],
            out_channels = MSGConv_cfg['out_channels'],
            K = MSGConv_cfg['K'],
            scales = MSGConv_cfg['scales'],
            improved = MSGConv_cfg['improved'],
            add_self_loops = MSGConv_cfg['add_self_loops'],
            vec_scaling = MSGConv_cfg['vec_scaling'],
            readout = MSGConv_cfg['readout']
        )

        self.ConvNet = ConvNet(num_classes = ConvNet_cfg['num_classes'])

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
    ) -> Tensor:

        out = self.MSGConv(x=x, edge_index=edge_index,
                          edge_weight=edge_weight, batch=batch)
        out = out.unsqueeze(dim=1)
        out = self.ConvNet(out)
        return out
