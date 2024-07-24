import pytest
import torch

import torch_geometric.typing
from torch_geometric.testing import is_full_test
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import to_torch_coo_tensor

from src.MSG_model.msg_conv import MSGConv


def test_MSGConv():
    in_channels, out_channels = (16, 32)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    value = torch.rand(edge_index.size(1))
    num_nodes = edge_index.max().item() + 1
    x = torch.randn((num_nodes, in_channels))

    scales = [1,5,10]
    conv = MSGConv(in_channels, out_channels, K=3, scales=scales)
    len_scales = len(scales)
    assert str(conv) == 'MSGConv(16, 32)'

    adj1 = to_torch_coo_tensor(edge_index, size=(4, 4))
    adj2 = to_torch_coo_tensor(edge_index, value, size=(4, 4))

    out1 = conv(x, edge_index)
    assert out1.size() == (1, len_scales, out_channels)
    assert torch.allclose(conv(x, adj1.t()), out1, atol=1e-6)

    out2 = conv(x, edge_index, value)
    assert out2.size() == (1, len_scales, out_channels)
    assert torch.allclose(conv(x, adj2.t()), out2, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(4, 4))
        adj4 = SparseTensor.from_edge_index(edge_index, value, sparse_sizes=(4, 4))
        assert torch.allclose(conv(x, adj3.t()), out1, atol=1e-6)
        assert torch.allclose(conv(x, adj4.t()), out2, atol=1e-6)

    if is_full_test():
        jit = torch.jit.script(conv)
        assert torch.allclose(jit(x, edge_index), out1, atol=1e-6)
        assert torch.allclose(jit(x, edge_index, value), out2, atol=1e-6)

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            assert torch.allclose(jit(x, adj3.t()), out1, atol=1e-6)
            assert torch.allclose(jit(x, adj4.t()), out2, atol=1e-6)
