import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepquantum import Circuit
from torch import Tensor, optim
from torch.nn import Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import BatchNorm, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree
from torch_scatter import scatter

class QuLinear(nn.Module):
    """
    quantum linear layer
    """
    def __init__(self, input_dim,  n_layers=5, gain=2 ** 0.5, use_wscale=True, lrmul=1):
        super(QuLinear, self).__init__()

        he_std = gain * 5 ** (-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.input_dim = input_dim
        self.n_qubits = int(math.log(input_dim, 2))
        self.N3 = 3 * self.n_qubits
        self.dim = (1 << self.n_qubits)  # 2**n_qubits
        self.n_layers = n_layers

        self.n_param = self.N3 * (self.n_layers + 1)
        self.weight = nn.Parameter(nn.init.uniform_(torch.empty(self.n_param), a=0.0, b=2 * np.pi) * init_std * self.w_mul)
self.cir = Circuit(self.n_qubits)
        self.wires_lst = list(range(self.n_qubits))
        self.is_batch = False
    
    def encoding_layer(self, x):
        out = F.normalize(x, dim=-1) + 0j
        return out

    def forward(self, x):
        x = x.unsqueeze(1)
        self.cir.clear()
        if x.ndim == 3:
            assert (x.shape[1] == 1) and (x.shape[2] <= self.dim), "批处理情况时，输入数据的1轴数据长度为1、2轴数据长度不超过2的比特数次幂"
            if x.shape[2] < self.dim:
                pad = nn.ZeroPad2d(padding=(0, self.dim - x.shape[2], 0, 0))
                x = pad(x)
            self.is_batch = True
            x = self.encoding_layer(x)
        else:
            raise ValueError("输入数据的维度大小限定为2(非批处理)或3(批处理)")

        for i in range(self.n_layers):
            index = i * self.N3
            self.cir.XYZLayer(self.wires_lst, self.weight[index: index + self.N3])
            self.cir.ring_of_cnot(self.wires_lst)
        index += self.N3
        self.cir.YZYLayer(self.wires_lst, self.weight[index:])

        if self.is_batch:
            x = x.view([x.shape[0]] + [2] * self.n_qubits)
            res = self.cir.TN_contract_evolution(x, batch_mod=True)
            res = res.reshape(res.shape[0], 1, -1)
            assert res.shape[2] == self.dim, "线性层MPS演化结果数据2轴数据大小要求为2的比特数次幂"
        else:
            # x = nn.functional.normalize(x, dim=1)
            x = self.cir.state_init()
            x = x.view([2] * self.n_qubits)
            res = self.cir.TN_contract_evolution(x, batch_mod=False)
            res = res.reshape(1, -1)
assert res.shape[1] == self.dim, "线性层MPS演化结果数据1轴数据大小要求为2的比特数次幂"

        return res.squeeze(1).real
    
    
class QuMLP(torch.nn.Module):
    def __init__(self, num_layers: int, dropout: float = 0.,
                 batch_norm: bool = True, relu_first: bool = False):
        super(QuMLP, self).__init__()
        assert num_layers >= 2
        self.num_layers = num_layers
        self.dropout = dropout
        self.relu_first = relu_first

        self.lins = torch.nn.ModuleList()
        for dims in range(num_layers):
            self.lins.append(QuLinear(dims))

        self.norms = torch.nn.ModuleList()
        for dim in range(num_layers):
            self.norms.append(BatchNorm1d(dim) if batch_norm else Identity())

    def forward(self, x: Tensor) -> Tensor:
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.relu_first:
                x = F.relu(x)
            x = norm(x)
            if not self.relu_first:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x)
        return x

class QuGINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(QuGINConv, self).__init__(aggr = "add")
self.mlp = nn.Sequential(QuLinear(emb_dim, n_layers=5), nn.BatchNorm1d(emb_dim), nn.ReLU(), QuLinear(emb_dim, n_layers=5))
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr) # First, convert the category edge attribute to edge representation
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        #`message() ` X in function_ j + edge_ The attr ` operation performs the fusion of node information and edge information
        return F.relu(x_j + edge_attr)
        
    def update(self, aggr_out):
        return aggr_out

