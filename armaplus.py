from typing import Callable, Optional
import torch
from torch.nn import Parameter, Dropout, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree
from scipy.special import factorial
from TDConv import TDConv



class ARMAPlusConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, init_t: float,
                 num_stacks: int = 1, num_layers: int = 1,
                 shared_weights: bool = False,
                 act: Optional[Callable] = ReLU(), dropout: float = 0.,
                 bias: bool = True):
        super(ARMAPlusConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.act = act
        self.shared_weights = shared_weights

        assert(num_layers == 1)
        self.diffusion = TDConv(in_channels, init_t)
        K, T, F_in, F_out = num_stacks, num_layers, in_channels, out_channels
        self.init_weight = Parameter(torch.Tensor(K, F_in, F_out))
        self.root_weight = Parameter(torch.Tensor(T, K, F_in, F_out))
        self.bias = Parameter(torch.Tensor(T, K, 1, F_out))
        self.dropout = Dropout(p=dropout)

        self.reset_parameters()


    def forward(self, x, edge_index, edge_weight=None):
        
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # # Step 2: Linearly transform node feature matrix.

        # # Step 3: Compute normalization.
        # row, col = edge_index
        # deg = degree(col, x.size(0), dtype=x.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.diffusion(x, edge_index)

        x = x.unsqueeze(-3)
        out = x
        out = out @ self.init_weight

        root = self.dropout(x)
        out += root @ self.root_weight[0]

        out += self.bias[0]

        out = self.act(out)

        return out.mean(dim=-3)

    
    def reset_parameters(self):
        glorot(self.init_weight)
        glorot(self.root_weight)
        zeros(self.bias)
        self.diffusion.reset_parameters()

    # def message(self, x_j, norm):
    #     # x_j has shape [E, out_channels]

    #     # Step 4: Normalize node features.
    #     return norm.view(-1, 1) * x_j
