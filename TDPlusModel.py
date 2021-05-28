import torch
from typing import Callable, Optional
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree
from scipy.special import factorial
from torch_geometric.data import Data, InMemoryDataset
from torch.nn import ModuleList, Dropout, ReLU, ELU
from typing import List
from args import get_citation_args

class TDPlusConv(MessagePassing):
    def __init__(self, in_channels, init_t):
        super(TDPlusConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        args = get_citation_args()
        self.init_t = init_t
        self.step = 10
        if not args.denseT:
            self.t = Parameter(torch.Tensor(self.step, in_channels))
        else:
            self.t = Parameter(torch.Tensor(self.step))
        # self.t.data.fill_(2)
        self.reset_parameters()
        # self.t.requires_grad = False


    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # print(self.t)
        # Step 1: Add self-loops to the adjacency matrix.
        self.t_norm = torch.nn.functional.softmax(self.t, dim=0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        x_list = [0 for i in range(self.step)]
        x_list[0] = x

        for i in range(1, self.step):
            x_list[i] = self.propagate(edge_index, x=x_list[i - 1], norm=norm)
        
        y = 0

        for k in range(self.step):
            x_list[k] = self.t_norm[k] * x_list[k] ## important!
            # x_list[k] = torch.pow(self.t, k) / factorial(k) * x_list[k]
            if k != 0: 
                y += x_list[k]
            else:
                y = x_list[k]
        return y
    
    def reset_parameters(self):
        torch.nn.init.constant_(self.t, self.init_t)
        #self.t.requires_grad = False
    

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class GCNPlusConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, init_t):
        super(GCNPlusConv, self).__init__()
        self.diffusion = TDPlusConv(in_channels, init_t)
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.diffusion(x, edge_index)
        x = self.lin(x)
        return x
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.diffusion.reset_parameters()

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
        self.diffusion = TDPlusConv(in_channels, init_t)
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

class GCN(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 t: float,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GCN, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            # layers.append(SGConv(in_features, out_features, K=2))
            layers.append(GCNPlusConv(in_features, out_features, init_t=t))
        self.layers = ModuleList(layers)

        # self.reg_params = list(layers[0].parameters())
        # self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight=edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1)

class JKNet(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 t: float,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(JKNet, self).__init__()
        args = get_citation_args()
        num_features = [dataset.data.x.shape[1]] + hidden
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GCNPlusConv(in_features, out_features, init_t=t))
        layers.append(torch.nn.Linear(sum(hidden), dataset.num_classes))
        self.layers = ModuleList(layers)

        # self.reg_params = list(layers[0].parameters())
        # self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        if args.shareT == True:
            for num in range(len(layers)):
                self.layers[num].diffusion = self.layers[0].diffusion

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        layer_outputs = []
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_weight=edge_attr)


            x = self.act_fn(x)
            x = self.dropout(x)
            layer_outputs.append(x)
        
        x = torch.cat(layer_outputs, dim=1)
        x = self.layers[-1](x)
        return torch.nn.functional.log_softmax(x, dim=1)

class ARMA(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 t: float,
                 stacks: int,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(ARMA, self).__init__()
        args = get_citation_args()
        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(ARMAPlusConv(in_features, out_features, init_t = t, num_stacks = stacks, num_layers = 1, shared_weights = False, dropout = dropout))
        self.layers = ModuleList(layers)

        if args.shareT == True:
            for num in range(len(layers)):
                self.layers[num].diffusion = self.layers[0].diffusion

        # self.reg_params = list(layers[0].parameters())
        # self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight=edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1)