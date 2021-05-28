__author__ = "Stefan Weißenberger and Johannes Klicpera"
__license__ = "MIT"

from typing import List

import torch
from torch.nn import ModuleList, Dropout, ReLU, ELU
from torch_geometric.nn import GCNConv, SGConv, GATConv, ARMAConv
from torch_geometric.data import Data, InMemoryDataset
from gcnplus import GCNPlusConv, GCNPlusPlusConv
from gatplus import GATPlusConv
from TDConv import TDConv


# class GCN(torch.nn.Module):
#     def __init__(self,
#                  dataset: InMemoryDataset,
#                  t: float,
#                  hidden: List[int] = [64],
#                  dropout: float = 0.5):
#         super(GCN, self).__init__()

#         num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
#         layers = []
#         for in_features, out_features in zip(num_features[:-1], num_features[1:]):
#             # layers.append(SGConv(in_features, out_features, K=2))
#             layers.append(GCNPlusConv(in_features, out_features, init_t=t))
#         self.layers = ModuleList(layers)

#         # self.reg_params = list(layers[0].parameters())
#         # self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

#         self.dropout = Dropout(p=dropout)
#         self.act_fn = ReLU()

#     def reset_parameters(self):
#         for layer in self.layers:
#             layer.reset_parameters()
    
#     def reset_linear(self):
#         for layer in self.layers:
#             layer.lin.reset_parameters()

#     def forward(self, data: Data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

#         for i, layer in enumerate(self.layers):
#             x = layer(x, edge_index, edge_weight=edge_attr)

#             if i == len(self.layers) - 1:
#                 break

#             x = self.act_fn(x)
#             x = self.dropout(x)

#         return torch.nn.functional.log_softmax(x, dim=1)

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
            layers.append(GCNConv(in_features, out_features))
        self.layers = ModuleList(layers)
        self.diffusion = TDConv(num_features[0], t)
        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.diffusion.reset_parameters()


    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.diffusion(x, edge_index)
        for i, layer in enumerate(self.layers):
            # if edge_attr is not None:
            #     x = layer(x, edge_index, edge_weight=edge_attr)
            # else:
            #     x = layer(x, edge_index)
            x = layer(x, edge_index, edge_weight=edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 t: float,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GAT, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GATConv(in_features, 8, heads=8, dropout=0.6))
        layers.append(GATConv(num_features[-1], dataset.num_classes, heads=1, concat=False, dropout=0.6))
        self.layers = ModuleList(layers)
        self.diffusion = TDConv(num_features[0], t)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ELU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.diffusion.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.diffusion(x, edge_index)
        x = self.dropout(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)

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

        num_features = [dataset.data.x.shape[1]] + hidden
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GCNConv(in_features, out_features))
        layers.append(torch.nn.Linear(sum(hidden), dataset.num_classes))
        self.layers = ModuleList(layers)

        self.diffusion = TDConv(num_features[0], t)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.diffusion.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        layer_outputs = []
        
        x = self.diffusion(x, edge_index)
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

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(ARMAConv(in_features, out_features, stacks, 1, False, dropout = dropout))
        self.layers = ModuleList(layers)

        self.diffusion = TDConv(num_features[0], t)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.diffusion.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.diffusion(x, edge_index)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight=edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1)