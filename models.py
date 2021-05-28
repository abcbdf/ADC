__author__ = "Stefan Weißenberger and Johannes Klicpera"
__license__ = "MIT"

from typing import List

import torch
from torch.nn import ModuleList, Dropout, ReLU, ELU, Linear, Sequential, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, ARMAConv
from torch_geometric.data import Data, InMemoryDataset


class GCN(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GCN, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GCNConv(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

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

class GAT(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(GAT, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GATConv(in_features, 8, heads=8, dropout=0.6))
        layers.append(GATConv(num_features[-1], dataset.num_classes, heads=1, concat=False, dropout=0.6))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ELU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

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
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(JKNet, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GCNConv(in_features, out_features))
        layers.append(torch.nn.Linear(sum(hidden), dataset.num_classes))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

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

# class GIN(torch.nn.Module):
#     def __init__(self,
#                  dataset: InMemoryDataset,
#                  hidden: List[int] = [64],
#                  dropout: float = 0.5):
#         super(GIN, self).__init__()
#         num_features = [dataset.data.x.shape[1]] + hidden
#         layers = []
#         for in_features, out_features in zip(num_features[:-1], num_features[1:]):
#             layers.append(GINConv(Sequential(
#                 Linear(in_features, out_features),
#                 ReLU(),
#                 Linear(out_features, out_features),
#                 ReLU(),
#                 BN(out_features),
#             ), train_eps=True))
#         layers.append(Linear(num_features[-1], num_features[-1]))
#         layers.append(Linear(num_features[-1], dataset.num_classes))
#         self.layers = ModuleList(layers)

#         self.reg_params = list(layers[0].parameters())
#         self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

#         self.dropout = Dropout(p=dropout)
#         self.act_fn = ReLU()

#         # self.conv1 = GINConv(
#         #     Sequential(
#         #         Linear(dataset.num_features, hidden),
#         #         ReLU(),
#         #         Linear(hidden, hidden),
#         #         ReLU(),
#         #         BN(hidden),
#         #     ), train_eps=True)
#         # self.convs = torch.nn.ModuleList()
#         # for i in range(num_layers - 1):
#         #     self.convs.append(
#         #         GINConv(
#         #             Sequential(
#         #                 Linear(hidden, hidden),
#         #                 ReLU(),
#         #                 Linear(hidden, hidden),
#         #                 ReLU(),
#         #                 BN(hidden),
#         #             ), train_eps=True))
#         # self.lin1 = Linear(hidden, hidden)
#         # self.lin2 = Linear(hidden, dataset.num_classes)

#     def reset_parameters(self):
#         for layer in self.layers:
#             layer.reset_parameters()

#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         # print(type(x.shape[0]))
#         #batch = torch.zeros(x.shape[0]).type(torch.LongTensor).to(data.x.get_device())
#         # print(batch)

#         for i, layer in enumerate(self.layers[:-2]):
#             x = layer(x, edge_index)
        
#         # x = global_mean_pool(x, batch)
#         # print(x.shape)
#         x = self.act_fn(self.layers[-2](x))
#         x = self.dropout(x)
#         x = self.layers[-1](x)


#         return torch.nn.functional.log_softmax(x, dim=1)
#         # x, edge_index, batch = data.x, data.edge_index, data.batch
#         # x = self.conv1(x, edge_index)
#         # for conv in self.convs:
#         #     x = conv(x, edge_index)
#         # x = global_mean_pool(x, batch)
#         # x = F.relu(self.lin1(x))
#         # x = F.dropout(x, p=0.5, training=self.training)
#         # x = self.lin2(x)
#         # return F.log_softmax(x, dim=-1)

#     def __repr__(self):
#         return self.__class__.__name__

class ARMA(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 stacks: int,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(ARMA, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(ARMAConv(in_features, out_features, stacks, 1, False, dropout = dropout))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

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
