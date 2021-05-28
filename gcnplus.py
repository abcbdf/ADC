import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from scipy.special import factorial
from TDConv import TDConv

class GCNPlusConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, init_t):
        super(GCNPlusConv, self).__init__()
        self.diffusion = TDConv(in_channels, init_t)
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.diffusion(x, edge_index)
        x = self.lin(x)
        return x
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.diffusion.reset_parameters()

class GCNPlusRConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, init_t):
        super(GCNPlusRConv, self).__init__()
        self.diffusion = TDConv(out_channels, init_t)
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.lin(x)
        x = self.diffusion(x, edge_index)
        return x
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.diffusion.reset_parameters()

# class GCNPlusConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, init_t):
#         super(GCNPlusConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
#         self.init_t = init_t
#         self.lin = torch.nn.Linear(in_channels, out_channels)
#         self.step = 20
#         self.t = Parameter(torch.Tensor(in_channels))
#         # self.t.data.fill_(2)
#         self.reset_parameters()
#         # self.t.requires_grad = False


#     def forward(self, x, edge_index, edge_weight=None):
#         # x has shape [N, in_channels]
#         # edge_index has shape [2, E]
#         # print(self.t)
#         # Step 1: Add self-loops to the adjacency matrix.
#         edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

#         # Step 2: Linearly transform node feature matrix.

#         # Step 3: Compute normalization.
#         row, col = edge_index
#         deg = degree(col, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         # Step 4-5: Start propagating messages.
#         x_list = [0 for i in range(self.step)]
#         x_list[0] = x

#         for i in range(1, self.step):
#             x_list[i] = self.propagate(edge_index, x=x_list[i - 1], norm=norm)
        
#         y = 0

#         for k in range(self.step):
#             x_list[k] = torch.exp(-self.t) * torch.pow(self.t, k) / factorial(k) * x_list[k] ## important!
#             # x_list[k] = torch.pow(self.t, k) / factorial(k) * x_list[k]
#             if k != 0: 
#                 y += x_list[k]
#             else:
#                 y = x_list[k]
        
#         y = self.lin(y)
#         return y
    
#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         #torch.nn.init.normal_(self.t, mean=4, std=1)
#         torch.nn.init.constant_(self.t, self.init_t)
#         #self.t.requires_grad = False
    

#     def message(self, x_j, norm):
#         # x_j has shape [E, out_channels]

#         # Step 4: Normalize node features.
#         return norm.view(-1, 1) * x_j

class GCNPlusPlusConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNPlusPlusConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        # self.lin = torch.nn.Linear(in_channels, out_channels)
        self.step = 20
        self.t = Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()


    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # print(self.t)
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        # x = self.lin(x)

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
            # x_list[k] = x_list[k] @ (torch.exp(-self.t) * torch.pow(self.t, k) / factorial(k)) ## important!
            x_list[k] = x_list[k] @ (torch.pow(self.t, k) / factorial(k))
            if k != 0: 
                y += x_list[k]
            else:
                y = x_list[k]

        return y
    
    def reset_parameters(self):
        # self.t.data.fill_(2)
        torch.nn.init.normal_(self.t, mean=2, std=1)
    

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j