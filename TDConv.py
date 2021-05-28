import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from scipy.special import factorial
from args import get_citation_args



class TDConv(MessagePassing):
    def __init__(self, in_channels, init_t):
        super(TDConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        args = get_citation_args()
        self.init_t = init_t
        self.step = args.step
        if not args.denseT:
            self.t = Parameter(torch.Tensor(in_channels))
        else:
            self.t = Parameter(torch.Tensor(1))
        # self.t.data.fill_(2)
        self.reset_parameters()
        # self.t.requires_grad = False


    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # print(self.t)
        # Step 1: Add self-loops to the adjacency matrix.
        #self.t_norm = torch.nn.functional.relu(self.t)

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
            x_list[k] = torch.exp(-self.t) * (torch.pow(self.t, k) / factorial(k)) * x_list[k] ## important!
            #x_list[k] = torch.exp(-self.t_norm) * (torch.pow(self.t_norm, k) / factorial(k)) * x_list[k]
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