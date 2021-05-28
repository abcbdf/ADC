import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from scipy.special import factorial
from torch_geometric.nn import GATConv


class GATPlusConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=True, dropout=0, init_t = 0):
        super(GATPlusConv, self).__init__(in_channels=in_channels, out_channels=out_channels,heads=heads,concat=concat, dropout=dropout)  # "Add" aggregation (Step 5).
        self.init_t = init_t
        self.t = Parameter(torch.Tensor(1))
        self.step = 20
        self.reset_parameters()

    def reset_parameters(self):
        super(GATPlusConv, self).reset_parameters()
        if hasattr(self, "t"):
            torch.nn.init.constant_(self.t, self.init_t)
            self.t.requires_grad = False


    def propagate(self, edge_index, x, *args, **kwargs):
        # print(x)
        x_list = [0 for i in range(self.step)]
        x_list[0] = x

        for i in range(1, self.step):
            #x_list[i] = self.propagate(edge_index, x=x_list[i - 1], norm=norm)
            x_list[i] = super(GATPlusConv, self).propagate(edge_index=edge_index,x=x_list[i - 1],*args, **kwargs)
        
        y = 0

        for k in range(self.step):
            x_list[k] = torch.exp(-self.t) * torch.pow(self.t, k) / factorial(k) * x_list[k] ## important!
            # x_list[k] = torch.pow(self.t, k) / factorial(k) * x_list[k]
            if k != 0: 
                y += x_list[k]
            else:
                y = x_list[k]
        return y
        