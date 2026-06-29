import torch
import torch.nn as nn
from torch import tensor,cat
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max

class Conv(torch.nn.Module) :
    def __init__(self,dataset,
                 hidden_dims = 64,
                 hidden_layers = 3,
                 activation_fun = 0,
                 pooling = 'both') :
        
        super(Conv,self).__init__()
        
        self.activf_list = {
            0 : nn.ELU,         # Default for GCN's
            1 : nn.GELU,        # For deep networks (> 5 layers)
            2 : nn.SiLU,        # Modern alternative
            3 : nn.Mish,        # For complex properties
            4 : nn.LeakyReLU,   # Baseline  
            5 : nn.ReLU,        # General representative of the model
            6 : nn.Sigmoid,     # Only for output
            7 : nn.Tanh,        # Avoid
        }

        # Dynamic architecture?
        self.dynamic = False
        if type(hidden_dims) is list :
            self.dynamic = True
        
        # Activation function
        self.act_fun = self.activf_list[activation_fun]()
        # Hidden dimensions
        self.hid_dims = hidden_dims
        # Hidden layers
        if self.dynamic :
            self.hid_layers = len(self.hid_dims)
        else :
            self.hid_layers = hidden_layers 
            
        self.pooling = pooling
        
        # Initial layer
        if self.dynamic :
            self.initial_conv = GCNConv(dataset[0].x.shape[1], hidden_dims[0])
        else :
            self.initial_conv = GCNConv(dataset[0].x.shape[1], hidden_dims)
        # Dynamic convolutional layers
        if self.dynamic :
            self.conv_layers = torch.nn.ModuleList([
                GCNConv(hidden_dims[i], hidden_dims[i+1]) 
                for i in range(len(hidden_dims)-1)
            ])
        else :
            self.conv_layers = torch.nn.ModuleList([
                GCNConv(hidden_dims, hidden_dims) 
                for _ in range(hidden_layers)
            ])
        # =======================================
        # IMPLEMENTING BATCH-NORM & DROPOUT
        if self.dynamic :
            self.bn_layers = nn.ModuleList([
                    nn.BatchNorm1d(hidden_dims[i+1]) 
                    for i in range(len(hidden_dims)-1)
            ])
            self.dropout = nn.Dropout(p=0.2)    # This parameter is not being optimized
        else :
            self.bn_layers = nn.ModuleList([
                    nn.BatchNorm1d(hidden_dims) for _ in range(hidden_layers)
            ])
            self.dropout = nn.Dropout(p=0.2)    # This parameter is not being optimized
        # =======================================
       
        # Exit layer
        if self.dynamic :
            if pooling == 'both':
                out_dim = hidden_dims[-1] * 2
            else:
                out_dim = hidden_dims[-1]
            self.out = Linear(out_dim, 1)
        else :
            if pooling == 'both':
                out_dim = hidden_dims * 2
            else:
                out_dim = hidden_dims
            self.out = Linear(out_dim, 1)
        
    def forward(self, x, edge_index, batch_index):
        # Inital layer
        hidden = self.initial_conv(x, edge_index)
        hidden = self.act_fun(hidden)
        
        # Dynamic convolutional layers
        for i,conv_layer in enumerate(self.conv_layers) :
            hidden = conv_layer(hidden, edge_index)
            hidden = self.bn_layers[i](hidden)
            hidden = self.act_fun(hidden)
            hidden = self.dropout(hidden)
        
        # Pooling
        if self.pooling == 'both':
            hidden = torch.cat([gmp(hidden, batch_index), gap(hidden, batch_index)], dim=1)
        elif self.pooling == 'max':
            hidden = gmp(hidden, batch_index)
        elif self.pooling == 'mean':
            hidden = gap(hidden, batch_index)
        
        out = self.out(hidden)
        return out, hidden 