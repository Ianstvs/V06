"""
Name of module: ENGINE.

Description:

            - INJECTOR : Trainer for the GCN
            
            - PISTON : GCN Architecture

Author: Joshua Ian Hernandez Esteves.

Creation date: 07-03-26

Updated version: v1.0.2

Modification date: 20-04-26

Changes history: 
                - Optimizations for graphic card
                
                - Elimination of .item() replaced by .detach
"""
from torch import tensor,cat
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
 

import torch 
from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Linear
import torch.nn as nn


class Injector :
    def __init__(self,model,
                 loss_function,
                 optimizer,
                 dataloader,
                 device,
                 train = True,
                 detransformer = None) :    # NEW
        """
        Training a single epoch
        """
        self.model = model
        self.fun_loss = loss_function
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device
        self.train_status = train
        
        self.detransformer = detransformer  # NEW
        
    def train_epoch(self,use_amp=False,scaler=None) :
        if not self.train_status :
            print("⚠ train_status is False, skipping training")
            return
        batch_count = 0
        total_loss = 0.0


        for batch in self.dataloader :
            try :
                batch_count +=1
                # Sending batch to device
                batch = batch.to(self.device,non_blocking=True)     # OPT
                # Resetting gradients in tensors
                self.optimizer.zero_grad(set_to_none=True)          # OPT

                if use_amp and 'cuda' in str(self.device) :         # OPT
                    with torch.amp.autocast('cuda'):
                        original, prediction = self.predictor(batch)
                        loss = self.fun_loss(prediction, original)

                    scaler.scale(loss).backward()

                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else :                                  # NON OPTIMIZED
                    # Getting predictions
                    original , prediction = self.predictor(batch)
                    loss = self.fun_loss(prediction,original)            # Call a backward function in loss function tensor
                    loss.backward()
                    # Adding gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                   max_norm=1.0)   # max_norm = 1 is the standard value for molecular GCN's 
                    # Make a step in optimzer 
                    self.optimizer.step()
                total_loss += loss.detach()     # New gpu detach
            finally :
                if 'loss' in locals(): del loss
                if 'prediction' in locals(): del prediction
                if 'original' in locals(): del original
                if 'batch' in locals(): del batch

        
                  
            
    def predictor(self,batch) :
        # Original property
        original = batch.y.view(-1,1)
        # Predicted property
        prediction,_ = self.model(batch.x.float(), batch.edge_index, batch.batch)
        prediction = prediction.view(-1, 1)        
        return original , prediction
    
    def evaluator(self,preloaded_batch=None) :
        if self.train_status :
            return
        total_loss = 0
        original_tensor_list = []
        prediction_tensor_list = []

        batches = [preloaded_batch] if preloaded_batch is not None else self.dataloader


        for batch in batches :
            if preloaded_batch is None :
                batch.to(self.device)

            # Get predictions and the original property
            original,prediction = self.predictor(batch)
            # Add the data to the lists
            original_tensor_list.append(original)
            prediction_tensor_list.append(prediction)
            # Compute the loss of each batch
            loss = self.fun_loss(prediction,original)
            # Add up the losses of all batches 
            total_loss +=loss.detach()      # New gpu opt
        
        # Getting the average loss of all epochs
        loss = total_loss.item()/len(self.dataloader)       # New gpu opt
        # Joining the prediction tensors, send them to cpu, and convert them to lists
        if self.detransformer is not None :
            property_predictions = self.detransformer(cat(prediction_tensor_list).cpu().numpy().flatten()) # NEW
            property_orignals = self.detransformer(cat(original_tensor_list).cpu().numpy().flatten())      # NEW
        else :
            property_predictions = cat(prediction_tensor_list).cpu().numpy().flatten()
            property_orignals = cat(original_tensor_list).cpu().numpy().flatten()
        r2 = r2_score(property_orignals,property_predictions)
        mse = mean_squared_error(property_orignals,property_predictions)
        mae = mean_absolute_error(property_orignals,property_predictions)
        
        stage_stats ={
            'originals' : property_orignals ,
            'predictions' : property_predictions ,
            'loss' : loss ,
            'MAE' : mae ,
            'MSE' : mse ,
            'R2' : r2   
        }
        
        return stage_stats
       
       
                 
        
class Piston(torch.nn.Module) :
    def __init__(self,dataset,
                 hidden_dims = 64,
                 hidden_layers = 3,
                 activation_fun = 0,
                 pooling = 'both') :
        
        super(Piston,self).__init__()
        
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
