"""
Name of module: ITINERARY.

Description: Itinerary for training

Author: Joshua Ian Hernandez Esteves.

Creation date: 07-03-26

Updated version: v1.1.0

Modification date: 28-04-26

Changes history: 
                - Optimization for graphic card
                
                - Debugging of GPU optimization : Deactivation of torch.compile
                
                - Replacement of evaluation mode : Use of torch.inference_mode():  
                
                - Moving the evaluation objects out of the epoch loops. 
                
                - Change of evaluation dataloaders : all_dataloaders = self.Deck.evaluation_data()   
                        -The rest of the training itienrary was adapted to the evaluation dataloaders.   
                
                - Additional WANDB logs have been added.    
"""
import numpy as np
import pandas as pd
from pathlib import Path
import gc
import time
import copy

import torch 
from torch.nn import L1Loss , SmoothL1Loss , MSELoss , HuberLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts ,ReduceLROnPlateau # NEW
from .Engine import Piston , Injector


from .Plotter import AllStats   

import wandb            # NEW

class Train :
    def __init__(self,t_setup,dataloaders,
                 device,scheduler,scheduler_factor,
                 scheduler_patience,epochs_number,
                 early_stop,early_stop_handicap,
                 early_stop_patience,
                 cache_loss,
                 cache_opt,
                 detransformer = None,
                 cosineannealing = False,
                 plot_best_trials = False,
                 global_score = float('inf')
                 ) :
        """
        
        """
        self.training_setup = t_setup
        self.Deck = dataloaders
        self.device = device
        self.learning_rate_scheduler = scheduler
        self.epochs = epochs_number
        self.early_stop = early_stop
        self.stop_handicap = early_stop_handicap
        self.stop_patience = early_stop_patience
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        
        self.detransformer = detransformer  # NEW
        self.cosineannealing = cosineannealing  # NEW   
        self.plot_best_trials = plot_best_trials  # NEW
        self.global_score = global_score           # NEW
        
        # Loss CACHE list of functions
        self.loss_cache = cache_loss
       
        # Opt CACHE list of functions
        self.optimizer_cache = cache_opt
        
        self.use_amp = True     # OPT
        #self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and str(device) == 'cuda' else None    # OPT
        self.scaler = torch.amp.GradScaler('cuda') if (self.use_amp and 'cuda' in str(device)) else None
        print(f"AMP Status: {self.use_amp}")
        print(f"Scaler: {self.scaler}")
        # ==========================================================
       
    # Get loss function from cache
    def get_loss_function(self, loss_idx):
        if loss_idx not in self.loss_cache:
            loss_functions = {
                0: L1Loss(),
                1: MSELoss(),
                2: HuberLoss(),
                3: SmoothL1Loss()
            }
            self.loss_cache[loss_idx] = loss_functions[loss_idx]
        return self.loss_cache[loss_idx]
    
    # Get optimization function from cache
    def get_opt_function(self, opt_idx):
        if opt_idx not in self.optimizer_cache:
            opt_functions = {
                0 : torch.optim.Adadelta,   # In practice, it perfoms poorly.
                1 : torch.optim.Adagrad,    # INADECUATE for GCN's : Accumulates squared gradients. Effective lr colapses
                                            # close to 0 around 100-200 epochs, faking convergence when the model has already stopped.
                2 : torch.optim.Adam,
                3 : torch.optim.AdamW,
                4 : torch.optim.Adamax,
                5 : torch.optim.NAdam,
                6 : torch.optim.RAdam,
                7 : torch.optim.RMSprop,
                8 : torch.optim.Rprop       # INADECUATE for GCN's: Dessigned for complete batch gradient descent. Using
                                            # mini-batches, gradient sign is noisy and Rprop loses it's convergence warranties,
                                            # causing an erratic behavior.
            }
            self.optimizer_cache[opt_idx] = opt_functions[opt_idx]
        return self.optimizer_cache[opt_idx]
        
    def cleanup_memory(self) :
        gc.collect()
        if torch.cuda.is_available() :
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    # ================================================================================================
    ### T  R  A  I  N    R  O  U  T  I  N  E
    # ================================================================================================
   
    def train_itinerary(self) :
        self.scheduler = None
        torch.manual_seed(self.training_setup['Random seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.training_setup['Random seed'])
        print(f"{'='*60}\n")   
        print(f"Device: {self.device}")
        print(f"Use AMP: {self.use_amp}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*60}\n")

        print(f"Number of neurons fed : {self.training_setup['Number of neurons']}")

        # Initialize the model
        model = Piston(self.Deck.train_geometric,
                       hidden_dims = self.training_setup['Number of neurons'],
                       hidden_layers = self.training_setup['Number of layers'],
                       activation_fun = self.training_setup['Activation function'],
                       pooling = 'both')   # Here it requires the Geometric train dataset
        
        print(f"✓ Model created")
        
        # Sends the model to the device available CPU or GPU
        model = model.to(self.device)
        
        if torch.cuda.is_available():
            param_device = next(model.parameters()).device
            print(f"✓ Model parameters are on: {param_device}")
            # Compara que la base del dispositivo coincida (ej. 'cuda' con 'cuda:0')
            if param_device.type != torch.device(self.device).type:
                print(f"⚠ WARNING: Device mismatch! Expected {self.device}, got {param_device}")


        
        # Create instances for training functions
        torch_loss = self.get_loss_function(self.training_setup['Loss function'])
        torch_optimizer = self.get_opt_function(self.training_setup['Optimization function'])
        
        optimizer = torch_optimizer(model.parameters(),lr = self.training_setup['Learning rate'])
        
        print(f"✓ Optimizer created")

        # Define scheduler for change learning rate
        if (self.learning_rate_scheduler):
            if not self.cosineannealing :
                self.scheduler = ReduceLROnPlateau(optimizer,
                                                   factor=self.scheduler_factor,
                                                   patience=self.scheduler_patience,
                                                   min_lr = 1e-5,
                                                   cooldown = 10,
                                                   verbose=False
                                                   )
            else :
                self.scheduler = CosineAnnealingWarmRestarts(optimizer,   # NEW
                                                             T_0=10,      # Reinicia cada 10 epochs
                                                             T_mult=2,    # Duplica el período después de cada reinicio
                                                             eta_min=1e-5 # LR mínimo
                                                            )
        
        # Declare metrics for current train
        self.evaluation_metrics = {
            'train' : {
                'loss' : [] ,
                'MAE' : [] ,
                'MSE' : [] ,
                'R2' : [] ,
            } ,
            'test' : {
                'loss' : [] ,
                'MAE' : [] ,
                'MSE' : [] ,
                'R2' : [] , 
            } ,
            'validation' : {
                'loss' : [] ,
                'MAE' : [] ,
                'MSE' : [] ,
                'R2' : [] ,
            }
        }
        self.evaluation_properties = {
            'train' : {} ,
            'test' : {} ,
            'validation' : {} 
        }
        # Creating DataLoaders
        all_dataloaders = self.Deck.evaluation_data   # FIX : These dataloaders have a single batch, so they don't need to be iterated.
        
        eval_train_dataloader = all_dataloaders['train']
        eval_test_dataloader = all_dataloaders['test']
        eval_validation_dataloader = all_dataloaders['validation']
        
        train_dataloader = self.Deck.train_data_loaders(self.training_setup['Batch size'])['train']
        try :        
            # Create a training class 
            trainer = Injector(model,torch_loss,optimizer,
                               train_dataloader,
                               self.device,
                               train=True)

            # Creating a best score variable
            self.best_score = float('inf')
            self.counter = 0
            self.score = float('inf')

            print(f"\n{'='*60}")
            print(f"STARTING TRAINING - {self.epochs} epochs")
            print(f"{'='*60}\n")
            #####
            #_____________________________________________________________________
            # Create evaluation objects
            eval_train = Injector(model,torch_loss,optimizer,
                                  eval_train_dataloader,
                                  self.device,train=False,
                                  detransformer=self.detransformer)     # NEW
            eval_test = Injector(model,torch_loss,optimizer,
                                 eval_test_dataloader,
                                 self.device,train=False,
                                 detransformer=self.detransformer)       # NEW
            eval_valid = Injector(model,torch_loss,optimizer,
                                  eval_validation_dataloader,
                                  self.device,train=False,
                                  detransformer=self.detransformer)      # NEW
            #_____________________________________________________________________
            #####

            # Iterate for each epoch
            for epoch in range(0,self.epochs):
                # Enable train mode
                ###############
                # TRAIN MODEL #
                ###############
                model.train()
                if 'cuda' in str(self.device) :
                    trainer.train_epoch(use_amp=self.use_amp, scaler=self.scaler)   # OPT
                else :
                    # Make a train with train dataloader
                    trainer.train_epoch()
                # Start evaluation
                ####################
                # MODEL EVALUATION #
                ####################
                model.eval()
                with torch.inference_mode():     #torch.no_grad():
                    eval_batch_train = next(iter(eval_train_dataloader)).to(self.device)        # NEW   NEW
                    eval_batch_test  = next(iter(eval_test_dataloader)).to(self.device)         # NEW   NEW
                    eval_batch_valid = next(iter(eval_validation_dataloader)).to(self.device)   # NEW   NEW
                    
                    # Creation of dictionaries containing all epoch metrics
                        # The dict contains: original and predicted properties, Loss, MSE, MAE, r2
                    train_eval = eval_train.evaluator(preloaded_batch=eval_batch_train)     # NEW   NEW
                    test_eval = eval_test.evaluator(preloaded_batch=eval_batch_test)        # NEW   NEW
                    valid_eval = eval_valid.evaluator(preloaded_batch=eval_batch_valid)     # NEW   NEW
                    
                    self.epoch_evaluation_saver(train_eval,test_eval,valid_eval)

                    if wandb.run is not None:           # NEW
                        wandb.log({
                            "epoch":      epoch,
                            "train_loss": train_eval['loss'],
                            "train_mse" : train_eval['MSE'] ,
                            'train_r2' : train_eval['R2'] ,
                            "val_loss":   valid_eval['loss'],
                            "val_mse" : valid_eval['MSE'] ,
                            'val_r2' : valid_eval['R2']
                        })

                    ### Updating the learning rate according to the loss. If lr scheduler is active.
                    if(self.learning_rate_scheduler):
                        if self.cosineannealing :
                            self.scheduler.step()  # NEW    
                        else :
                            self.scheduler.step(valid_eval['loss'])

                    # Extracting a representative score for the epoch-train
                    score = (self.evaluation_metrics['test']['MSE'][-1]+ self.evaluation_metrics['validation']['MSE'][-1]) / 2  # Mean of the MSE

                    # ______________________________________________________________
                    # EARLY STOP
                    stop = False
                    if self.early_stop :
                        stop = self.early_stopper(epoch=epoch,
                                                  model=model,
                                                  score=score,
                                                  train_data=train_eval,
                                                  test_data=test_eval,
                                                  val_data=valid_eval)
                    if stop :
                        break
                    
                    # Extracting all metrics for final epoch
                    if epoch == self.epochs - 1 :
                        self.training_prediction_properties(train_eval,test_eval,valid_eval)
                        self.score = ((np.array(self.evaluation_metrics['test']['MSE']) + np.array(self.evaluation_metrics['validation']['MSE']))/2).min().item()  # NEW
                        self.best_score = self.score
                        self.training_setup['Epoch'] = epoch

                #if epoch % 10 == 0:  # Cada 10 epochs
                #    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            self.cleanup_memory()        
            model.cpu()

            self.paperwork_secretary(model)
        finally :
            ## C L E A N I N G  
            for dl in [train_dataloader, test_dataloader, validation_dataloader]:
                if hasattr(dl, '_iterator') and dl._iterator is not None:
                    del dl._iterator

            # Variables that mey not exist if there was an error before assigning them 
            for var_name in ['train_dataloader', 'test_dataloader', 'validation_dataloader',
                             'all_dataloaders', 'model', 'optimizer', 'trainer']:
                if var_name in locals():
                    del locals()[var_name]  # No NameError
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                del self.scheduler
            if hasattr(self, 'best_state'):
                del self.best_state
            if hasattr(self, 'best_data_train'):
                del self.best_data_train, self.best_data_test, self.best_data_validation

            self.cleanup_memory()
        # __________________________________________________________________________
    
        
    
    def epoch_evaluation_saver(self,train_metrics,test_metrics,validation_metrics) :
        """
        Saves the metrics of each epoch into the self.evaluation_metrics variable.
        ____________________________
        The data given must be dicts arranging the metrics in the following order:
            (train metrics , test metrics , evaluation metrics)
        -------------
        This function does not return anything.
        """
        # Getting all the metrics of the different subsets in a list to iterate'em.
        metrics =[train_metrics,test_metrics,validation_metrics]
        # Appending the metrics of each epoch
        for index,subset in enumerate(self.evaluation_metrics.keys()) : # Iterating the subsets list.
            for metric in self.evaluation_metrics[subset].keys() :  # Iterating the metrics list.
                self.evaluation_metrics[subset][metric].append(metrics[index][metric])
                
    # ================================================================================================================================
    # E A R L Y    S T  O P
    # ================================================================================================================================
    def early_stopper(self,epoch,model,score,train_data,test_data,val_data) -> bool:
        if self.best_score > score:
            self.score = score
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())#{k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.best_data_train = train_data
            self.best_data_test = test_data
            self.best_data_validation = val_data
            self.training_setup['Epoch'] = epoch
            self.counter = 0
            
            
            if epoch == self.epochs - 1:
                self.training_prediction_properties(self.best_data_train,
                                                    self.best_data_test,
                                                    self.best_data_validation)
                model.load_state_dict({k: v.to(self.device) for k, v in self.best_state.items()})

                return True
                
            return False
            
        else:            
            if epoch < self.stop_handicap:
                return False
                
            self.counter += 1
            
            if self.counter >= self.stop_patience:
                print('! ! ! E A R L Y   S T O P   H A S   B E E N   E X E C U T E D ! ! !\n\n')
                print(f'The training stopped on epoch {epoch}')
                self.training_prediction_properties(self.best_data_train,
                                                    self.best_data_test,
                                                    self.best_data_validation)
                model.load_state_dict(self.best_state)
                
                return True
        
            return False
    
    def training_prediction_properties(self,train_data,test_data,validation_data) :
        """
        Saves the predicted  and true properties in the self.evaluation_properties dictionary.
        ____________________________
        The data given must be dicts arranging them in the following order:
            (train data,test data,evaluation data)
        -------------
        This function does not return anything.
        """
        data = [train_data,test_data,validation_data]
        for index,subset in enumerate(self.evaluation_properties.keys()) :
            self.evaluation_properties[subset]['originals'] = data[index]['originals']
            self.evaluation_properties[subset]['predictions'] = data[index]['predictions']
            
    def paperwork_secretary(self,model) :
        try :
            AllStats(self.Deck.smiles_dealed,
                     self.evaluation_metrics,
                     self.evaluation_properties,
                     self.training_setup,model,
                     self.plot_best_trials,
                     self.global_score)
        except Exception as e :
            print('ERROR !!! SAVING THE RESULTS OF THE TRAINING :\n\n'
                  f'{self.training_setup}\n\n')
            ts = int(time.time())
            print(ts) 
            root = Path(__file__).resolve().parent.parent / 'trainings' / 'EMERGENCY'
            print(str(root))
            root.mkdir(parents=True, exist_ok=True)
            emergency_csv_path = root / f'{ts}.csv'
            emergency_pth_path = root / f'{ts}.pth'
            pd.DataFrame([{'error': str(e), 'setup': str(self.training_setup)}]).to_csv(emergency_csv_path, index=False)
            torch.save(model.state_dict(),emergency_pth_path)
            
            print('RESULTS HAVE BEEN SAVED ON AN EMERGENCY FOLDER\n'
                  'LOOK FOR THE FOLLOWING KEY:\n\n'
                  f'{ts}')            
