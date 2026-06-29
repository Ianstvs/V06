"""
Name of module: MANAGER.

Description: Experiment routine. 

Author: Joshua Ian Hernandez Esteves.
 
Creation date: 07-02-26

Updated varsion: v1.3.3

Modification date: 23-04-26

Changes history: 
                - WandB : Debugging
                                
                - optuna-dashboard : Debugging and clearer db log 
                
                - Graphic card optimization
                
                - New bayssian optimization value "optimize_width"
                
                - Debugging GPU opt :  torch.compile method was disabled due to incompatibility with graph sizes
                
                - Reorder of graph creation along with saving them on a cache folder
                
                - Avoiding recreation of Deck
                
                - Batch size optimization bypass
                
                - Layer 1 # Neurons optimization implementation
                
"""
# Script for graph and dataloader generation
from .DataHandler import DataExtractor,DataDealer
# Script for the training routine
from .Itinerary import Train
# Normalizer script
from .Normalizer import Norm
# Torch FRAMEWORK
import torch
from torch.nn import L1Loss , SmoothL1Loss , MSELoss , HuberLoss
# Bayessian optimizer
import optuna
# Trash disposal
import gc

# K_FOLD data splitting
from sklearn.model_selection import KFold

import wandb                # NEW
import optuna_dashboard     # NEW
import time

import hashlib              # NEW
from pathlib import Path



class V05 :
    def __init__(self,setup_data) :
        
        ### =======================================================================
        ###  I N P U T   D A T A 
        # _________________________________________________________________________
        ### Training data
        self.dataset_treatment = setup_data['dataset_treatment'] # If required, True    # NEW
        if not isinstance(self.dataset_treatment, bool):
            raise ValueError('"dataset_treatment" requires to be a boolean')
        self.name_of_id = setup_data['column_name_of_id']
        if not isinstance(self.name_of_id, str):                                        # NEW
            raise ValueError('"column_name_of_id" requires to be a string')
        self.name_of_property = setup_data['column_name_of_property']
        if not isinstance(self.name_of_property ,str):
            raise ValueError('"column_name_of_property" requires to be a string')
        self.name_of_smiles = setup_data['column_name_of_smiles']
        if not isinstance(self.name_of_smiles,str) :
            raise ValueError('"column_name_of_smiles" requires to be a string')
        self.node_features = setup_data['node_features']
        if not isinstance(self.node_features, list) :
            raise ValueError('"custom_graph_features" requires to be a list of integers')
        self.dataset_path = setup_data['dataset_path']
        if not isinstance(self.dataset_path, str):
            raise ValueError('"dataset_path" requires to be a string')
        self.test_size = setup_data['test_dataset_ratio']
        if not isinstance(self.test_size, float):
            raise ValueError('"test_dataset_ratio" requires to be a float')
        self.val_size = setup_data['validation_dataset_ratio']
        if not isinstance(self.val_size, float):
            raise ValueError('"validation_dataset_ration" requires to be a float')
        self.epochs_number = setup_data['number_of_epochs']
        if not isinstance(self.epochs_number, int):
            raise ValueError('"epochs_number" requires to be an integer')
        
        self.plot_best_trials = setup_data['plot_best_trials']                          # NEW
        if not isinstance(self.plot_best_trials, bool):
            raise ValueError('"plot_best_trials" requires to be a boolean')

        # _________________________________________________________________________
        ### Regular grid search input data
        self.early_stopping = setup_data['early_stopping'] # If required, True
        if not isinstance(self.early_stopping, bool):
            raise ValueError('"early_stopping" requires to be a boolean')
        self.early_stop_handicap = setup_data['early_stop_handicap'] # Number of epochs before starting the early stop
        if not isinstance(self.early_stop_handicap, int):
            raise ValueError('"early_stop_handicap" requires to be a integer')
        self.early_stop_patience = setup_data['early_stop_patience'] # Patience of the early stopping 
        if not isinstance(self.early_stop_patience, int):
            raise ValueError('"early_stop_patience" requires to be an integer')
        #The following data shall be used for the training
        self.random_states = setup_data['random_states_list']  
        if not isinstance(self.random_states, list):
            raise ValueError('"random_states" requires to be a list')
        self.random_seeds = setup_data['random_seeds_list']
        if not isinstance(self.random_seeds, list):
            raise ValueError('"random_seeds" requires to be a list')
         # In case it's not bypassed, only one random STATE and SEED should be introduced,
         #otherwise the first data in the list will be used.
        self.optimizers = setup_data['optimizers_list']
        if not isinstance(self.optimizers, list):
            raise ValueError('"optimizers_list" requires to be a list')
        self.loss_fun = setup_data['loss_functions_list']
        if not isinstance(self.loss_fun, list):
            raise ValueError('"loss_funcions_list" requires to be a list')
        self.learning_rate = setup_data['learning_rates_list']
        if not isinstance(self.learning_rate, list):
            raise ValueError('"learning_rates_list" requires to be a list')
        self.batch_size = setup_data['batch_sizes_list']
        if not isinstance(self.batch_size, list):
            raise ValueError('"batch_sizes_list" requires to be a list')
        self.activation_fun =setup_data['activation_functions_list']
        if not isinstance(self.activation_fun, list):
            raise ValueError('"activation_functions_list" requires to be a list')
        self.layers = setup_data['number_of_layers_list']
        if not isinstance(self.layers, list):
            raise ValueError('"number_of_layers_list" requires to be a list')
        self.neurons = setup_data['number_of_neurons_list']
        if not isinstance(self.neurons, list):
            raise ValueError('"number_of_neurons_list" requires to be a list')
        # _________________________________________________________________________
        ### learning rate scheduler
        self.learning_rate_scheduler = setup_data['learning_rate_scheduler']
        if not isinstance(self.learning_rate_scheduler, bool):
            raise ValueError('"learning_rate_scheduler" requires to be a boolean')
        self.cosine_annealing_scheduler = setup_data['cosine_annealing_scheduler']  # NEW
        if not isinstance(self.cosine_annealing_scheduler, bool):
            raise ValueError('"cosine_annealing_scheduler" requires to be a boolean')
        self.scheduler_factor = setup_data['scheduler_factor']
        if not isinstance(self.scheduler_factor, float):
            raise ValueError('"scheduler_factor" requires to be an integer')
        self.scheduler_patience = setup_data['scheduler_patience']
        if not isinstance(self.scheduler_patience, int):
            raise ValueError('"scheduler_patience" requires to be an integer')
          
        #________________________________________________________________________
        # ARCHITECTURE OPTIMIZER INPUT DATA
        self.architecture_optimizer = setup_data['architecture_optimizer']
        if not isinstance(self.architecture_optimizer, bool):
            raise ValueError('"architecture_optimizer" requires to be a boolean')
        self.optimize_width = setup_data['optimize_width']
        if not isinstance(self.optimize_width, bool):
            raise ValueError('"optimize_width" has to be a bool')
        if self.optimize_width :
            self.width_space = setup_data['width_optimization_space'] 
            if not isinstance(self.width_space,list) :
                raise ValueError('"width_optimization_space" requires to be a list of 2 integers arranged from the smallest to the biggest')               
        if self.architecture_optimizer :    
            self.arch_optimizer_trials = setup_data['architecture_optimizer_trials']
            if not isinstance(self.arch_optimizer_trials, int):
                raise ValueError('"architecture_optimizer_trials" requires to be an integer')
            self.arch_optimizer_timeout = setup_data['architecture_optimizer_timeout']
            if not isinstance(self.arch_optimizer_timeout, int) and self.arch_optimizer_timeout is not None:
                raise ValueError('"architecture_optimizer_timeout" requires to be an integer or None, if you wanna skip the timeout')
            self.arch_optimizer_patience = setup_data['architecture_optimizer_patience']
            if not isinstance(self.arch_optimizer_patience, int) and self.arch_optimizer_patience is not None:
                raise ValueError('"architecture_optimizer_patience" requires to be an integer or None, if you wanna skip the patience')
        
        #________________________________________________________________________
        # HYPERPARAMETERS OPTIMIZER INPUT DATA
        self.hyperparameters_optimizer = setup_data['hyperparameter_optimizer']  # If hyperparameters optimization is TRUE
        if not isinstance(self.hyperparameters_optimizer, bool):
            raise ValueError('"hyperparameters_optimization" requires to be a boolean')
        if self.hyperparameters_optimizer :
            self.hyper_optimizer_trials = setup_data['hyperparameter_optimizer_trials']
            if not isinstance(self.hyper_optimizer_trials, int):
                raise ValueError('"hyperparameter_optimizer_trials" requires to be an integer')
            self.hyper_optimizer_timeout = setup_data['hyperparameter_optimizer_timeout']
            if not isinstance(self.hyper_optimizer_timeout, int) and self.hyper_optimizer_timeout is not None:
                raise ValueError('"hyperparameter_optimizer_timeout" requires to be an integer or None, if you wanna skip the timeout')
            self.hyper_optimizer_patience = setup_data['hyperparameter_optimizer_patience']
            if not isinstance(self.hyper_optimizer_patience, int) and self.hyper_optimizer_patience is not None:
                raise ValueError('"hyperparameter_optimizer_patience" requires to be an integer or None, if you wanna skip the patience')
            self.force2set_batch_size = setup_data['force_batch_size_value']
            if not self.force2set_batch_size is None and not isinstance(self.force2set_batch_size,int) :
                raise ValueError('"force_batch_size_value" requires to be an integer or a None to ignore.') 
            # _________________________________________________________________________
            ### Bayessian optimization search ranges
            # (if bypass is false)
            self.batch_size_range = setup_data['batch_size_range']
            if not isinstance(self.batch_size_range, list):
                raise ValueError('"batch_size_range" requires to be a list of floats')
            elif (len(self.batch_size_range) != 2) or (self.batch_size_range[0]>=self.batch_size_range[-1]) :
                raise ValueError('"batch_size_range" list lenght requires to be 2\n'
                                 'and need to be arranged from smaller to bigger')
            self.lr_range = setup_data['learning_rate_range']
            if not isinstance(self.lr_range, list):
                raise ValueError('"learning_rate_range" requires to be a list of floats')
            elif (len(self.lr_range) != 2) or (self.lr_range[0]>=self.lr_range[-1]) :
                raise ValueError('"learning_rate_range" list lenght requires to be 2\n'
                                 'and need to be arranged from smaller to bigger')
        # _________________________________________________________________________
        # PLOTTING RESTRICTORS AND VALIDATION
        self.min_best_score = setup_data['minimum_best_score']      # NEW2
        if not isinstance(self.min_best_score,float):
            if self.min_best_score is not None :
                raise ValueError('"minimum_best_score" requires to be a float value if a pre-set global score is required,\n'
                                 'else fill it with a None value.')
        self.x_validation = setup_data['cross_validation']          # NEW
        if not isinstance(self.x_validation,bool):
            raise ValueError('"cross_validation" requires to be a bool')
        else :
            self.k_folds = setup_data['k_folds']                    # NEW
            if not isinstance(self.k_folds,int) :
                raise ValueError('"k_folds" requires to be an integer')
        # _________________________________________________________________________
        # USE OF WANDB AND OPTUNA DASHBOARD
        self.use_optuna_db = setup_data['use_optuna_db']
        if not isinstance(self.use_optuna_db,bool) :
            raise ValueError('"use_optuna_db" requires to be a bool')
        
        self.use_wandb = setup_data['use_wandb']
        if not isinstance(self.use_wandb,bool) :
            raise ValueError('"use_wandb" requires to be a bool')
        else :
            if self.use_wandb or self.use_optuna_db :
                self.project_name = setup_data['project_name']
                if not isinstance(self.project_name,str) :
                    raise ValueError('"wandb_project" requires to be a str')
            else :
                self.project_name = None
        
        
        self.feature_optimization = setup_data['feature_optimization']
        if not isinstance(self.feature_optimization, bool):
            raise ValueError('"feature_optimization" requires to be a bool')
        if self.feature_optimization :    
            self.feat_optimizer_trials = setup_data['feature_optimizer_trials']
            if not isinstance(self.feat_optimizer_trials, int):
                raise ValueError('"feature_optimizer_trials" requires to be an integer')
            self.feat_optimizer_timeout = setup_data['feature_optimizer_timeout']
            if not isinstance(self.feat_optimizer_timeout, int) and self.arch_optimizer_timeout is not None:
                raise ValueError('"feature_optimizer_timeout" requires to be an integer or None, if you wanna skip the timeout')
            self.feat_optimizer_patience = setup_data['feature_optimizer_patience']
            if not isinstance(self.arch_optimizer_patience, int) and self.arch_optimizer_patience is not None:
                raise ValueError('"feature_optimizer_patience" requires to be an integer or None, if you wanna skip the patience')
        
        self.gcn_model = setup_data['model']            # NEW
        if not isinstance(self.gcn_model , str) :
            raise ValueError('"model" requires to be a str')
        
        ### ======================================================================
        # D I C T I O N A R I E S
        self.features_dict = {
            0 : 'AtomNum',
            1 : 'Degree',
            2 : 'FormalCh',
            3 : 'Hybridization',
            4 : 'Aromatic',
            5 : 'NumHs',
            6 : 'InRing',
            7 : 'GastCh'
        }

        self.optf_dict = {
            0 : 'Adadelta',
            1 : 'Adagrad',
            2 : 'Adam',
            3 : 'AdamW',
            4 : 'Adamax',
            5 : 'NAdam',
            6 : 'RAdam',
            7 : 'RMSprop',
            8 : 'Rprop'
        }
        self.inv_optf_dict = {
            
            'Adadelta' : 0 ,
            'Adagrad' : 1 ,
            'Adam' : 2 ,
            'AdamW' : 3 ,
            'Adamax' : 4 ,
            'NAdam' : 5 ,
            'RAdam' : 6 ,
            'RMSprop' :7  ,
            'Rprop' : 8 
        }
        
        self.lossf_dict = {
            0 : 'L1Loss',
            1 : 'MSELoss',
            2 : 'HuberLoss',
            3 : 'SmoothL1Loss'    
        }
        self.inv_lossf_dict ={
            'L1Loss' : 0 ,
            'MSELoss' : 1 ,
            'HuberLoss' : 2 ,
            'SmoothL1Loss' : 3 
        }
        
        self.activf_dict = {
            0 : "ELU",         # Default for GCN's
            1 : "GELU",        # For deep networks (> 5 layers)
            2 : "SiLU",        # Modern alternative
            3 : "Mish",        # For complex properties
            4 : "LeakyReLU",   # Baseline  
            5 : "ReLU",        # General representative of the model
            6 : "Sigmoid",     # Only for output
            7 : "Tanh",        # Avoid
        }
        ### ======================================================================
        self.norm_data = None               # NEW
        self.detransformfun = None 
        
        # Setting up a global reference score
        if self.min_best_score is None :        # NEW
            # if there's no best score set
            self.global_score = float('inf')    # NEW
        else :
            # if a best score's been set
            self.global_score = self.min_best_score # NEW
        
        
        if self.feature_optimization :
            self.feat_score = float('inf')
        ### Dataset cleaning and dataset selection.                 # NEW    
        if self.dataset_treatment :
            self.normalizer = Norm(self.dataset_path,
                                     self.name_of_id,
                                     self.name_of_smiles,
                                     self.name_of_property)
            
            self.norm_dataset = self.normalizer.norm_dataset
            if self.normalizer.selected_transformation == 'BOXCOX' or self.normalizer.selected_transformation == 'YEOJ' :
                self.detransformfun = self.normalizer.detransform
        
        
        ### Creation of an empty dict for setup configuration
        self.training_setup = {}
        self.training_setup['Dataset'] = self.dataset_path.split("/")[-1].split(".")[0]
        self.training_setup['Property'] = self.name_of_property
        self.training_setup['Node features'] = self.node_features
        
        ### Loading graph creation data ON-DEMAND
        self.graph_data = {
            'dataset_path': self.dataset_path,
            'name_of_smiles': self.name_of_smiles,
            'name_of_property': self.name_of_property,
            'node_features': self.node_features
        }    
        
        ### Getting the device
        self.device = self.get_device()
        
        # Loss CACHE list of functions
        self.loss_cache = {}
       
        # Opt CACHE list of functions
        self.optimizer_cache = {}

        # =====================================================
    # =========================================================
    
    # Emptying cache
    def cleanup_memory(self,aggressive = False) :
        if aggressive:
            self.loss_cache.clear()
            self.optimizer_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()      
        
    def get_device(self) :
        """
        Check if acelerator is avalible, examples CUDA, MPS, MTIA, or XPU.

        Returns
        ----------
        device : `String`
            Name with current device.
        """
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        return device   
    
    def load_data_lazy(self, bypass=True):

        # Generating a unique key for cache (dataset + features)
        cache_key = hashlib.md5(
            f"{self.graph_data['dataset_path']}_{self.training_setup['Node features']}".encode()
        ).hexdigest()[:12]
        cache_path = Path("trainings/cache") / f"graphs_{cache_key}.pt"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if not hasattr(self, 'graphs_list') or self.graphs_list is None or not bypass:
            if cache_path.exists():
                print(f"✓ Loading graphs from cache: {cache_path}")
                self.graphs_list = torch.load(cache_path, weights_only=False)
            else:
                print(f"⚙ Generating graphs ...")
                self.graphs_list = DataExtractor(
                    self.graph_data['dataset_path'],
                    self.graph_data['name_of_smiles'],
                    self.graph_data['name_of_property'],
                    self.graph_data['node_features'],
                    self.norm_dataset).data_ready
                torch.save(self.graphs_list, cache_path)
                print(f"✓ Graphs save on cache: {cache_path}")
        return self.graphs_list

    def objective_features(self,trial) :
        """
        Objective function for feature optimization using Bayesian optimization.
        Suggests feature configuration and runs hyperparameter iteration.
        """
        if any(isinstance(obj,list) for obj in self.node_features) :
            raise ValueError("For the Feature Optimization mode,A single list of features is required to avoid trivial iterations")
        try :
            # To register the feature names correctly
            num_total_features = len(self.features_dict) 
            selected_indices = []

            for idx in range(num_total_features):
                if idx in self.node_features:
                    # Use of the feature name for the dashboard
                    feat_name = self.features_dict[idx]
                    # suggest_int es preferible para 0/1 en samplers numéricos
                    is_active = int(trial.suggest_categorical(f"{feat_name}", [True,False]))

                    if is_active == 1:
                        selected_indices.append(idx)
                else:
                    # Si no está en node_features, no se toca (queda apagada)
                    pass

            #  VALIDATE there's at leat a single active feature
            # If all features are deactivated 
            if not selected_indices:
                # Force a punishment for the sistem
                return float('inf') 

            self.training_setup['Node features'] = selected_indices
            
                # Reset feat_score for this trial
            self.feat_score = float('inf')
            
            graphs_list = self.load_data_lazy(bypass=False)
            
            self.hyperparameter_iteration_start()
            
            # The feat_score is updated internally through compare_feat_optimizers
            # which is called from within hyperparameter_iteration_start
            self.cleanup_memory()
            return self.feat_score
        finally :
            self.cleanup_memory(aggressive=False)
            
        
    def optimize_features(self):
        """
        Bayessian optimization for the node features
        """
        if self.use_optuna_db :
            storage = optuna.storages.RDBStorage(           # NEW
                url=f"sqlite:///trainings/optuna_study_{self.project_name}__features.db"  # Saves results
            )
            study = optuna.create_study(
                                        study_name=f"{self.gcn_model}",
                                        storage=storage,
                                        direction='minimize',
                                        load_if_exists=True,   # retakes optimization tests
                                        sampler=optuna.samplers.TPESampler(
                                            multivariate=True,
                                            group=True,
                                            seed=25)
                                        )
        else :
            study = optuna.create_study(direction='minimize',
                                        sampler=optuna.samplers.TPESampler(
                                        multivariate=True,                 #For no correlation between parameters
                                        group=True,
                                        seed=25))
        
        # Construction of the early stopping callback , patience , and/or timeout
        optimize_kwargs = {'n_trials' : self.feat_optimizer_trials,            # Number of iterations
                           'show_progress_bar' : True,
                           'gc_after_trial' : True}
        
        if self.arch_optimizer_timeout is not None :
            optimize_kwargs['timeout'] = self.feat_optimizer_timeout
           
        if self.arch_optimizer_patience is not None :
            optimize_kwargs['callbacks'] = [lambda study,trial: self.early_stop_callback(study ,trial, self.feat_optimizer_patience)]
        
        try:
            study.optimize(self.objective_features,**optimize_kwargs) 
        finally :
            del study
            self.cleanup_memory(aggressive=True)
        

    
    # ================================================================================================
    #   H  Y   P  E  R  P  A  R  A  M  E  T  E  R    C  O  N  F  I G  U  R  A  T  I  O  N
    # ================================================================================================
    def hyperparameter_iteration_start(self) :
        """
        This function creates different architectures based on the methodology selected in the set-up.
        As of now, 2026-03, the following methodologies are implemented:
            - Grid of architectures: This method creates a grid of, equally shaped, rectangular hidden layer architectures with the combinations of the following parameters:  
                - Number of layers
                - Number of neurons
                
                Therefore, the shape of the hidden layers will remain the same for all the generated architectures, rectangular shaped. 
                The only difference between them will be the width and depth.
                
            - Bayessian polynomial optimization: This method optimizes the architecture by using bayessian optimization on 2 parameters of a polynomial function that defines the number of neurons per layer. The parameters are:
                - i_index: This is the index of the exponential function that defines the funneling of the architecture to the last hidden layer.
                - min_neurons: This is the minimum number of neurons that the architecture can have in its last layer. 
                
        """
        # Getting graphs list
        graphs_list = self.load_data_lazy()
        
        # Creating data deck or using an existent
        if not hasattr(self, 'Deck') or self.Deck is None or \
        getattr(self, '_deck_random_state', None) != self.training_setup['Random state']:
        
            self.Deck = DataDealer(graphs_list,
                                   self.val_size,
                                   self.test_size,
                                   self.training_setup['Random state'],
                                   self.dataset_path,
                                   self.x_validation,
                                   self.k_folds if self.x_validation else None)
            self._deck_random_state = self.training_setup['Random state']
        
        try :
            for rand_seed in self.random_seeds : #List of int
                self.training_setup['Random seed'] = rand_seed
                for activ in self.activation_fun : # List of str
                    self.training_setup['Activation function'] = activ
                    for layers in self.layers : # List of int
                        self.arch_score = float('inf')
                        self.training_setup['Number of layers'] = layers
                        for neurons in self.neurons : # List of int
                            self.seen_architectures = set() # This set is used to store the architectures that have already been trained during the optimization process, to avoid training the same architecture more than once, since the optimization process can generate duplicate architectures.
                            try :
                                if self.architecture_optimizer and layers > 1 :
                                    self.neurons_for_opt= neurons   # Dummy value, it will be updated in the optimization process, but it needs to be defined to avoid errors in the polynomial function.  
                                    self.optimize_architecture()
                                else :
                                    self.training_setup['Number of neurons'] = neurons
                                    if layers == 1 :
                                        print('Only one layer has been specified, skipping architecture optimization\n\n'
                                              'The rest of the layers greater than 1 will be optimized.')
                                    if self.hyperparameters_optimizer :
                                        self.optimize_hyperparameters()
                                    else :
                                        self.hyperparameters_grid()
                            finally :
                                self.cleanup_memory(aggressive=False)
                                
                        self.cleanup_memory(aggressive=True)
        finally :
            if hasattr(self, 'Deck'):
                del self.Deck
            
            # OPTIMIZACIÓN 11: Liberar graphs_list si es muy grande
            if hasattr(self, 'graphs_list'):
                del self.graphs_list
                self.graphs_list = None
            
            # Limpieza agresiva final
            self.cleanup_memory(aggressive=True)
    
    #================================================================================================        
    #__ARCHITECTURE HYPERPARAMETERS CONFIGURATION FUNCTIONS__________________________________________

    def objective_architecture(self,trial) :
        """
        
        """
        self.arch_score = float('inf') # This variable is used to store the best score during the optimization process, and compare it with the current score to decide if we update the best score or not.
        
        try :
            max_neurons = None
            
            if self.training_setup['Number of layers'] > 2 :
                if self.optimize_width :
                    max_neurons = trial.suggest_int('max_neurons',self.width_space[0],self.width_space[-1],log=True)
                
                i_index = trial.suggest_float('i_index',0.1,10,log=True)       # It's needed a further analysis on the parameter's sensitivity range, but for now it will be between 0.1 and 10.
                min_neurons = trial.suggest_int('min_neurons',4,self.neurons_for_opt,log=True)    # I'm not sure about the minimum number of neurons, but I'll leave it in 4 for now. The maximum will be the maximum number of neurons introduced in the set-up.
            else :
                i_index = 1.0       # Dummy value, it won't affect the architecture since it's only one layer, but it needs to be defined to avoid errors in the polynomial function.
                min_neurons = trial.suggest_int('min_neurons',4,self.neurons_for_opt,log=True)
                
                if self.optimize_width :
                    max_neurons = trial.suggest_int('max_neurons',32,256,log=True)
            # Creating the architecture with a polynomial function.
            neurons_per_layer = self.poly_fun(i_index,min_neurons,max_neurons)

            #__Arquitecture duplicates detection____________________________________________________________________________
            arch_signature = tuple(neurons_per_layer)
            if arch_signature in self.seen_architectures:
                raise optuna.TrialPruned()  # Optuna descarta el trial y prueba otros parámetros
            self.seen_architectures.add(arch_signature)
            # _______________________________________________________________________________________________________________
            self.training_setup['Number of neurons'] = neurons_per_layer
            # Continuing with the training and returning the score to optimize.
            if self.hyperparameters_optimizer :
                self.optimize_hyperparameters()
            else :
                self.hyperparameters_grid()
            
            if self.feature_optimization :   
                self.compare_feat_optimizers(self.arch_score)

            return self.arch_score
        
        finally :
            self.cleanup_memory(aggressive=False)


    def poly_fun(self,i,n_min,max_neurons=None) :
        if self.training_setup['Number of layers'] > 2 :
            if self.optimize_width :
                m = (n_min-max_neurons)/(self.training_setup['Number of layers']-1)**i
                b = max_neurons
            else : 
                m = (n_min-self.neurons_for_opt)/(self.training_setup['Number of layers']-1)**i
                b = self.neurons_for_opt
            f = lambda layer : round(m * (layer-1)**i + b)
            neurons_per_layer = []
            for layer in range(1 , self.training_setup['Number of layers'] + 1) :
                neurons_per_layer.append(f(layer))
        elif self.training_setup['Number of layers'] == 2:
            neurons_per_layer = [self.neurons_for_opt,n_min]
        else :
            raise ValueError('Impossible to optimize a single layer. Change the layers setup')
            
        return neurons_per_layer
    
    #==============================================================================================
    #__REST OF HYPERPARAMETERS CONFIGURATION FUNCTIONS_____________________________________________
    
    def hyperparameters_grid(self) :
        for batch in self.batch_size : # List of int
            self.training_setup['Batch size'] = batch
            for lr in self.learning_rate : # List of int
                self.training_setup['Learning rate'] = lr
                for opt in self.optimizers : # List of int
                    self.training_setup['Optimization function'] = opt
                    for loss in self.loss_fun : # List of int
                        self.training_setup['Loss function'] = loss
                        try :
                            self.training_setup['Fold'] = 'F1'

                            score = self.training()
                            
                            if self.architecture_optimizer :
                                self.compare_arch_optimizers(score)
                            if self.feature_optimization :
                                self.compare_feat_optimizers(score)
                            self.global_scoring(score)      # NEW
                        finally :
                            self.cleanup_memory(aggressive=False)
                            
                            time.sleep(0.1)
    
    def objective_hyperparameters(self,trial) :
        """
        
        """
        if any(isinstance(obj,list) for obj in self.optimizers) :
            raise ValueError('The list of optimization functions requires to be a single list, rather than a list of lists')
        if any(isinstance(obj,list) for obj in self.loss_fun) :
            raise ValueError('The list of loss functions requires to be a single list, rather than a list of lists')
        try :
            # Bypass to optimize or not the batch size.
            if self.force2set_batch_size is None :
                self.training_setup['Batch size'] = trial.suggest_int('batch_size',self.batch_size_range[0],self.batch_size_range[-1],log=True)
            else :
                self.training_setup['Batch size'] = self.force2set_batch_size
            
            self.training_setup['Learning rate'] = trial.suggest_float('learning_rate',self.lr_range[0],self.lr_range[-1],log=True)
            
            optimizers_list = [self.optf_dict[i] for i in self.optimizers]
            losers_list = [self.lossf_dict[i] for i in self.loss_fun]
            opt_fun = trial.suggest_categorical('optimizer',optimizers_list)
            loss_fun = trial.suggest_categorical('loss_function',losers_list) 
            self.training_setup['Optimization function'] = self.inv_optf_dict[opt_fun]
            self.training_setup['Loss function'] = self.inv_lossf_dict[loss_fun]
            
            self.training_setup['Fold'] = 'F1' # NEW
            
            score = self.training()
            
            if self.architecture_optimizer :   
                self.compare_arch_optimizers(score)
            if self.feature_optimization :
                self.compare_feat_optimizers(score)
            self.global_scoring(score)      # NEW
            self.cleanup_memory()
            return score
        finally :
            self.cleanup_memory(aggressive=False)
    #_________________________________________________________________________________________________
    
    # ================================================================================================
    #  O  P  T  I  M  I  Z  A  T  I  O  N    F  U  N  C  T  I  O  N  S
    # ================================================================================================
    def optimize_architecture(self) :
        """
        
        """
        if self.use_optuna_db :
            storage = optuna.storages.RDBStorage(           # NEW
                url=f"sqlite:///trainings/optuna_study_{self.project_name}_architectures.db"  # Saves results
            )
            study = optuna.create_study(
                                        study_name=f"{self.gcn_model}_{self.training_setup['Node features']}_architectures_study",
                                        storage=storage,
                                        direction='minimize',
                                        load_if_exists=True,   # retakes optimization tests
                                        sampler=optuna.samplers.TPESampler(
                                            multivariate=True,
                                            group=True,
                                            seed=25)
                                        )
        else :
            study = optuna.create_study(direction='minimize',
                                        sampler=optuna.samplers.TPESampler(
                                        multivariate=True,                 #For no correlation between parameters
                                        group=True,
                                        seed=25))
        
        # Construction of the early stopping callback , patience , and/or timeout
        optimize_kwargs = {'n_trials' : self.arch_optimizer_trials,            # Number of iterations
                           'show_progress_bar' : True,
                           'gc_after_trial' : True}
        
        if self.arch_optimizer_timeout is not None :
            optimize_kwargs['timeout'] = self.arch_optimizer_timeout
           
        if self.arch_optimizer_patience is not None :
            optimize_kwargs['callbacks'] = [lambda study,trial: self.early_stop_callback(study ,trial, self.arch_optimizer_patience)]
        
        try:
            study.optimize(self.objective_architecture,**optimize_kwargs) 
        finally :
            del study
            self.cleanup_memory(aggressive=True)
                                        
    def optimize_hyperparameters(self) :
        """
        
        """
        if self.use_optuna_db :
            storage = optuna.storages.RDBStorage(           # NEW
                url=f"sqlite:///trainings/optuna_study_{self.project_name}_hiperparameters.db"  # Saves results
            )
            study = optuna.create_study(
                                        study_name=f"{self.gcn_model}_{self.training_setup['Node features']}_{self.training_setup['Number of neurons']}_hiperparaemters_study",
                                        storage=storage,
                                        direction='minimize',
                                        load_if_exists=True,   # retakes optimization tests
                                        sampler=optuna.samplers.TPESampler(
                                            multivariate=True,
                                            group=True,
                                            seed=25)
                                        )
        else :
            study = optuna.create_study(direction='minimize',
                                        sampler=optuna.samplers.TPESampler(
                                        multivariate=True,                 #For no correlation between parameters
                                        group=True,
                                        seed=25))
        
        # Construction of the early stopping callback , patience , and/or timeout
        optimize_kwargs = {'n_trials' : self.hyper_optimizer_trials,            # Number of iterations
                           'show_progress_bar' : True,
                           'gc_after_trial' :True}
        if self.hyper_optimizer_timeout is not None :
            optimize_kwargs['timeout'] = self.hyper_optimizer_timeout
        if self.hyper_optimizer_patience is not None :
            optimize_kwargs['callbacks'] = [lambda study,trial: self.early_stop_callback(study , trial,self.hyper_optimizer_patience)]
        
        try :
            study.optimize(self.objective_hyperparameters,**optimize_kwargs)   
        finally :
            del study
            self.cleanup_memory(aggressive=True)   
    #__Architecture optimization score comparison function____________________________________________
    def compare_arch_optimizers(self,new_score) :
        if new_score < self.arch_score :
            self.arch_score = new_score  
    #__Feature optimization score comparison function____________________________________________
    def compare_feat_optimizers(self,new_score) :
        if new_score < self.feat_score :
            self.feat_score = new_score
    # ================================================================================================   
    #   G  L  O  B  A  L    S  C  O  R  E
    # ================================================================================================
    def global_scoring(self,new_score) :    # NEW
        if new_score < self.global_score :
            self.global_score = new_score   
            # If the option of cross-validation is 
            if self.x_validation:           # NEW
                """
                Cross validation is only available for best scores
                """
                for k_fold in range(1,self.k_folds) :
                    # Updating the Deck for the folds
                    self.Deck.set_fold(k_fold)
                    self.training_setup['Fold'] = f'F{k_fold+1}'
                    self.training()
                del self.training_setup['Fold']
                self.Deck.set_fold(0)   # Restore to initial deck
                    
    # ================================================================================================
    #   E  A  R  L  Y - S  T  O  P    B  A  Y  E  S  S  I  A  N    O  P  T  I  M  I  Z  A  T  I  O  N
    # ================================================================================================
    def early_stop_callback(self,study,trial,patience) :
        trials = study.trials
        completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    
        if len(completed) < patience:
            return
    
        if study.best_trial.number < len(completed) - patience:
            study.stop()

    # ================================================================================================
    #   T  R  A  I  N    R  O  U  T  I  N  E
    # ================================================================================================
    def training(self) :
        train = None
        wandb_run = None
        try :
            # Initialize wandb run if enabled
            if self.use_wandb :
                Label_order = ['Dataset','Property','Node features','Number of layers', 
                                    'Number of neurons','Activation function','Random state',
                                    'Batch size','Learning rate','Optimization function',
                                    'Loss function','Random seed','Fold']
                wandb_run = wandb.init(
                    project=self.project_name,
                    name = '_'.join([str(self.training_setup[key]) for key in Label_order]) ,
                    config=self.training_setup,
                    reinit=True  # important : allows multiple runs on the same process
                )
            
            train = Train(self.training_setup,
                          self.Deck,
                          self.device,
                          self.learning_rate_scheduler,
                          self.scheduler_factor,
                          self.scheduler_patience,
                          self.epochs_number,
                          self.early_stopping,
                          self.early_stop_handicap,
                          self.early_stop_patience,
                          self.loss_cache,
                          self.optimizer_cache,
                          self.detransformfun,
                          self.cosine_annealing_scheduler,
                          self.plot_best_trials,
                          self.global_score)
            
            

            train.train_itinerary()
            
            # Log results to wandb
            if self.use_wandb and wandb_run is not None:
                # Prepare metrics to log
                metrics = {
                    "best_score": train.best_score,
                    "dataset": self.training_setup['Dataset'],
                    "property": self.training_setup['Property'],
                    "node_features": str(self.training_setup['Node features']),
                    "random_state": self.training_setup['Random state'],
                    "random_seed": self.training_setup['Random seed'],
                    "activation_function": self.activf_dict[self.training_setup['Activation function']],
                    "num_layers": self.training_setup['Number of layers'],
                    "batch_size": self.training_setup['Batch size'],
                    "learning_rate": self.training_setup['Learning rate'],
                    "optimizer": self.optf_dict[self.training_setup['Optimization function']],
                    "loss_function": self.lossf_dict[self.training_setup['Loss function']],
                }
                
                # Add neurons info (can be int or list)
                if isinstance(self.training_setup['Number of neurons'], list):
                    metrics["neurons_architecture"] = str(self.training_setup['Number of neurons'])
                else:
                    metrics["neurons_architecture"] = [int(self.training_setup['Number of neurons'])] * self.training_setup['Number of layers']
                
                # Add fold info if present
                if 'Fold' in self.training_setup:
                    metrics["fold"] = self.training_setup['Fold']
                
                wandb.log(metrics)
        
            return train.best_score
        
        finally :
            # Close wandb run
            if self.use_wandb and wandb_run is not None:
                wandb_run.finish()
            
            if train is not None:
                # Guardar el score antes de eliminar
                best_score = train.best_score if hasattr(train, 'best_score') else float('inf')
                del train
                self.cleanup_memory(aggressive=False)
                return best_score
    # ================================================================================================ 
    
    def main(self) :
        """
        Initialization of the experiment by choosing the features configuration
            - Bayessian optimization of the features 
            - Grid search of features
        """
        for rand_state in self.random_states : # List of int
            self.training_setup['Random state'] = rand_state
            
            if self.feature_optimization :
                self.optimize_features()
            else :
                if any(isinstance(element,list) for element in self.node_features) :
                    for feat_setup in self.node_features :
                        self.training_setup['Node features'] = feat_setup
                        self.hyperparameter_iteration_start()

                else :
                    self.training_setup['Node features'] = self.node_features
                    self.hyperparameter_iteration_start()

