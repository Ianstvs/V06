"""
Name of module: DATAHANDLER.

Description: Script for data extraction from .csv dataset file , convertion into graphs,
             data treatment and normalization, dataloaders generation, and geometric dataloaders

Author: Joshua Ian Hernandez Esteves.

Creation date: 07-03-26

Updated version: v1.1.0

Modification date: 28-04-26

Changes history: 
                - Optimization for graphic card
                
                - Optimization on graphs generation
                
                - Debugging of GPU optimization :  
                                                    - Fixed prefetch factor
                                                        True - > 3
                                                    - Pinned in memory PyG Dataloaders

                - Change of InMemoryDataset to TorchDataset
                
                - Training dataloaders: test and validation subsets have been deleted. Validation dataloaders have been implemented (they weren't used before)
                
"""
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split , KFold

from torch_geometric.data import InMemoryDataset

from multiprocessing import Pool, cpu_count

from torch.utils.data import Dataset as TorchDataset

class DataExtractor :
    """
    Class to handle the data within the .csv file
    """
    def __init__(self,file_path,
                 smile_label,
                 property_label,
                 node_features,
                 treated_data = None) :
        
        # ====================================================================================
        # F E A T U R E functions dict________________________________________________________
        self.features_dict = {
            0 : lambda atom : atom.GetAtomicNum(),
            1 : lambda atom : atom.GetDegree(),
            2 : lambda atom : atom.GetFormalCharge(),
            3 : lambda atom : int(atom.GetHybridization()),
            4 : lambda atom : int(atom.GetIsAromatic()),
            5 : lambda atom : atom.GetTotalNumHs(),
            6 : lambda atom : int(atom.IsInRing()),
            7 : lambda atom : float(atom.GetProp('_GasteigerCharge'))
        }
        
        # P A T H file________________________________________________________________________
        if treated_data is None :
            self.path_now = Path(__file__)
            self.base_path = self.path_now.parent.parent
            self.file_path = self.get_filepath(file_path)
        
            # D A T A H A N D L I N G_____________________________________________________________
            self.file_validation()
            self.features_validation(node_features)
            self.data = self.data_extraction(smile_label,property_label)
        else :
            if isinstance(treated_data,dict) :
                self.data = treated_data
            else:
                raise KeyError('An internal error has occurred with the data normalization treatment')
        self.data_ready = self.graph_list_creation(smile_label,property_label,node_features)
        # ====================================================================================
        
    # D A T A S E T   P A T H aquisition___________________________________________________________________________   
    def get_filepath(self,short_path) -> str :  # This is a STR
        str_base_path = str(self.base_path)
        str_file_path = str_base_path + '/' + short_path
        return str_file_path
    
    # V A L I D A T I O N functions_________________________________________________________________________
    def file_validation(self) :
        path_path = Path(self.file_path)
        if path_path.exists() == False:
            raise ValueError('Make sure the path introduced in the configuration exists.')
        else:
            if path_path.is_file() :
                if path_path.suffix != '.csv' :
                    raise ValueError("Please, make sure you've introduced a valid '.csv' file.")
            elif path_path.is_dir() :
                raise ValueError('Make sure the path given corresponds to a file and not to a directory.')
     
    def features_validation(self,features) :
        feat_dic = self.features_dict
        if any(feat not in feat_dic for feat in features):
            raise ValueError(f'The provided features list {features} contains unrecognized features.\n'
                            f'Valid keys are: {list(feat_dic.keys())}.\n '
                            'Please double check your setup configuration at your .json file.')
    
    # D A T A - P R O C E S S I N G_____________________________________________________________________________          
    def data_extraction(self,smile,property) -> dict:
        """
        Extracts requested data, by labels from the .csv file as a dictionary
      
        """
        all_data = pd.read_csv(self.file_path)
        
        labels = [smile,property]
        for label in labels :
            if label not in all_data.keys() :
                raise KeyError(f'The label {label} given in the configuration does not match with any keys on the dataset.\n KEYS: {all_data.keys()}')
        data = {
            smile : all_data[smile],
            property : all_data[property]     
        }
        return data
    
    def graph_list_creation(self,label_smiles,label_prop,node_features):
        graph_list = {
            'smiles' : self.data[label_smiles],
            'graphs' : []
        }

        for index,smiles in enumerate(graph_list['smiles']) :
            graph = self.smiles2graphs(smiles,self.data[label_prop][index],node_features)
            graph_list['graphs'].append(graph)
        
        return graph_list
    
    def smiles2graphs(self,smiles,property,custom_list) :
        """
        Creation of a graph dependant to node features
        
        
        """
        rdkit_mol = Chem.MolFromSmiles(smiles)
        # Define a list characteristics of atoms
        
        if any(features == 7 for features in custom_list)  :
            AllChem.ComputeGasteigerCharges(rdkit_mol)
        atoms_features = [[self.features_dict[feature](atom) for feature in custom_list] 
                          for atom in rdkit_mol.GetAtoms()]
        if any(c is None or not np.isfinite(c) for atom in atoms_features for c in atom) :
            raise ValueError('While computing the features of a non computable value has popped up. \n\n'
                             f'Ensure the SMILE "{smiles}" is computable for the feature functions requested.')
        
        # Convert atom characteristics into a tensor
        if len(custom_list) == 1 :
            atoms_features = [feature for sublist in atoms_features for feature in sublist]
            atom_tensor = torch.tensor(atoms_features, dtype=torch.float).unsqueeze(1)
        else :
            atom_tensor = torch.tensor(atoms_features, dtype=torch.float)
        
        
        # Get bond from rdkit mol 
        bonds = []
        for bond in rdkit_mol.GetBonds():
            # Add x and y bond directions
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            bonds.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        # Convert bond data into a tensor
        bond_tensor = torch.tensor(bonds, dtype=torch.long).t().contiguous()
        # Convert property into a tensor 
        property_tensor = torch.tensor([property], dtype=torch.float)
        # Create data instance
        data = Data(x=atom_tensor, edge_index=bond_tensor, y = property_tensor)
        return data
    

class DataDealer :
    """
    Object that stores all the data that'll be used at the training and after.
    The input variables to create this object are :
            (graphs list : as a dictionary with keys as ['smiles'] and ['graphs] ,
             validation size : the proportion of the validation data with respect the dataset size ,
             test size : the proportion of the test data with respect the dataset size ,
             random state : the random state number ,
             dataset path : the dataset doument location)
             
    ------------------------------------------------------------------------------------------------         
    From this object you can extract the following data :
    
        .dataset_path : The location of the dataset given to the object.
        .smiles_dealed : The shuffled smiles dictionary split into the keys ['train'],['test'],['validation].
        .graphs_dealed : The shuffled graphs dictionary split into the keys ['train'],['test'],['validation].
        .evaluation_data : The evaluation dataloaders in a dictionary split into the keys ['train'],['test'],['validation].
        .train_geometric : The trianing data as a PytorchGeometric object.
        .test_geometric : The test data as a PytorchGeometric object.
        .validation_geometric : The validation data as a PytorchGeometric object.
    ________________________________________________________________________________________________
    ------------------------------------------------------------------------------------------------
    As of the methods, there's a single method available to create a dictionary of the training data as dataloader objects.
    The only input required is the batch size.
    
            .train_data_loaders(batch_size)

                retrurns {
                    'train'     : Dataloader(train_data)
                    'test'      : Dataloader(test_data)
                    'validation': Dataloader(val_data)
                }
    """
    def __init__(self,graphs_list,
                 validation_ratio,
                 test_ratio,random,
                 dataset_path,
                 x_validation=False,
                 k_fold=5) :
        
        self.dataset_path = dataset_path
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.random = random
        self.graphs_list = graphs_list
        self.x_validation = x_validation
        if self.x_validation :
            self.k_fold = k_fold
            smiles_remain_set , smiles_test_set , graphs_remain_set , graphs_test_set = train_test_split(self.graphs_list['smiles'],self.graphs_list['graphs'],test_size = test_ratio,random_state = self.random,shuffle = True)
            self.folds = self.x_val(smiles_remain_set,
                                    list(graphs_remain_set),
                                    smiles_test_set,
                                    list(graphs_test_set),
                                    self.k_fold)
        
        self.smiles_dealed , self.graphs_dealed = self.data_dealing(graphs_list,validation_ratio,
                                                                    test_ratio,random)
        self.evaluation_data = self.evaluation_data_loaders()
        
    def data_dealing(self,list_graph,val_ratio,test_ratio,random_state,k=0) :
        if not self.x_validation :
            smiles_train_set , smiles_remain_set , graphs_train_set , graphs_remain_set = train_test_split(list_graph['smiles'],list_graph['graphs'],test_size = (val_ratio + test_ratio),random_state = random_state,shuffle = True)
            remain_test_ratio = test_ratio/(val_ratio+test_ratio)
            smiles_val_set , smiles_test_set , graphs_val_set , graphs_test_set = train_test_split(smiles_remain_set,graphs_remain_set,test_size = remain_test_ratio,random_state = random_state,shuffle = True)
        else :
            smiles_train_set = self.folds[k]['train']['smiles']
            smiles_test_set = self.folds[k]['test']['smiles']
            smiles_val_set = self.folds[k]['validation']['smiles']
            
            graphs_train_set = self.folds[k]['train']['graphs']
            graphs_test_set = self.folds[k]['test']['graphs']
            graphs_val_set = self.folds[k]['validation']['graphs']
                    
        shuffled_smiles = {
            'train' : list(smiles_train_set),
            'test' : list(smiles_test_set),
            'validation' : list(smiles_val_set)
        }
        
        shuffled_graphs = {
           'train' : list(graphs_train_set),
            'test' : list(graphs_test_set),
            'validation' : list(graphs_val_set)
        }
        return shuffled_smiles , shuffled_graphs
    
    def evaluation_data_loaders(self) : 
        self.train_geometric = PytorchGeometricDataset(self.dataset_path,
                                                       self.graphs_dealed['train'])
        self.test_geometric = PytorchGeometricDataset(self.dataset_path,
                                                      self.graphs_dealed['test'])
        self.validation_geometric = PytorchGeometricDataset(self.dataset_path,
                                                            self.graphs_dealed['validation'])
        
        evaluation_dataloaders ={
            'train' : DataLoader(self.train_geometric,
                                 batch_size = len(self.graphs_dealed['train']),
                                 shuffle = False,    # To show different arrengements in each epoch
                                 drop_last = False,
                                 pin_memory = True),    # OPT 
            'test' : DataLoader(self.test_geometric,
                                batch_size = len(self.graphs_dealed['test']),
                                shuffle = False,
                                drop_last = False,
                                pin_memory = True),     # OPT
            'validation' : DataLoader(self.validation_geometric,
                                      batch_size = len(self.graphs_dealed['validation']),
                                      shuffle = False,
                                      drop_last = False,
                                      pin_memory = True)# OPT
        }
        return evaluation_dataloaders
    
    def train_data_loaders(self,batch_size) :
        
        training_dataloaders ={
            'train' : DataLoader(self.train_geometric,
                                 batch_size = batch_size,
                                 shuffle = True,
                                 drop_last = False,
                                 num_workers=4,              # OPT: Parallel loading
                                 pin_memory=True,            # OPT: Fast transferences to GPU 
                                 persistent_workers=False,    # OPT: Reusable workers 
                                 prefetch_factor=2)                          # OPT: 2 or 3 factor  
        }
        return training_dataloaders
    
    def x_val(self,remain_smiles,remain_graphs,test_smiles,test_graphs,k_folds) :          # NEW
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.random)
        folds = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(remain_smiles)))):
            folds.append({
                'fold' : fold,
                'train': {
                    'smiles' : [remain_smiles[i] for i in train_idx],
                    'graphs' : [remain_graphs[i] for i in train_idx]
                },
                'test': {
                    'smiles' : test_smiles,
                    'graphs' : test_graphs
                },
                'validation': {
                    'smiles' : [remain_smiles[i] for i in val_idx],
                    'graphs' : [remain_graphs[i] for i in val_idx]
                }
            })

        return folds
    
    def set_fold(self, k) :
        if not self.x_validation:
            raise RuntimeError('set_fold() is only available when x_validation=True')
        if k >= self.k_fold :
            raise IndexError(f'Fold {k} does not exist. Available folds: 0 to {self.k_fold-1}')
        self.smiles_dealed , self.graphs_dealed = self.data_dealing(self.graphs_list,
                                                                    self.validation_ratio,
                                                                    self.test_ratio,
                                                                    self.random,
                                                                    k= k)
        # Update dataloaders
        self.evaluation_data = self.evaluation_data_loaders()
        
    
    
class PytorchGeometricDataset(InMemoryDataset):
    """Create a specific dataset for pytorch geometric library using a dataset path and list of grafos as input.

    Args:
        InMemoryDataset (class): Class InMemoryDataset
    """
    def __init__(self, root, data_list, transform=None, pre_transform=None):
        super(PytorchGeometricDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(data_list) 
