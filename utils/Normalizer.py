"""
Name of module: NORMALIZER.

Description: Script for data cleaning, filtration, property transformation, and normalization
             method selection.
             
Author: Joshua Ian Hernandez Esteves.

Creation date: 20-03-26

Versión actual: v1.0.0

Changes history: 
                - Creation of the module
"""
import pandas as pd
import numpy as np
from pathlib import Path

from typing import Tuple, Optional
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.preprocessing import PowerTransformer
from scipy.stats import normaltest


class Norm :
    """
    
    """
    def __init__(self,dataset_path,id_tag,smiles_tag,prop_tag) :
        
        self.smiles_tag = smiles_tag
        self.prop_tag = prop_tag
        self.id_tag = id_tag
         
        self.dataset_path = self.get_path(dataset_path)
        self.df_dataset = pd.read_csv(self.dataset_path)
        self.df_unique = self.get_uniques_df()
        self.df_organics , self.df_inorganics = self._filter_organics(self.df_unique)
        self.df_clean , self.df_blacklist = self._process_duplicates()
        self.df_valid = self.rdkit_validation(self.df_clean)
        self.norm_dataset = self.normalizers(self.df_valid)
        
    def get_path(self,str_path) :
        path_now = Path(__file__)
        base_path = path_now.parent.parent
        
        return str(base_path /str_path)
    
    def get_uniques_df(self) :
        df_raw = self.df_dataset[[self.id_tag,self.smiles_tag,self.prop_tag]].copy()
        df_unique = df_raw[[self.id_tag,self.smiles_tag,self.prop_tag]].drop_duplicates(subset=[self.smiles_tag], keep=False)
        
        return df_unique
    
    def is_organic(self,smiles) :
        """
        
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Elementos permitidos en moléculas orgánicas
            allowed_atoms = {'C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I'}
            
            # Verificar todos los átomos
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in allowed_atoms:
                    return False
            
            return True
            
        except:
            return False
    
    def _filter_organics(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separa moléculas orgánicas de inorgánicas usando RDKit.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con SMILES a filtrar
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (df_organics, df_inorganics)
        """
        # Operación vectorizada con apply (mucho más rápida que loops)
        df = df.copy()
        df['is_organic'] = df[self.smiles_tag].apply(self.is_organic)
        
        df_organic = df[df['is_organic']].drop(columns=['is_organic']).reset_index(drop=True)
        df_inorganic = df[~df['is_organic']].drop(columns=['is_organic']).reset_index(drop=True)
        
        return df_organic, df_inorganic
               
    def _process_duplicates(
        self,
        size_2_threshold_factor: float = np.sqrt(200),
        size_n_threshold_factor: float = 20
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Procesa SMILES duplicados y decide cuáles mantener según variabilidad.
        
        Criterio:
        - Si hay 2 réplicas: mantener si std <= |mean| / sqrt(200)
        - Si hay >2 réplicas: mantener si std <= |mean| / 20
        
        Parameters
        ----------
        size_2_threshold_factor : float
            Factor de umbral para réplicas de tamaño 2 (default: sqrt(200))
        size_n_threshold_factor : float
            Factor de umbral para réplicas de tamaño >2 (default: 20)
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (df_clean, df_blacklist) donde df_clean contiene datos únicos 
            + duplicados aceptables, y df_blacklist contiene duplicados sospechosos
        """
        # Identificar duplicados (SMILES que aparecen más de una vez)
        duplicates_mask = self.df_organics.duplicated(subset=[self.smiles_tag], keep=False)
        df_duplicates = self.df_organics[duplicates_mask]
        
        if len(df_duplicates) == 0:
            # No hay duplicados
            return self.df_organics, pd.DataFrame()
        
        # Agrupar y calcular estadísticas - OPERACIÓN VECTORIZADA
        blacklist_stats = df_duplicates.groupby(self.smiles_tag)[self.prop_tag].agg([
            'size', 'mean', 'min', 'max', 'std'
        ]).reset_index()
        
        # Calcular umbral de aceptación - OPERACIÓN VECTORIZADA
        def calculate_threshold(row):
            if row['size'] == 2:
                return abs(row['mean']) / size_2_threshold_factor
            else:
                return abs(row['mean']) / size_n_threshold_factor
        
        blacklist_stats['threshold'] = blacklist_stats.apply(calculate_threshold, axis=1)
        
        # Determinar status - OPERACIÓN VECTORIZADA
        blacklist_stats['status'] = np.where(
            blacklist_stats['std'] <= blacklist_stats['threshold'],
            'KEEP',
            'CHECK'
        )
        
        # Asignar IDs
        blacklist_stats.insert(0, 'ID', [f"BL-{i+1}" for i in range(len(blacklist_stats))])
        
        # Separar datos a mantener vs revisar
        blacklist_keep = blacklist_stats[blacklist_stats['status'] == 'KEEP'][[
            'ID', self.smiles_tag, 'mean'
        ]].rename(columns={'mean': self.prop_tag})
        
        blacklist_check = blacklist_stats[blacklist_stats['status'] == 'CHECK']
        
        # Dataset limpio: únicos + duplicados aceptables
        df_clean = pd.concat([self.df_organics, blacklist_keep], ignore_index=True)
        
        return df_clean, blacklist_check
    
    def _smiles_validation(self,smile):
        # 1. Validar si el SMILES es procesable por RDKit
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return 'DISCARD'
    
        # 2. Preparar la molécula (Hidrógenos)
        mol = Chem.AddHs(mol)
    
        try:
            # 3. Calcular cargas
            AllChem.ComputeGasteigerCharges(mol)
        
            # 4. Validar las cargas átomo por átomo
            for atom in mol.GetAtoms():
                # Verificamos si la propiedad existe
                if not atom.HasProp('_GasteigerCharge'):
                    return 'DISCARD'
            
                # Verificamos si es convertible a número (no es NaN ni string vacío)
                val = atom.GetProp('_GasteigerCharge')
                try:
                    float(val)
                except ValueError:
                    return 'DISCARD'
                
        except Exception:
            return 'GasCompProblem'
    
        return "KEEP"
    
    def rdkit_validation(self,clean_df) :
        df = clean_df.copy()
        df['status'] = df[self.smiles_tag].apply(self._smiles_validation)   
        df_validated = df[df['status'] == 'KEEP']  
        
        return df_validated
    
    def normalizers(self,_dataset) :
        dataset = _dataset.copy()
        proprty = self.prop_tag
        dataset['NORM'] = (dataset[proprty]-dataset[proprty].mean())/dataset[proprty].std()
        if any(prop <= 0 for prop in dataset[proprty]) :
            # Log-transformación para datos con valores negativos (se añade un shift para evitar log(0))
            dataset['LOG'] = dataset[proprty].apply(lambda x : np.log10(x))
        
        if any(prop < 0 for prop in dataset[proprty]) :
            
            # Name of the transformation
            name = 'YEOJ'
            
            # Configuration of the YEO-JOHNSON transformer
            self.pt = PowerTransformer(method='yeo-johnson', standardize=True)

            # Preparing the data (reshaping it)
            datos_reshaped = np.array(dataset[proprty].tolist()).reshape(-1, 1)

            # Adjust and transform
            dataset['YEOJ'] = self.pt.fit_transform(datos_reshaped).flatten().tolist()
            
            # Clean for outliers
            no_outliers_datasets =  self.outliers_cleaner(dataset,name)          
        
        else :
            # Name of the transformation
            name = 'BOXCOX'
            
            # Configuration of the BOX-COX transformer
            self.pt = PowerTransformer(method='box-cox', standardize=True)

            # Preparing the data (reshaping it)
            datos_reshaped = np.array(dataset[proprty].tolist()).reshape(-1, 1)

            # Adjust and transform
            dataset['BOXCOX']= self.pt.fit_transform(datos_reshaped).flatten().tolist()
            
            # Clean for outliers
            no_outliers_datasets =  self.outliers_cleaner(dataset,name)
            
        return no_outliers_datasets
            
    def outliers_cleaner(self,dataset,trans_label) :
        dataset_labels = [self.prop_tag,'NORM',trans_label]
        if 'LOG' in dataset.columns:
            dataset_labels.insert(2, 'LOG')  
        best_p = 0
        no_outliers_dataset = {}
        for name_dataset in dataset_labels :
            Q1 = dataset[name_dataset].quantile(0.25)
            Q3 = dataset[name_dataset].quantile(0.75)
            IQR = Q3-Q1
            L1 = Q1 - 1.5*IQR
            L2 = Q3 + 1.5*IQR
            name = name_dataset
            id = dataset.loc[(dataset[name_dataset]>=L1) & (dataset[name_dataset]<=L2),self.id_tag].tolist()
            smiles = dataset.loc[(dataset[name_dataset]>=L1) & (dataset[name_dataset]<=L2),self.smiles_tag].tolist()
            prop = dataset.loc[(dataset[name_dataset]>=L1) & (dataset[name_dataset]<=L2),name_dataset].tolist()

            stat , p = normaltest(prop)
            if best_p < p :
                best_p = p
                no_outliers_dataset['NAME'] = name
                no_outliers_dataset[self.id_tag] = id
                no_outliers_dataset[self.smiles_tag] = smiles
                no_outliers_dataset[self.prop_tag] = prop       # Transformed property
            
        if best_p == 0 :
            raise ValueError('For the given properties dataset, the distribution does not resemble normal one')
        self.selected_transformation = no_outliers_dataset['NAME']
        print(f'{self.selected_transformation} trainsformation was selected')
        
        return no_outliers_dataset 
    
    def detransform(self,external_dataset) :
        
        data = np.array(external_dataset).reshape(-1,1)
        recovered_data = self.pt.inverse_transform(data).flatten().tolist()
        
        return recovered_data