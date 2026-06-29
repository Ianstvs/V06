"""
Name of module: PLOTTER.

Description: Script for registering metrics and predictions of the trained model into 
             scatter plots, plots, and .csv files.

Author: Joshua Ian Hernandez Esteves.

Creation date: 07-02-26

Modification date : 23-03-26

Versión actual: v1.0.2

Changes history: 
                - plotting_restristor : Restricts the plotting of trivial trainings, determined
                  by the score of the best performed model.
                
                - fold label :  A new 'fold' column to detect k-folds from regular models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
# import matplotlib as mpl  (only needed for colorbar in scatterplot)
import seaborn as sns
from datetime import date
from sklearn.metrics import r2_score


from pathlib import Path
 
import torch

class AllStats :
    def __init__(self,smiles_list,metrics,true_pred_properties,lables_list,model,plot_best_trials=False,global_score=float('inf')) :
        # ===============================================
        # Labels lists
        self.node_features_list = {
            0 : 'AtomNum',
            1 : 'Degree',
            2 : 'FormalCh',
            3 : 'Hybridization',
            4 : 'Aromatic',
            5 : 'NumHs',
            6 : 'InRing',
            7 : 'GastCh'
        }
        self.activf_list = {
            0 : 'ELU' ,
            1 : 'GELU' ,
            2 : 'SiLU' ,
            3 : 'Mish' ,
            4 : 'LeakyReLU' ,
            5 : 'ReLU' ,
            6 : 'Sigmoid' ,
            7 : 'Tanh' 
        }
        self.optf_list = {
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
        self.lossf_list = {
            0 : 'L1Loss',
            1 : 'MSELoss',
            2 : 'HuberLoss',
            3 : 'SmoothL1Loss'    
        }
        # ===============================================
        # ===============================================
        # Data, metrics,and model importation
        
        self.labels = copy.deepcopy(lables_list)     # Making a copy of the train setup labels
        self.model = model          # Trained model
        self.metrics = metrics
        self.properties = true_pred_properties
        self.smiles = smiles_list
        self.plot_best_trials = plot_best_trials            # NEW
        self.global_score = global_score                    # NEW
        # ===============================================
        # Directory paths
        self.main_dir = Path(__file__).resolve().parent.parent
        self.trainings = self.main_dir / 'trainings'
        self.plots = self.trainings / 'plots_repository'
        self.metrics_pth = self.trainings / 'training_metrics'
        self.trained = self.trainings / 'trained_models'
        self.eval =self.trainings / 'training_evaluations'
        
        # ===============================================
        # Validation and labeling
        self.validate_dirs()
        self.validate_scores_list()
        self.training_label_list = self.training_label_creator()
        
        # ===============================================
        # Dataframe creation and saving of all data generated in the training 
        self.metrics_df = self.metrics_dataframe()
        self.eval_df = self.evaluations_dataframe()
        
        # ===============================================
        # Registering score
        self.registering_training_scores()
        
        # ===============================================
        # Plotting
        self.plot_settings()
        self.plotting_restrictor()
        
        # ===============================================
        # Saving model
        self.saving_model()

        
    
    ### D I R E C T O R Y   S E T T I N G S ###
    
    # Validate the existence of the directory, if not, it creates it.
    def validate_dirs(self) :
        for folder in [self.plots,self.metrics_pth,self.trained,self.eval] :
            folder.mkdir(parents=True, exist_ok=True)

    def plot_settings(self) :
        sns.set_theme(
        style="whitegrid",
        context="talk",
        font="serif"
        )

        plt.rcParams.update({
            "figure.dpi": 300,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        })

    def validate_scores_list(self) :
        self.train_scores = self.trainings / 'training_scores_list.csv'
        
        if self.train_scores.exists() and self.train_scores.is_file() :
            pass
        else :
            # Modify the file appending the new data  
            column_names = ['Dataset','Property','Node features','Number of layers', 'Number of neurons',
                           'Activation function','Random state','Batch size',
                           'Learning rate','Optimization function','Loss function',
                           'Random seed','Epoch','Fold']
            scores_list = ['loss','MAE','MSE','R2','Date']
            
            column_names = [*column_names,*scores_list]
            scores_df = pd.DataFrame(columns=column_names)
            scores_df.to_csv(self.train_scores , index=False) 
    
    def training_label_creator(self) :
        label_list = []
        Order_of_hyperparameters = ['Dataset','Property','Node features','Number of layers', 
                                    'Number of neurons','Activation function','Random state',
                                    'Batch size','Learning rate','Optimization function',
                                    'Loss function','Random seed','Epoch']
        # Renaming label functions (according to number reference)
        self.labels['Node features'] = [self.node_features_list[feature] for feature in self.labels['Node features']]
        self.labels['Activation function'] = self.activf_list[self.labels['Activation function']]
        self.labels['Optimization function'] = self.optf_list[self.labels['Optimization function']]
        self.labels['Loss function'] = self.lossf_list[self.labels['Loss function']]
        if 'Fold' in self.labels :      # Add Fold label if cross validation is active
            Order_of_hyperparameters.append('Fold')
        # Getting an ordered list of hyperparameter training setup
        for hyper in Order_of_hyperparameters :
            label_list.append(str(self.labels[hyper]))
        
        self.training_name = "_".join(label_list)
        
        return label_list
        
    ### P L O T I N G   S E T T I N G S ###
    
    def metrics_dataframe(self) :
        metrics_list = {
            'subset' : [] ,
            'epoch' : [] ,
            'loss' : [] ,
            'MAE' : [] ,
            'MSE' : [] ,
            'R2' : []
        }
        # To create a dict with all the metrics/epoch for all subsets.
        for metric in self.metrics['train'].keys() :    # The key given is irrelevant, its just needed to extract the keys of the dicts contained in each key.
            for subset in self.metrics :
                # Appending the metric in subset order
                metrics_list[metric] = [*metrics_list[metric],*self.metrics[subset][metric]]
        # To assign a subset label and epoch to each training ussing append
        for subset in self.metrics.keys() :
            list_lenght = len(self.metrics[subset]['loss']) # The key 'loss' can be replaced by any other key present in the subset dict.
            metrics_list['subset'] = [*metrics_list['subset'],*[subset]*list_lenght] 
            metrics_list['epoch'] = [*metrics_list['epoch'],*list(range(list_lenght))]
        
        metrics_df = pd.DataFrame(metrics_list)
        
        # Creating a name for the dataframe
        name = f'TRAINING_{self.training_name}_METRICS.csv'
        path_name = self.metrics_pth / name
        # Importing the metrics dataframe to a csv file
        metrics_df.to_csv(path_name, index = False)
        
        return metrics_df
            
    def evaluations_dataframe(self) :
        evaluation_list = {
            'smiles' : [] ,
            'subset' : [] ,
            'property' : [],
            'prediction' : []
        }
        ## Filling the evaluation list with the lists of smiles and properties
        for subset in self.smiles.keys() :
            evaluation_list['smiles'] = [*evaluation_list['smiles'],*self.smiles[subset]]
            evaluation_list['subset'] = [*evaluation_list['subset'],*[subset]*len(self.smiles[subset])]   
        for subset in self.properties.keys() :
            evaluation_list['property'] = [*evaluation_list['property'],*self.properties[subset]['originals']]
            evaluation_list['prediction'] = [*evaluation_list['prediction'],*self.properties[subset]['predictions']]
        
        # Creating the dataframe with all predictions
        evaluation_df = pd.DataFrame(evaluation_list)
        
        # Creating a name for the dataframe
        name = f'TRAINING_{self.training_name}_EVALUATION.csv'
        path_name = self.eval / name
        # Importing the evaluations dataframe to a csv file
        evaluation_df.to_csv(path_name, index = False)
        
        return evaluation_df
    
    def plotting_restrictor(self) :
        if self.plot_best_trials :
            if self.global_score > self.best_MSE :
                self.metrics_vs_epochs_plot()
                self.pred_orig_scatter_plot()
        else :
            self.metrics_vs_epochs_plot()
            self.pred_orig_scatter_plot()
                
        
    def metrics_vs_epochs_plot(self) :
        metrics = ['loss','MAE','MSE','R2']
        # ===============================
        # Extracting plot info from DF
        # ===============================
        train_data = self.metrics_df.loc[self.metrics_df['subset'] == 'train',['epoch','loss','MAE','MSE','R2']] 
        test_data = self.metrics_df.loc[self.metrics_df['subset'] == 'test',['epoch','loss','MAE','MSE','R2']] 
        val_data = self.metrics_df.loc[self.metrics_df['subset'] == 'validation',['epoch','loss','MAE','MSE','R2']] 
        
        for metric in metrics :
            # ===============================
            # Plots creation
            # ===============================
            fig, ax = plt.subplots(figsize=(9, 6))
            
            # Train metrics
            sns.lineplot(
                x=train_data['epoch'].tolist(),
                y=train_data[metric].tolist(),
                ax=ax,
                linewidth=2.5,
                label="Train",
                color ="#1f77b4"
            )
            # Validation metrics
            sns.lineplot(
                x=val_data['epoch'].tolist(),
                y=val_data[metric].tolist(),
                ax=ax,
                linewidth=2.5,
                linestyle="-",
                label="Validation",
                color = "#2ca02c"
            )
            # Test metrics
            sns.lineplot(
                x=test_data['epoch'].tolist(),
                y=test_data[metric].tolist(),
                ax=ax,
                linewidth=2.5,
                linestyle="-",
                label="Test",
                color ="#d62728"
            )

            # ===============================
            # Labels & title
            # ===============================
            ax.set_title(f"{metric} vs epochs", pad=15)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)

            # ===============================
            # Visual settings
            # ===============================
            if metric == 'R2' :
                ax.set_ylim(0,1)
            ax.legend(frameon=False)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.margins(x=0)

            plt.tight_layout()
            
            # ===============================
            # Importing the plot
            # ===============================
            # Creating a name
            name = f'{metric}_{self.training_name}_PLOT.png'
            path_name = self.plots / name
            
            # Importing the plot to its respective dir.
            plt.savefig(path_name,bbox_inches = 'tight')
            
            plt.close(fig)
                       
    def pred_orig_scatter_plot(self) :
        subsets = self.eval_df['subset'].unique().tolist()
        for subset in subsets :
            # Extracting the required data by subset
            df = self.eval_df.loc[self.eval_df['subset'] == subset , ['property','prediction']]
            fig, ax = plt.subplots(figsize=(8, 6))

            sns.scatterplot(
                data=df,
                x="property",
                y="prediction",
                #hue = "z",        
                #palette='plasma',
                legend=False,
                ax=ax,
                s=10,                 # tamaño del punto
                alpha=0.3,           # transparencia
                edgecolor="black",
                linewidth=0.6,
                label='Data'
            )
            # ===============================
            # Colorbar (based on z)
            # ===============================
            #norm = mpl.colors.Normalize(
            #    vmin=df["z"].min(),
            #    vmax=df["z"].max()
            #)

            #sm = mpl.cm.ScalarMappable(
            #    cmap="plasma",
            #    norm=norm
            #)
            #sm.set_array([])  # requerido por Matplotlib

            #cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            #cbar.set_label("Variable Z")

            # ===============================
            # Trend line
            # ===============================
            sns.regplot(
                data=df,
                x="property",
                y="prediction",
                scatter=False,
                ax=ax,
                color="darkblue",
                line_kws={"linewidth": 2, "linestyle": "-"},
                label= 'Trend line'
            )


            # ===============================
            # Reference line
            # ===============================
            ax.plot(
                ax.get_xlim(), 
                ax.get_xlim(),
                linestyle="--", 
                color="darkred", 
                linewidth=2,
                label='Reference line')

            # ===============================
            # Labels & title
            # ===============================
            ax.set_title(f'"{subset}" subset scatter plot', pad=15)
            ax.set_xlabel("True data")
            ax.set_ylabel("Predicted data")
            ax.legend(frameon = False)

            # ===============================
            # R2 label
            # ===============================
            r2 = r2_score(df['property'].tolist(),
                          df['prediction'].tolist())
            ax.text(
                0.05, 0.95,
                f"$R^2 = {r2:.3f}$",
                transform=ax.transAxes,
                fontsize=13,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            # ===============================
            # Last settings
            # ===============================
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.tight_layout()
            
            # ===============================
            # Importing the plot
            # ===============================
            # Creating a name
            name = f'{subset}_SUBSET_{self.training_name}_SCATTER-PLOT.png'
            path_name = self.plots / name
            # Importing the plot to its respective dir.
            plt.savefig(path_name,bbox_inches = 'tight')
            
            plt.close(fig)                                                                  
            
    def saving_model(self) :
        # Creating a name
        name = f'MODEL_{self.training_name}.pth'
        # Creating the path
        path_name = self.trained / name
        torch.save(self.model.state_dict(), path_name)
      
    def registering_training_scores(self) :
        """
        Docstring for registering_training_scores
        
            The scores created are a weighted aggregation of the best performances across 
            epochs for test and validation subsets. Therefore, the overall performance of 
            the model might be misinterpreted since the best scores can occur at different epochs.
        """
        val_metrics = self.metrics_df.loc[self.metrics_df['subset']=='validation',['loss','MAE','MSE','R2']]
        test_metrics = self.metrics_df.loc[self.metrics_df['subset']=='test',['loss','MAE','MSE','R2']]
        minimum_location = int((test_metrics['MSE'].values + val_metrics['MSE'].values).argmin())  # NEW
        if not isinstance(minimum_location, int) :
             raise ValueError('The minimum location is not an integer, check the metrics dataframe (in the script)for possible errors.')
        i = minimum_location
        best_loss = (test_metrics['loss'].iloc[i] + val_metrics['loss'].iloc[i]) / 2      # NEW
        best_MAE = (test_metrics['MAE'].iloc[i] + val_metrics['MAE'].iloc[i])/2         # NEW
        self.best_MSE = (test_metrics['MSE'].iloc[i] + val_metrics['MSE'].iloc[i])/2         # NEW
        best_R2 = (test_metrics['R2'].iloc[i] + val_metrics['R2'].iloc[i])/2            # NEW
        new_data = {
            'Dataset' : self.training_label_list[0],
            'Property' : self.training_label_list[1],
            'Node features' : self.training_label_list[2],
            'Number of layers' : self.training_label_list[3],
            'Number of neurons' : self.training_label_list[4],
            'Activation function' : self.training_label_list[5],
            'Random state' : self.training_label_list[6],
            'Batch size' : self.training_label_list[7],
            'Learning rate' : self.training_label_list[8],
            'Optimization function' : self.training_label_list[9],
            'Loss function' : self.training_label_list[10],
            'Random seed' : self.training_label_list[11],
            'Epoch' : self.training_label_list[12],
            'Fold' : self.training_label_list[13],
            'loss' : best_loss,
            'MAE' : best_MAE,
            'MSE' : self.best_MSE,      # NEW
            'R2' :  best_R2,
            'Date' : str(date.today()), 
            'Label' : self.training_name        # NEW
        }
        scores_df = pd.read_csv(self.train_scores)
        scores_df = pd.concat([scores_df, pd.DataFrame([new_data])], ignore_index=True)
        scores_df.to_csv(self.train_scores , index=False)
#==========================================================================================