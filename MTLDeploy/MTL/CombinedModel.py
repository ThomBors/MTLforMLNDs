#!/usr/bin/env python


from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,roc_curve,roc_auc_score,precision_score,recall_score,precision_recall_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm as tq
import pandas as pd
import numpy as np
import random
import wandb
import time
import yaml
import glob
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader, Dataset,WeightedRandomSampler
from torcheval.metrics import BinaryAUROC,BinaryAUPRC,BinaryF1Score

# custom made
import MTL.utils as utils
from MTL.famo import FAMO
from MTL.utils import AD_Dataset,Final_metrics
from MTL.trainig_testing import train,test,train_FAMO,test_FAMO,train_FL,test_FL,train_FLAMO,test_FLAMO
from MTL.loss_function import UniformWeighting,BinaryCrossEntropy,AutomaticWeightedLoss
from MTL.earlyStopper import EarlyStopper,EarlyStopperMTL
# from MTL.TransferLearningMetric import task_affitity

from MTL.Network import PDNetwork,ADNetwork,CNNetwork,LMCINetwork,MCINetwork,EMCINetwork,FTDNetwork,MultiTaskModel


def run_model(path_roi,
              path_colnames,
              path_result,
              result_folder,
              random_stat,
              selected_gender,
              selected_diagnosis,
              network_design,
              training_algortim,
              freeze,
              learning_rate,
              LR_schedurler,
              epochs,
              step_size,
              gamma,
              batch_size,
              weight_decay,
              project_wandb,
              online,
              save_resultsCSV,
              save_model):

    print(r"""
        ___  ________ _           _     _          _                     _   _      _   
        |  \/  |_   _| |         | |   | |        (_)                   | \ | |    | |  
        | .  . | | | | |     __ _| |___| |__   ___ _ _ __ ___   ___ _ __|  \| | ___| |_ 
        | |\/| | | | | |    / _` | |_  / '_ \ / _ \ | '_ ` _ \ / _ \ '__| . ` |/ _ \ __|
        | |  | | | | | |___| (_| | |/ /| | | |  __/ | | | | | |  __/ |  | |\  |  __/ |_ 
        \_|  |_/ \_/ \_____/\__,_|_/___|_| |_|\___|_|_| |_| |_|\___|_|  \_| \_/\___|\__|
                                                                                                                                                                                                                        
          """)
    #0# set random seed
    torch.manual_seed(random_stat)

    #1.1# Import subject ROIs data with 4555 images
    df = pd.read_csv(path_roi)

    #1.2# Import new column names
    df_colnames = pd.read_csv(path_colnames)
    ## Rename 2 cells in df_colnames
    df_colnames.loc[df_colnames['Original Feature Name']=='total CNR','Feature ID']='SD007'
    # df_colnames.loc[df_colnames['Original Feature Name']=='eTIV','Feature ID']='SD008'

    #2.1# Create dictionary using 2 columns of df_columns from old to new
    colnames_dict_old_to_new = dict(zip(df_colnames['Original Feature Name'], df_colnames['Feature ID']))

    #2.2# Rename features from old to new
    df = df.rename(columns=colnames_dict_old_to_new)

    ## Create dictionary using 2 columns of df_columns from new to old
    #colnames_dict_new_to_old = dict(zip(df_colnames['Feature ID'], df_colnames['Original Feature Name']))
    ## Rename features from new to old
    #df = df.rename(columns=colnames_dict_new_to_old)

    #2.3# Select 1 image per subject
    df = df.sort_values(by=['SD005','SD007'],ascending=False)
    df = df[~df.duplicated(subset=['SD005'])] 

    #2.4# Set SID as row index
    df = df.set_index('SD005')

    #2.5# ROI exclusion
    df = df[df.columns.difference(df_colnames[df_colnames['exclude']=='y']['Feature ID'].values)]

    #2.6# Add class
    df_class = pd.DataFrame({'SD003':df['SD003'].unique()})
    df_class['class'] = df_class.index
    df = pd.merge(df, df_class, how='left', on=['SD003']).set_index(df.index)

    #3.1# Select data subset - diag and gender
    df_CNAD = df[((df['SD003'].isin(selected_diagnosis))&(df['SD002'].isin(selected_gender)))]

    # 3.2# drop na
    df_CNAD = df_CNAD.dropna()

    #4.1# freasurfer feature names
    fs_feature_names = list(df_CNAD.filter(regex=r'(^FS|SD004)').columns)

    #4.2# Train/Test split with stratification on Study and class
    X_train,X_test, y_train,y_test = train_test_split(df_CNAD,df_CNAD['SD003'], 
                                                        test_size=0.1, 
                                                        random_state=23,
                                                        stratify=df_CNAD[['SD001','class']])


    #5.1# create target variable
    ## create variable for each diseases
    y_train_AD,y_train_PD,y_train_MCI,y_train_EMCI,y_train_LMCI,y_train_FTD,y_train_CN = utils.Create_target(y_train)
    y_test_AD,y_test_PD,y_test_MCI,y_test_EMCI,y_test_LMCI,y_test_FTD,y_test_CN = utils.Create_target(y_test)


    #6.1# select feature
    X_train = X_train[fs_feature_names]
    X_test = X_test[fs_feature_names]

    #6.2# compute normalization
    scaler = StandardScaler()
    scaler.fit(X_train)

    #6.3# apply normalization
    X_train = pd.DataFrame(data=scaler.transform(X_train),index=X_train.index,columns=X_train.columns)
    X_test = pd.DataFrame(data=scaler.transform(X_test),index=X_test.index,columns=X_test.columns)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    print(f"Train the model on {X_train.shape[0]} observation with {X_train.shape[1] } features and test it on {X_test.shape[0]}")

    #7.1# create data set and data DataLoader
    ds_train = AD_Dataset(feature_data=X_train, AD=y_train_AD, PD=y_train_PD, 
                        MCI=y_train_MCI, LMCI=y_train_LMCI, EMCI=y_train_EMCI, 
                        FTD=y_train_FTD,CN=y_train_CN)
    dl_train =  DataLoader(ds_train,batch_size=batch_size, num_workers=4,shuffle=True)#,sampler=sampler)

    
    ds_test = AD_Dataset(feature_data=X_test, AD=y_test_AD, PD=y_test_PD, 
                        MCI=y_test_MCI, LMCI=y_test_LMCI, EMCI=y_test_EMCI, 
                        FTD=y_test_FTD,CN=y_test_CN)

    dl_test = DataLoader(ds_test,batch_size=batch_size, num_workers=4,shuffle=False)


    #8# read pretrained single-task models
    all_files = []
    path = '../MTLExperiment/STL/experiments_stl/'
    files = glob.glob(os.path.join(path, "*.csv"))
    all_files.extend(files)
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    filtered_df = df[df['data'] == 'validation']
    result_df = filtered_df.groupby('target').max('f1_score')
    Best_ModelSTL = result_df['Unnamed: 0']
    
    #8.1# read pretrained single-task models
    task_specific_layers = {}
    for disease in selected_diagnosis:
        if disease == 'AD':
            net = ADNetwork()
        elif disease == 'PD':
            net = PDNetwork()
        elif disease == 'FTD':
            net = FTDNetwork()
        elif disease == 'MCI':
            net = MCINetwork()
        elif disease == 'LMCI':
            net = LMCINetwork()
        elif disease == 'EMCI':
            net = EMCINetwork()
        elif disease == 'CN':
            net = CNNetwork()
        
        r = Best_ModelSTL[disease]

        # load model
        str_mod = "network/" + disease + "Network"+str(r)+".pth"
        
        state_dict = torch.load(str_mod,map_location=torch.device('cpu'))
        
        # Load the state dictionary into the model
        net.load_state_dict(state_dict)
        #net = torch.load(str_mod)
        
        task_specific_layers[disease] = net


    # inizializationas suggestd by FAMO papaer
    def init_weights(m):
        if type(m) == nn.BatchNorm1d:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        

    #9 # MTL composition
    multi_task_model = MultiTaskModel(task_specific_layers,freeze=False)
    Freezer_hybrid = EarlyStopperMTL(patience=3,min_delta=0.2)
    # Apply the initialization function to the shared layers of your MTL multi_task_modelwork
    multi_task_model.shared_layers.apply(init_weights)

    

    ## enviroment settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_task_model.to(device)
    print(device)

    ## Early Stopper parameter
    patience = 5
    min_delta = 1


    #### UniformWeighting ####
    loss_fn = UniformWeighting()
    weight_opt = FAMO(n_tasks = 7, device = device, w_lr = learning_rate, gamma = 0.001)
    optimizer = optim.Adagrad(multi_task_model.parameters(),lr=learning_rate,weight_decay=weight_decay)

   
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size, gamma=gamma)

    all_files = []
    for i in range(111,1011):
        path = '../MTLExperiment/MTL/experiments_mtl/exp'+str(i)
        files = glob.glob(os.path.join(path, "*.csv"))
        all_files.extend(files)

    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

    filtered_df = df[df['data'] == 'Validation']
    result_df = filtered_df.groupby('Unnamed: 0').mean(numeric_only=True)
    
    epochs = int(result_df['EarlyStopping'].mean())

    print(f"""
    ###################################################################################
    #   architecture: {network_design}
    #   dataset: {path_roi}
    #   target: {'modified'}
    #   random state: {random_stat}
    #   selected_gender: {selected_gender}
    #   selected_diagnosis: {selected_diagnosis}
    #   epochs: {epochs}
    #   training_algortim: {training_algortim}
    #   learning_rate: {learning_rate}
    #   optimizer : {optimizer.__class__.__name__}
    #   batch size: {batch_size}
    #   scheduler: {scheduler.__class__.__name__}
    #   weight_decay : {weight_decay}
    #   gamma : {gamma}
    #   EarlyStopper
    #   patience: {patience}
    #   min_delta: {min_delta}
    ###################################################################################
    """)

    #8.2# set up wandb logging
    config={
            "architecture": network_design,
            "training_algortim": training_algortim,
            "dataset": path_roi,
            "target": 'modified',
            "random state": random_stat,
            "selected_diagnosis":selected_diagnosis,
            "selected_gender":selected_gender,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "freeze_parametrs": freeze,
            "optimizer" : optimizer.__class__.__name__,
            "batch size": batch_size,
            "scheduler": scheduler.__class__.__name__,
            "weight_decay" : weight_decay,
            "gamma" : gamma,
            "EarlyStopper_patience":patience,
            "EarlyStopper_min_delta":min_delta
            }

    if online:
        wandb.init(
            # set the wandb project where this run will be logged
            project=project_wandb,
            # track hyperparameters and run metadata
            config=config
        )

    

    best_threshold = {}
    for k,v in result_df['treshold'].items():
        if k not in best_threshold.keys():
            best_threshold[k] = {}
        best_threshold[k]['thresholds'] = v

    #9.1# training neural multi_task_modelwork
    t0 = time.time()
    bar = tq(range(1, epochs + 1), postfix={"Train Loss":0.0 ,"validation loss":0.0})
    for epoch in bar:
        train_loss = train_FAMO(multi_task_model,dl_train,loss_fn,optimizer,weight_opt,scheduler,epoch,online,device)
        test_loss,test_dict = test_FAMO(multi_task_model, dl_test,epoch,loss_fn,weight_opt,online,device)
        if freeze == 'Hybrid':
            Freezer_hybrid.early_stop(multi_task_model,test_dict)
        bar.set_postfix({'Train Loss' : train_loss,'validation loss':test_loss})
    if online:
        wandb.finish()
    t1 = time.time()

    #10.1# Test on test set
    #### testing
    y_pred = {}
    y_test_metrix = {}
    with torch.no_grad():
        # Plot the ROC curve
        for x in ds_test:
            output = multi_task_model(x['subject'].unsqueeze(1).to(device))
            for i in output.keys():
                if i not in y_pred.keys():
                    y_pred[i] = []
                y_pred_i = y_pred[i]
                y_pred_i.append(output[i])

                if i not in y_test_metrix.keys():
                    y_test_metrix[i] = []
                y_test_i = y_test_metrix[i]
                y_test_i.append(x[i])

    res_test = Final_metrics(y_pred=y_pred,y_test_metrix=y_test_metrix,y_test_CN=y_test_CN,best_threshold=best_threshold,DataSetName='Test')

    if save_resultsCSV:
        ### save results
        os.makedirs(result_folder, exist_ok=True)
        os.makedirs(os.path.join(result_folder,path_result), exist_ok=True)

        #### save result csv
        df_results_test = pd.DataFrame(res_test)
        df_results_test.to_csv(result_folder+'/'+path_result+'/results_test.csv')

        # #### save configuration file
        with open(result_folder+'/'+path_result+'/config.yaml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

    #### save model
    if save_model:
        torch.save(multi_task_model.state_dict(), 'network/'+multi_task_model.__class__.__name__+str(random_stat)+'.pth')

    



def main():
    run_model(path_roi='../../data/MRI_rois_20211114//MRI_rois_20211114.csv',
            path_colnames='../../data/MRI_Features/MRI_Features.csv',
            path_result='exp'+ str(1000),
            result_folder = 'experiments_mtl',
            random_stat = 23,
            selected_gender = ['M','F'],
            selected_diagnosis = ['CN','AD','PD','LMCI','EMCI','MCI','FTD'],
            network_design = 'CombinMTL',
            training_algortim = 'FAMO',
            freeze = 'Hybrid',
            learning_rate = 0.001,
            LR_schedurler = 'StepLR',
            epochs = 1000,
            step_size = 500,
            gamma = 0.5,
            batch_size = 256,
            weight_decay = 0.00025,
            project_wandb = "MultiDiseases_Greece",
            online = True,
            save_resultsCSV=False,
            save_model = False)
                    
if __name__ == "__main__":
    main()