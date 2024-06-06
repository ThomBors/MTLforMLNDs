#!/usr/bin/env python

from sklearn.metrics import  accuracy_score,f1_score,roc_auc_score,precision_score,recall_score,precision_recall_curve,auc,balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
from tqdm.auto import tqdm as tq
import pandas as pd
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import BinaryAUROC,BinaryAUPRC


import STL.SingleNetUtils as SingleNetUtils
import STL.utils as utils
import copy
# from NetworkOptimal import *  # Import classes from your network.py file
from STL.Network import PDNetwork,ADNetwork,CNNetwork,LMCINetwork,MCINetwork,EMCINetwork,FTDNetwork


# set up
def SingleNetTunuing(net):
        
    model = copy.deepcopy(net)
    print(net)

    path_roi = '../../data/MRI_rois_20211114//MRI_rois_20211114.csv'
    
    project_wandb='MTL_experiment'
    selected_gender = ['M','F']

    # create dict to save all results
    result = {}
    result['target'] = {}
    result['data'] = {}
    result['random_state'] = {}
    result['Gender'] = {}
    result['accuracy'] = {}
    result['f1_score'] = {}
    result['AUC_ROC'] = {}
    result['AUC_PR'] = {}
    result['balanced_accuracy_score'] = {}
    result['precision_score'] = {}
    result['recall_score'] = {}
    result['threshold'] = {}
    result["total_neurons"] = {}
    result["total_layers"] = {}


    for r in [23, 362, 191, 80, 769, 328, 204, 281, 841, 31]:

        # inizilisation
        torch.cuda.manual_seed(r)
        torch.manual_seed(r)
        online = (r == 23)
        net = copy.deepcopy(model)

        # import data
        path_roi = '../../data/MRI_rois_20211114//MRI_rois_20211114.csv'
        path_colnames = '../../data/MRI_Features/MRI_Features.csv'
        selected_diagnosis = ['CN','AD','PD','LMCI','EMCI','MCI','FTD']
        selected_gender = ['M','F']
    
        """function to red data and create Training validation and test set equal for all"""
        #3.1# Import subject ROIs data with 4555 images
        df = pd.read_csv(path_roi)

        #3.2# Import new column names
        df_colnames = pd.read_csv(path_colnames)
        ## Rename 2 cells in df_colnames
        df_colnames.loc[df_colnames['Original Feature Name']=='total CNR','Feature ID']='SD007'
        df_colnames.loc[df_colnames['Original Feature Name']=='eTIV','Feature ID']='SD008'

        #4.1# Create dictionary using 2 columns of df_columns from old to new
        colnames_dict_old_to_new = dict(zip(df_colnames['Original Feature Name'], df_colnames['Feature ID']))

        #4.2# Rename features from old to new
        df = df.rename(columns=colnames_dict_old_to_new)

        ## Create dictionary using 2 columns of df_columns from new to old
        #colnames_dict_new_to_old = dict(zip(df_colnames['Feature ID'], df_colnames['Original Feature Name']))
        ## Rename features from new to old
        #df = df.rename(columns=colnames_dict_new_to_old)

        #4.3# Select 1 image per subject
        df = df.sort_values(by=['SD005','SD007'],ascending=False)
        df = df[~df.duplicated(subset=['SD005'])] 

        #4.4# Set SID as row index
        df = df.set_index('SD005')

        #4.5# ROI exclusion
        df = df[df.columns.difference(df_colnames[df_colnames['exclude']=='y']['Feature ID'].values)]

        #4.6# Add class
        df_class = pd.DataFrame({'SD003':df['SD003'].unique()})
        df_class['class'] = df_class.index
        df = pd.merge(df, df_class, how='left', on=['SD003']).set_index(df.index)

        ## create variable for gender
        df['SD002_M'] = [1 if i == 'M' else 0 for i in df['SD002']]
        #5.1# Select data subset - diag and gender
        df_CNAD = df[((df['SD003'].isin(selected_diagnosis))&(df['SD002'].isin(selected_gender)))]

        # 5.2# drop na
        df_CNAD = df_CNAD.dropna()
        ## freasurfer feature names
        fs_feature_names = list(df_CNAD.filter(regex=r'(^FS|SD004|SD008)').columns)

        ## Train/Test split with stratification on Study and class
        X_train,X_test, y_train,y_test = train_test_split(df_CNAD,df_CNAD['SD003'], 
                                                            test_size=0.1, 
                                                            random_state=23,
                                                            stratify=df_CNAD[['SD001','class']])

        
    
        X_train = X_train[fs_feature_names]
        X_test = X_test[fs_feature_names]

        # prepare data
        scaler = StandardScaler()
        # fit on training
        scaler.fit(X_train)
        # save StandardScaler
        #dump(scaler, 'model/std_scaler.bin', compress=True)

        X_train = pd.DataFrame(data=scaler.transform(X_train),index=X_train.index,columns=X_train.columns)
        
        X_test = pd.DataFrame(data=scaler.transform(X_test),index=X_test.index,columns=X_test.columns)

        X_train = X_train.reset_index(drop=True)
        
        X_test = X_test.reset_index(drop=True)

        # design target
        y_train_AD,y_train_PD,y_train_MCI,y_train_EMCI,y_train_LMCI,y_train_FTD,y_train_CN = utils.Create_target(y_train)
        
        y_test_AD,y_test_PD,y_test_MCI,y_test_EMCI,y_test_LMCI,y_test_FTD,y_test_CN = utils.Create_target(y_test)

        print(f"Train the model on {X_train.shape[0]} observation with {X_train.shape[1] } features and test it on {X_test.shape[0]}")
        
        diagnosis = net.__class__.__name__.split('Network')[0]
        
        print(f"{'-'*20} diagnosis: {diagnosis}")
        if diagnosis == 'AD':
            y_train = y_train_AD
            y_test = y_test_AD
            
            learning_rate = 0.01
            step_size = 10
            gamma = 0.5
            weight_decay = 2.5e-2
            

        elif diagnosis == 'PD':
            y_train = y_train_PD
            y_test = y_test_PD
            
            learning_rate = 0.01
            step_size = 10
            gamma = 0.5
            weight_decay = 2.5e-2
            

        elif diagnosis == 'FTD':
            y_train = y_train_FTD
            y_test = y_test_FTD
            
            learning_rate = 0.01
            step_size = 5
            gamma = 0.5
            weight_decay = 2.5e-4
            

        elif diagnosis == 'LMCI':
            y_train = y_train_LMCI
            y_test = y_test_LMCI
            
            learning_rate = 0.01
            step_size = 10
            gamma = 0.5
            weight_decay = 2.5e-4
            

        elif diagnosis == 'EMCI':
            y_train = y_train_EMCI
            y_test = y_test_EMCI
            
            learning_rate = 0.01
            step_size = 10
            gamma = 0.5
            weight_decay = 2.5e-4
            

        elif diagnosis == 'MCI':
            y_train = y_train_MCI
            y_test = y_test_MCI
            
            learning_rate = 0.01
            step_size = 10
            gamma = 0.5
            weight_decay = 2.5e-2
            

        elif diagnosis == 'CN':
            y_train = y_train_CN
            y_test = y_test_CN
                        
            learning_rate = 0.01
            step_size = 5
            gamma = 0.5
            weight_decay = 2.5e-4
            
        else:
            print('error')
        

        # data loader
        ds_train = SingleNetUtils.AD_Dataset(feature_data=X_train,target=y_train)
        dl_train =  DataLoader(ds_train,batch_size=64, num_workers=0,shuffle=True)

        ds_test = SingleNetUtils.AD_Dataset(feature_data=X_test,target=y_test)
        dl_test = DataLoader(ds_test,batch_size=64, num_workers=0,shuffle=False)
        


        total_neurons,total_layers = utils._total_num_neurons(net)

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        net.apply(init_weights)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)
        print(device)

        optimizer = optim.Adagrad(net.parameters(), lr=learning_rate,  weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size, gamma=gamma)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma)
        rs = pd.read_csv('../MTLExperiment/STL/experiments_stl/SingleNetVal_'+diagnosis+'.csv')
        threshold = rs['threshold'].mean()
        epochs = int(rs['EarlyStopping'].mean())

        


        # training loop

        print(f"""
        ###################################################################################
        #   architecture: {net}
        #   dataset: {path_roi}
        #   target: {diagnosis}
        #   random state: {r}
        #   selected_gender: {selected_gender}
        #   selected_diagnosis: {diagnosis}
        #   epochs: {epochs}
        #   learning_rate: {learning_rate}
        #   optimizer : {optimizer.__class__.__name__}
        #   batch size: 64
        #   scheduler: {scheduler.__class__.__name__}
        #   weight_decay : {weight_decay}
        #   step_size:{step_size}
        #   gamma : {gamma}
        ###################################################################################
            """)
        #8.2# set up wandb logging
        config={
                "architecture": net,
                "total_neurons": total_neurons,
                "total_weighted_layers": total_layers,
                "dataset": path_roi,
                "random state": r,
                "diagnosis":diagnosis,
                "selected_gender":selected_gender,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "optimizer" : optimizer.__class__.__name__,
                "batch size": 64,
                "scheduler": scheduler.__class__.__name__,
                "weight_decay" : weight_decay,
                "step_size":step_size,
                "gamma" : gamma
                }
        
        if online:
            wandb.init(
                # set the wandb project where this run will be logged
                project=project_wandb,
                # track hyperparameters and run metadata
                config=config
            )

        
        for epoch in range(1, epochs + 1):
            SingleNetUtils.train(net,dl_train,optimizer,scheduler, epoch,device=device,online=online)
            test_loss = SingleNetUtils.test(net, dl_test,epoch, device=device,online=online)
        wandb.finish()
        torch.save(net.state_dict(), 'network/'+net.__class__.__name__+str(r)+'.pth')
        
        

        ## test loop
        y_pred = []
        
        with torch.no_grad():
            # Plot the ROC curve
            for x,y in ds_test:
                x=x.to(device)
                output = net(x.unsqueeze(1))
                y_pred.append(output.item() )
        
        ## statistics        
        prediction_thr = [1 if i > threshold else 0 for i in y_pred]
        result['target'][r] = diagnosis
        result['data'][r] = 'test'
        result['Gender'][r] = ','.join(selected_gender)
        result['random_state'][r] = r
        result['accuracy'][r] = accuracy_score(y_test,prediction_thr)
        result['f1_score'][r] = f1_score(y_test,prediction_thr)
        result['AUC_ROC'][r] = roc_auc_score(y_test,prediction_thr)
        fpr, tpr, thresholds = precision_recall_curve(y_test,prediction_thr)
        try:
            result['AUC_PR'][r] = auc(fpr, tpr)
        except:
            result['AUC_PR'][r] = None
        result['precision_score'][r] = precision_score(y_test,prediction_thr)
        result['recall_score'][r] = recall_score(y_test,prediction_thr)
        result['threshold'][r] = threshold
        result['balanced_accuracy_score'] = balanced_accuracy_score(y_test,prediction_thr)
        result["total_neurons"][r]  = total_neurons
        result["total_layers"][r]  = total_layers
        df_results = pd.DataFrame(result)

        ### save results
        df_results.to_csv(str("STL/experiments_stl/SingleNetTest_"+str(diagnosis)+".csv"))


        del(net)




def mainSTL():
    classes = [obj for name, obj in globals().items() if isinstance(obj, type)]

    for cls in classes:
        if issubclass(cls, nn.Module):
            print(f"{'#'*10} Class: {cls.__name__}")
            Model = cls()
            SingleNetTunuing(net=Model)

if __name__ == "__mainSTL__":
    mainSTL()