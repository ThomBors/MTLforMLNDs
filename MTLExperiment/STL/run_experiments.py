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

from STL.earlyStopper import EarlyStopper
import STL.SingleNetUtils as SingleNetUtils
import STL.CreateTVT as CreateTVT
import STL.utils as utils
import copy

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
    result['architecture'] = {}
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
    result['EarlyStopping'] = {}
    result["total_neurons"] = {}
    result["total_layers"] = {}
    # create dict to save all results
    result_validation = {}
    result_validation['target'] = {}
    result_validation['data'] = {}
    result_validation['architecture'] = {}
    result_validation['random_state'] = {}
    result_validation['Gender'] = {}
    result_validation['accuracy'] = {}
    result_validation['f1_score'] = {}
    result_validation['AUC_ROC'] = {}
    result_validation['AUC_PR'] = {}
    result_validation['balanced_accuracy_score'] = {}
    result_validation['precision_score'] = {}
    result_validation['recall_score'] = {}
    result_validation['threshold'] = {}
    result_validation['EarlyStopping'] = {}
    result_validation["total_neurons"] = {}
    result_validation["total_layers"] = {}


    for r in [23, 362, 191, 80, 769, 328, 204, 281, 841, 31]:

        # inizilisation
        torch.cuda.manual_seed(r)
        torch.manual_seed(r)
        online = (r == 23)
        net = copy.deepcopy(model)

        # import data
        X_train,y_train,X_validation,y_validation,X_test,y_test = CreateTVT.createTVT(
            path_roi = '../../data/MRI_rois_20211114//MRI_rois_20211114.csv',
            path_colnames = '../../data/MRI_Features/MRI_Features.csv',
            selected_diagnosis = ['CN','AD','PD','LMCI','EMCI','MCI','FTD'],
            selected_gender = ['M','F'],
            random_seed = r)

        # prepare data
        scaler = StandardScaler()
        # fit on training
        scaler.fit(X_train)
        # save StandardScaler
        #dump(scaler, 'model/std_scaler.bin', compress=True)

        X_train = pd.DataFrame(data=scaler.transform(X_train),index=X_train.index,columns=X_train.columns)
        X_validation = pd.DataFrame(data=scaler.transform(X_validation),index=X_validation.index,columns=X_validation.columns)
        X_test = pd.DataFrame(data=scaler.transform(X_test),index=X_test.index,columns=X_test.columns)

        X_train = X_train.reset_index(drop=True)
        X_validation = X_validation.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)

        # design target
        y_train_AD,y_train_PD,y_train_MCI,y_train_EMCI,y_train_LMCI,y_train_FTD,y_train_CN = utils.Create_target(y_train)
        y_validation_AD,y_validation_PD,y_validation_MCI,y_validation_EMCI,y_validation_LMCI,y_validation_FTD,y_validation_CN = utils.Create_target(y_validation)
        y_test_AD,y_test_PD,y_test_MCI,y_test_EMCI,y_test_LMCI,y_test_FTD,y_test_CN = utils.Create_target(y_test)

        print(f"Train the model on {X_train.shape[0]} observation with {X_train.shape[1] } features and test it on {X_test.shape[0]}")
        
        diagnosis = net.__class__.__name__.split('Network')[0]
        
        print(f"{'-'*20} diagnosis: {diagnosis}")
        if diagnosis == 'AD':
            y_train = y_train_AD
            y_validation = y_validation_AD
            y_test = y_test_AD
            
            learning_rate = 0.01
            epochs = 50
            step_size = 10
            gamma = 0.5
            weight_decay = 2.5e-2
            early_stopper = EarlyStopper(patience=3,min_delta=0.2)

        elif diagnosis == 'PD':
            y_train = y_train_PD
            y_validation = y_validation_PD
            y_test = y_test_PD
            
            learning_rate = 0.01
            epochs = 50
            step_size = 10
            gamma = 0.5
            weight_decay = 2.5e-2
            early_stopper = EarlyStopper(patience=2,min_delta=0.1)

        elif diagnosis == 'FTD':
            y_train = y_train_FTD
            y_validation = y_validation_FTD
            y_test = y_test_FTD
            
            learning_rate = 0.01
            epochs = 50
            step_size = 5
            gamma = 0.5
            weight_decay = 2.5e-4
            early_stopper = EarlyStopper(patience=3,min_delta=0.2)

        elif diagnosis == 'LMCI':
            y_train = y_train_LMCI
            y_validation = y_validation_LMCI
            y_test = y_test_LMCI
            

            learning_rate = 0.01
            epochs = 50
            step_size = 10
            gamma = 0.5
            weight_decay = 2.5e-4
            early_stopper = EarlyStopper(patience=3,min_delta=0.2)

        elif diagnosis == 'EMCI':
            y_train = y_train_EMCI
            y_validation = y_validation_EMCI
            y_test = y_test_EMCI
            

            learning_rate = 0.01
            epochs = 50
            step_size = 10
            gamma = 0.5
            weight_decay = 2.5e-4
            early_stopper = EarlyStopper(patience=3,min_delta=0.2)

        elif diagnosis == 'MCI':
            y_train = y_train_MCI
            y_validation = y_validation_MCI
            y_test = y_test_MCI
            
            learning_rate = 0.01
            epochs = 50
            step_size = 10
            gamma = 0.5
            weight_decay = 2.5e-2
            early_stopper = EarlyStopper(patience=3,min_delta=0.2)

        elif diagnosis == 'CN':
            y_train = y_train_CN
            y_validation = y_validation_CN
            y_test = y_test_CN
                        
            learning_rate = 0.01
            epochs = 50
            step_size = 5
            gamma = 0.5
            weight_decay = 2.5e-4
            early_stopper = EarlyStopper(patience=3,min_delta=0.2)
        else:
            print('error')
        

        # data loader
        ds_train = SingleNetUtils.AD_Dataset(feature_data=X_train,target=y_train)
        dl_train =  DataLoader(ds_train,batch_size=64, num_workers=0,shuffle=True)

        ds_validation = SingleNetUtils.AD_Dataset(feature_data=X_validation,target=y_validation)
        dl_validation =  DataLoader(ds_validation,batch_size=64, num_workers=0,shuffle=False)

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
        # #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=gamma)

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
            validation_loss = SingleNetUtils.test(net, dl_validation,epoch, device=device,online=online)
            if early_stopper.early_stop(validation_loss):
                                    break
        wandb.finish()

        
        torch.save(net.state_dict(), 'network/'+net.__class__.__name__+str(r)+'.pth')
        
        ## validation loop and treshold optimization
        y_pred = []
        y_trut = []
        with torch.no_grad():
            for x,y in ds_validation:
                x = x.to(device)
                output = net(x.unsqueeze(1))
                y_pred.append(output.item() )
                y_trut.append(y)
            y_trut = np.array(y_trut)

        st = SingleNetUtils.get_metric_and_best_threshold_from_roc_curve(y_trut,y_pred,show_result=False)
        
        ## statistics        
        prediction_thr = [1 if i > st['best_threshold'] else 0 for i in y_pred]
        result_validation['target'][r] = diagnosis
        result_validation['data'][r] = 'validation'
        result_validation['architecture'][r] = net.__class__.__name__
        result_validation['Gender'][r] = ','.join(selected_gender)
        result_validation['random_state'][r] = r
        result_validation['accuracy'][r] = accuracy_score(y_trut,prediction_thr)
        result_validation['f1_score'][r] = f1_score(y_trut,prediction_thr)
        result_validation['AUC_ROC'][r] = roc_auc_score(y_trut,prediction_thr)
        fpr, tpr, thresholds = precision_recall_curve(y_trut,prediction_thr)
        try:
            result_validation['AUC_PR'][r] = auc(fpr, tpr)
        except:
            result_validation['AUC_PR'][r] = None
        result_validation['precision_score'][r] = precision_score(y_trut,prediction_thr)
        result_validation['recall_score'][r] = recall_score(y_trut,prediction_thr)
        result_validation['threshold'][r] = st['best_threshold']
        result_validation['EarlyStopping'][r] = 50
        result_validation['balanced_accuracy_score'] = balanced_accuracy_score(y_trut,prediction_thr)
        result_validation["total_neurons"][r]  = total_neurons
        result_validation["total_layers"][r]  = total_layers
        df_resultsv = pd.DataFrame(result_validation)

        ### save results
        df_resultsv.to_csv(str("STL/experiments_stl/SingleNetVal_"+str(diagnosis)+".csv"))

        ## test loop
        y_pred = []
        
        with torch.no_grad():
            # Plot the ROC curve
            for x,y in ds_test:
                x=x.to(device)
                output = net(x.unsqueeze(1))
                y_pred.append(output.item() )
        
        ## statistics        
        prediction_thr = [1 if i > st['best_threshold'] else 0 for i in y_pred]
        result['target'][r] = diagnosis
        result['data'][r] = 'test'
        result['architecture'][r] = net.__class__.__name__
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
        result['threshold'][r] = st['best_threshold']
        result['EarlyStopping'][r] = 50
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