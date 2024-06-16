from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm as tq
import pandas as pd
import numpy as np
import logging
import wandb


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import BinaryAUROC,BinaryAUPRC,BinaryF1Score

from loss_function import BinaryCrossEntropy,BinaryCrossEntropy_CN,BinaryCrossEntropy_FL,focal_loss,BinaryCrossEntropyCN_FL
import utils as utils

############################### traning istruction ###############################
def train(model, dl_train,loss_fn,optimizer, scheduler,epoch,online, device='cpu'):
    model.train()
    rersults_AUC_ROC = 0
    rersults_AUC_PR = 0
    
    for batch_idx, (data) in enumerate(dl_train):
        training_data = data['subject'].to(device)

        # first we need to zero the gradient, otherwise PyTorch would accumulate them
        optimizer.zero_grad() 

        ##### training #####
        output = model(training_data)

        ##### loss with CN #####
        if len(output.keys()) == 7 :
            loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN = BinaryCrossEntropy_CN(output,data,device)

            if online:
                wandb.log({"Loss_AD" : loss_ad,
                            "Loss_FTD": loss_FTD,
                            "Loss_LMCI" : loss_LMCI,
                            "Loss_EMCI" : loss_EMCI,
                            "Loss_MCI" : loss_MCI,
                            "Loss_PD" : loss_pd,
                            "loss_CN" : loss_CN})

            loss = loss_fn(loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN)

        ##### loss without CN #####
        else:
            loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd = BinaryCrossEntropy(output,data,device)

            if online:
                wandb.log({"Loss_AD" : loss_ad,
                            "Loss_FTD": loss_FTD,
                            "Loss_LMCI" : loss_LMCI,
                            "Loss_EMCI" : loss_EMCI,
                            "Loss_MCI" : loss_MCI,
                            "Loss_PD" : loss_pd})

            loss = loss_fn(loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd)

        loss.backward()
        optimizer.step()
        ####################
        
        result_metrics = utils.stat(output,data,device)
        
        rersults_AUC_ROC += result_metrics['rersults_AUC_ROC']
        rersults_AUC_PR += result_metrics['rersults_AUC_PR']

        if batch_idx == 0 :
            train_result_metrics = result_metrics
        else:
            for i in train_result_metrics.keys():
                train_result_metrics[i] += result_metrics[i]
                
    if scheduler:
        scheduler.step()

    for i in train_result_metrics.keys():
        train_result_metrics[i] = train_result_metrics[i]/(batch_idx+1)

    if online:
        wandb.log(train_result_metrics)
        wandb.log({'Train Epoch' : epoch,
                                    'Loss' : loss.item(),
                                    'Lr' : scheduler.optimizer.param_groups[0]['lr'],
                                    'Train AUC_ROC': rersults_AUC_ROC/(batch_idx+1),
                                    'Train AUC_PR': rersults_AUC_PR/(batch_idx+1)})
            
    #if epoch % 10 == 0:
    #    print('#'*100)
    #    print('-'*45+f' Epoch:{epoch} '+'-'*45)
    #    print(f"Training set: Average AUC ROC: {round(rersults_AUC_ROC/(batch_idx+1),2)}\
    #        Average AUC PR: {round(rersults_AUC_PR/(batch_idx+1),2)}")
    return loss.item()

############################### testing istruction ###############################
def f(mydict):
    return dict((k+"_test",f(v) if hasattr(v,'keys') else v) for k,v in mydict.items())

def test(model, dl_test,epoch,loss_fn,online, device='cpu'):
    model.eval()

    test_loss = 0
    rersults_AUC_PR = 0
    rersults_AUC_ROC = 0
    
    test_loss_AD = 0
    test_loss_PD = 0
    test_loss_MCI = 0
    test_loss_LMCI = 0
    test_loss_EMCI = 0
    test_loss_FTD = 0
    test_loss_CN = 0

    with torch.no_grad():
        for batch_idx, (data) in enumerate(dl_test):
            training_data = data['subject'].to(device)
            target_AD = data['AD'].to(device)
            target_PD = data['PD'].to(device)
            target_MCI = data['MCI'].to(device)
            target_LMCI = data['LMCI'].to(device)
            target_EMCI = data['EMCI'].to(device)
            target_FTD = data['FTD'].to(device)
            target_CN = data['CN'].to(device)

            output = model(training_data)
            
            test_loss_AD = F.binary_cross_entropy(output['AD'].squeeze(1), target_AD.float()).item()  # sum up batch loss
            test_loss_PD = F.binary_cross_entropy(output['PD'].squeeze(1), target_PD.float()).item()  # sum up batch loss
            test_loss_MCI = F.binary_cross_entropy(output['MCI'].squeeze(1), target_MCI.float()).item()  # sum up batch loss
            test_loss_LMCI = F.binary_cross_entropy(output['LMCI'].squeeze(1), target_LMCI.float()).item()  # sum up batch loss
            test_loss_EMCI = F.binary_cross_entropy(output['EMCI'].squeeze(1), target_EMCI.float()).item()  # sum up batch loss
            test_loss_FTD = F.binary_cross_entropy(output['FTD'].squeeze(1), target_FTD.float()).item()  # sum up batch loss
            
            if len(output.keys()) == 7 :
                test_loss_CN = F.binary_cross_entropy(output['CN'].squeeze(1), target_CN.float()).item()  # sum up batch loss

                if online:
                    wandb.log({"test_Loss_AD" : test_loss_AD,
                        "test_Loss_FTD": test_loss_FTD,
                        "test_Loss_LMCI" : test_loss_LMCI,
                        "test_Loss_EMCI" : test_loss_EMCI,
                        "test_Loss_MCI" : test_loss_MCI,
                        "test_Loss_PD" : test_loss_PD,
                        "test_Loss_CN" : test_loss_CN})
                
                test_loss = loss_fn(test_loss_AD,test_loss_FTD,test_loss_LMCI,test_loss_EMCI,test_loss_MCI,test_loss_PD,test_loss_CN)
                test_loss_dict = {"AD" : test_loss_AD,"FTD": test_loss_FTD,"LMCI" : test_loss_LMCI,"EMCI" : test_loss_EMCI,"MCI" : test_loss_MCI,"PD" : test_loss_PD, "CN" :test_loss_CN}
            
            else:
                if online:
                    wandb.log({"test_Loss_AD" : test_loss_AD,
                        "test_Loss_FTD": test_loss_FTD,
                        "test_Loss_LMCI" : test_loss_LMCI,
                        "test_Loss_EMCI" : test_loss_EMCI,
                        "test_Loss_MCI" : test_loss_MCI,
                        "test_Loss_PD" : test_loss_PD})
                
                test_loss = loss_fn(test_loss_AD,test_loss_FTD,test_loss_LMCI,test_loss_EMCI,test_loss_MCI,test_loss_PD)
                test_loss_dict = {"AD" : test_loss_AD,"FTD": test_loss_FTD,"LMCI" : test_loss_LMCI,"EMCI" : test_loss_EMCI,"MCI" : test_loss_MCI,"PD" : test_loss_PD}
            
            result_metrics = utils.stat(output,data,device)

            if batch_idx == 0 :
                test_result_metrics = f(result_metrics)
            else:
                for i in test_result_metrics.keys():
                    test_result_metrics[i] += f(result_metrics)[i]
        
            rersults_AUC_ROC += result_metrics['rersults_AUC_ROC']
            rersults_AUC_PR += result_metrics['rersults_AUC_PR']

        
        for i in test_result_metrics.keys():
            test_result_metrics[i] = test_result_metrics[i]/(batch_idx+1)

        if online:
            wandb.log(test_result_metrics)

            wandb.log({'Test Epoch' : epoch,
                        'test_loss' : test_loss ,
                        'Test AUC_ROC': rersults_AUC_ROC/(batch_idx+1),
                        'Test AUC_PR': rersults_AUC_PR/(batch_idx+1)})
        
    #if epoch % 10 == 0:
    #    print('-'*100)
    #    print(f"Test set:     Average AUC ROC: {round(rersults_AUC_ROC/(batch_idx+1),2)}\
    #        Average AUC PR: {round(rersults_AUC_PR/(batch_idx+1),2)}")
    return test_loss,test_loss_dict
    



############################### traning istruction pcgrad ###############################
def train_FL(model, dl_train,loss_fn,optimizer, scheduler,epoch,online, device='cpu'):
    model.train()
    rersults_AUC_ROC = 0
    rersults_AUC_PR = 0
    
    for batch_idx, (data) in enumerate(dl_train):
        training_data = data['subject'].to(device)

        # first we need to zero the gradient, otherwise PyTorch would accumulate them
        optimizer.zero_grad() 

        ##### training #####
        output = model(training_data)

        ##### loss with CN #####
        if len(output.keys()) == 7 :
            loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN = BinaryCrossEntropyCN_FL(output,data,device)

            if online:
                wandb.log({"Loss_AD" : loss_ad,
                            "Loss_FTD": loss_FTD,
                            "Loss_LMCI" : loss_LMCI,
                            "Loss_EMCI" : loss_EMCI,
                            "Loss_MCI" : loss_MCI,
                            "Loss_PD" : loss_pd,
                            "loss_CN" : loss_CN})

            loss = loss_fn(loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN)

        ##### loss without CN #####
        else:
            loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd = BinaryCrossEntropy_FL(output,data,device)

            if online:
                wandb.log({"Loss_AD" : loss_ad,
                            "Loss_FTD": loss_FTD,
                            "Loss_LMCI" : loss_LMCI,
                            "Loss_EMCI" : loss_EMCI,
                            "Loss_MCI" : loss_MCI,
                            "Loss_PD" : loss_pd})

            loss = loss_fn(loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd)

        loss.backward()
        optimizer.step()
        ####################
        
        result_metrics = utils.stat(output,data,device)
        
        rersults_AUC_ROC += result_metrics['rersults_AUC_ROC']
        rersults_AUC_PR += result_metrics['rersults_AUC_PR']

        if batch_idx == 0 :
            train_result_metrics = result_metrics
        else:
            for i in train_result_metrics.keys():
                train_result_metrics[i] += result_metrics[i]
                
    if scheduler:
        scheduler.step()

    for i in train_result_metrics.keys():
        train_result_metrics[i] = train_result_metrics[i]/(batch_idx+1)

    if online:
        wandb.log(train_result_metrics)
        wandb.log({'Train Epoch' : epoch,
                                    'Loss' : loss.item(),
                                    'Lr' : scheduler.optimizer.param_groups[0]['lr'],
                                    'Train AUC_ROC': rersults_AUC_ROC/(batch_idx+1),
                                    'Train AUC_PR': rersults_AUC_PR/(batch_idx+1)})
            
    #if epoch % 10 == 0:
    #    print('#'*100)
    #    print('-'*45+f' Epoch:{epoch} '+'-'*45)
    #    print(f"Training set: Average AUC ROC: {round(rersults_AUC_ROC/(batch_idx+1),2)}\
    #        Average AUC PR: {round(rersults_AUC_PR/(batch_idx+1),2)}")
    return loss.item()

############################### testing istruction ###############################
def f(mydict):
    return dict((k+"_test",f(v) if hasattr(v,'keys') else v) for k,v in mydict.items())

def test_FL(model, dl_test,epoch,loss_fn,online, device='cpu'):
    model.eval()

    test_loss = 0
    rersults_AUC_PR = 0
    rersults_AUC_ROC = 0
    
    test_loss_AD = 0
    test_loss_PD = 0
    test_loss_MCI = 0
    test_loss_LMCI = 0
    test_loss_EMCI = 0
    test_loss_FTD = 0
    test_loss_CN = 0

    with torch.no_grad():
        for batch_idx, (data) in enumerate(dl_test):
            training_data = data['subject'].to(device)
            target_AD = data['AD'].to(device)
            target_PD = data['PD'].to(device)
            target_MCI = data['MCI'].to(device)
            target_LMCI = data['LMCI'].to(device)
            target_EMCI = data['EMCI'].to(device)
            target_FTD = data['FTD'].to(device)
            target_CN = data['CN'].to(device)

            output = model(training_data)
            test_loss_AD = focal_loss(F.binary_cross_entropy(output['AD'].squeeze(1), target_AD.float(),reduction='none'),output['AD'].squeeze(1),target_AD.float()).mean().item()  # sum up batch loss
            test_loss_PD = focal_loss(F.binary_cross_entropy(output['PD'].squeeze(1), target_PD.float(),reduction='none'),output['PD'].squeeze(1),target_PD.float()).mean().item()  # sum up batch loss
            test_loss_MCI = focal_loss(F.binary_cross_entropy(output['MCI'].squeeze(1), target_MCI.float(),reduction='none'),output['MCI'].squeeze(1),target_MCI.float()).mean().item()  # sum up batch loss
            test_loss_LMCI = focal_loss(F.binary_cross_entropy(output['LMCI'].squeeze(1), target_LMCI.float(),reduction='none'),output['LMCI'].squeeze(1),target_EMCI.float()).mean().item()  # sum up batch loss
            test_loss_EMCI = focal_loss(F.binary_cross_entropy(output['EMCI'].squeeze(1), target_EMCI.float(),reduction='none'),output['EMCI'].squeeze(1),target_LMCI.float()).mean().item()  # sum up batch loss
            test_loss_FTD = focal_loss(F.binary_cross_entropy(output['FTD'].squeeze(1), target_FTD.float(),reduction='none'),output['FTD'].squeeze(1),target_FTD.float()).mean().item()  # sum up batch loss

            if len(output.keys()) == 7 :
                test_loss_CN = F.binary_cross_entropy(output['CN'].squeeze(1), target_CN.float()).item()  # sum up batch loss

                if online:
                    wandb.log({"test_Loss_AD" : test_loss_AD,
                        "test_Loss_FTD": test_loss_FTD,
                        "test_Loss_LMCI" : test_loss_LMCI,
                        "test_Loss_EMCI" : test_loss_EMCI,
                        "test_Loss_MCI" : test_loss_MCI,
                        "test_Loss_PD" : test_loss_PD,
                        "test_Loss_CN" : test_loss_CN})
                
                test_loss = loss_fn(test_loss_AD,test_loss_FTD,test_loss_LMCI,test_loss_EMCI,test_loss_MCI,test_loss_PD,test_loss_CN)
                test_loss_dict = {"AD" : test_loss_AD,"FTD": test_loss_FTD,"LMCI" : test_loss_LMCI,"EMCI" : test_loss_EMCI,"MCI" : test_loss_MCI,"PD" : test_loss_PD, "CN" :test_loss_CN}
            
            else:
                if online:
                    wandb.log({"test_Loss_AD" : test_loss_AD,
                        "test_Loss_FTD": test_loss_FTD,
                        "test_Loss_LMCI" : test_loss_LMCI,
                        "test_Loss_EMCI" : test_loss_EMCI,
                        "test_Loss_MCI" : test_loss_MCI,
                        "test_Loss_PD" : test_loss_PD})
                
                test_loss = loss_fn(test_loss_AD,test_loss_FTD,test_loss_LMCI,test_loss_EMCI,test_loss_MCI,test_loss_PD)
                test_loss_dict = {"AD" : test_loss_AD,"FTD": test_loss_FTD,"LMCI" : test_loss_LMCI,"EMCI" : test_loss_EMCI,"MCI" : test_loss_MCI,"PD" : test_loss_PD}
            
            
            result_metrics = utils.stat(output,data,device)

            if batch_idx == 0 :
                test_result_metrics = f(result_metrics)
            else:
                for i in test_result_metrics.keys():
                    test_result_metrics[i] += f(result_metrics)[i]
        
            rersults_AUC_ROC += result_metrics['rersults_AUC_ROC']
            rersults_AUC_PR += result_metrics['rersults_AUC_PR']

        
        for i in test_result_metrics.keys():
            test_result_metrics[i] = test_result_metrics[i]/(batch_idx+1)

        if online:
            wandb.log(test_result_metrics)

            wandb.log({'Test Epoch' : epoch,
                        'test_loss' : test_loss ,
                        'Test AUC_ROC': rersults_AUC_ROC/(batch_idx+1),
                        'Test AUC_PR': rersults_AUC_PR/(batch_idx+1)})
        
    #if epoch % 10 == 0:
    #    print('-'*100)
    #    print(f"Test set:     Average AUC ROC: {round(rersults_AUC_ROC/(batch_idx+1),2)}\
    #        Average AUC PR: {round(rersults_AUC_PR/(batch_idx+1),2)}")
    return test_loss,test_loss_dict

############################### traning istruction FAMO ###############################
def train_FAMO(model, dl_train,loss_fn,optimizer,weight_opt, scheduler,epoch,online, device):
    model.train()
    rersults_AUC_ROC = 0
    rersults_AUC_PR = 0
    
    for batch_idx, (data) in enumerate(dl_train):
        training_data = data['subject'].to(device)


        ##### training #####
        output = model(training_data)
        optimizer.zero_grad()
        
        ##### loss with CN #####
        if len(output.keys()) == 7 :
            loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN = BinaryCrossEntropy_CN(output,data,device)

            if online==True:
                wandb.log({"Loss_AD" : loss_ad,
                            "Loss_FTD": loss_FTD,
                            "Loss_LMCI" : loss_LMCI,
                            "Loss_EMCI" : loss_EMCI,
                            "Loss_MCI" : loss_MCI,
                            "Loss_PD" : loss_pd,
                            "loss_CN" : loss_CN})

            loss_avg = loss_fn(loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN)
            losses = torch.stack((loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN))

        ##### loss without CN #####
        else:
            loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd = BinaryCrossEntropy(output,data,device)

            if online==True:
                wandb.log({"Loss_AD" : loss_ad,
                            "Loss_FTD": loss_FTD,
                            "Loss_LMCI" : loss_LMCI,
                            "Loss_EMCI" : loss_EMCI,
                            "Loss_MCI" : loss_MCI,
                            "Loss_PD" : loss_pd})

            loss_avg = loss_fn(loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd)
            losses = torch.stack((loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd))
        
        ##### backward #####
        loss = weight_opt.backward(losses=losses, 
                            # shared_parameters = list(model.shared)
                            )
        optimizer.step()

        ##### update FAMO weights #####
        with torch.no_grad():
            new_output = model(training_data)
            if len(output.keys()) == 7 :
                loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN = BinaryCrossEntropy_CN(new_output,data,device)
                new_losses = torch.stack((loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN))
            else:
                loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd = BinaryCrossEntropy(new_output,data,device)

                new_losses = torch.stack((loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd))
            weight_opt.update(new_losses.detach())
        
        result_metrics = utils.stat(output,data,device)
        
        rersults_AUC_ROC += result_metrics['rersults_AUC_ROC']
        rersults_AUC_PR += result_metrics['rersults_AUC_PR']

        if batch_idx == 0 :
            train_result_metrics = result_metrics
        else:
            for i in train_result_metrics.keys():
                train_result_metrics[i] += result_metrics[i]
                
    if scheduler:
        scheduler.step()

    for i in train_result_metrics.keys():
        train_result_metrics[i] = train_result_metrics[i]/(batch_idx+1)

    if online==True:
        wandb.log(train_result_metrics)
        wandb.log({'Train Epoch' : epoch,
                                    'Loss' : loss_avg.item(),
                                    'Lr' : scheduler.optimizer.param_groups[0]['lr'],
                                    'Train AUC_ROC': rersults_AUC_ROC/(batch_idx+1),
                                    'Train AUC_PR': rersults_AUC_PR/(batch_idx+1)})
    else:
        if epoch % 100 == 0:
            print('#'*100)
            print('-'*45+f' Epoch:{epoch} '+'-'*45)
            print('-- Training set:')
            print(f"Loss: {loss_avg.item()}, Lr: {scheduler.optimizer.param_groups[0]['lr']}")
            print(f"Average AUC ROC: {round(rersults_AUC_ROC/(batch_idx+1),2)}\
                Average AUC PR: {round(rersults_AUC_PR/(batch_idx+1),2)}")
    return loss_avg.item()

############################### testing istruction PCGrad ###############################
def f(mydict):
    return dict((k+"_test",f(v) if hasattr(v,'keys') else v) for k,v in mydict.items())

def test_FAMO(model, dl_test,epoch,loss_fn,weight_opt,online, device):
    model.eval()

    test_loss = 0
    rersults_AUC_PR = 0
    rersults_AUC_ROC = 0
    
    test_loss_AD = 0
    test_loss_PD = 0
    test_loss_MCI = 0
    test_loss_LMCI = 0
    test_loss_EMCI = 0
    test_loss_FTD = 0
    test_loss_CN = 0

    with torch.no_grad():
        for batch_idx, (data) in enumerate(dl_test):
            training_data = data['subject'].to(device)
            target_AD = data['AD'].to(device)
            target_PD = data['PD'].to(device)
            target_MCI = data['MCI'].to(device)
            target_LMCI = data['LMCI'].to(device)
            target_EMCI = data['EMCI'].to(device)
            target_FTD = data['FTD'].to(device)
            target_CN = data['CN'].to(device)

            output = model(training_data)
            
            test_loss_AD = F.binary_cross_entropy(output['AD'].squeeze(1), target_AD.float()).item()  # sum up batch loss
            test_loss_PD = F.binary_cross_entropy(output['PD'].squeeze(1), target_PD.float()).item()  # sum up batch loss
            test_loss_MCI = F.binary_cross_entropy(output['MCI'].squeeze(1), target_MCI.float()).item()  # sum up batch loss
            test_loss_LMCI = F.binary_cross_entropy(output['LMCI'].squeeze(1), target_LMCI.float()).item()  # sum up batch loss
            test_loss_EMCI = F.binary_cross_entropy(output['EMCI'].squeeze(1), target_EMCI.float()).item()  # sum up batch loss
            test_loss_FTD = F.binary_cross_entropy(output['FTD'].squeeze(1), target_FTD.float()).item()  # sum up batch loss
            if len(output.keys()) == 7 :
                test_loss_CN = F.binary_cross_entropy(output['CN'].squeeze(1), target_CN.float()).item()  # sum up batch loss

                if online==True:
                    wandb.log({"test_Loss_AD" : test_loss_AD,
                        "test_Loss_FTD": test_loss_FTD,
                        "test_Loss_LMCI" : test_loss_LMCI,
                        "test_Loss_EMCI" : test_loss_EMCI,
                        "test_Loss_MCI" : test_loss_MCI,
                        "test_Loss_PD" : test_loss_PD,
                        "test_Loss_CN" : test_loss_CN})
                
                test_loss = loss_fn(test_loss_AD,test_loss_FTD,test_loss_LMCI,test_loss_EMCI,test_loss_MCI,test_loss_PD,test_loss_CN)
                test_loss_dict = {"AD" : test_loss_AD,"FTD": test_loss_FTD,"LMCI" : test_loss_LMCI,"EMCI" : test_loss_EMCI,"MCI" : test_loss_MCI,"PD" : test_loss_PD, "CN" :test_loss_CN}
            
            else:
                if online==True:
                    wandb.log({"test_Loss_AD" : test_loss_AD,
                        "test_Loss_FTD": test_loss_FTD,
                        "test_Loss_LMCI" : test_loss_LMCI,
                        "test_Loss_EMCI" : test_loss_EMCI,
                        "test_Loss_MCI" : test_loss_MCI,
                        "test_Loss_PD" : test_loss_PD})
                
                test_loss = loss_fn(test_loss_AD,test_loss_FTD,test_loss_LMCI,test_loss_EMCI,test_loss_MCI,test_loss_PD)
                test_loss_dict = {"AD" : test_loss_AD,"FTD": test_loss_FTD,"LMCI" : test_loss_LMCI,"EMCI" : test_loss_EMCI,"MCI" : test_loss_MCI,"PD" : test_loss_PD}
            
            result_metrics = utils.stat(output,data,device)

            if batch_idx == 0 :
                test_result_metrics = f(result_metrics)
            else:
                for i in test_result_metrics.keys():
                    test_result_metrics[i] += f(result_metrics)[i]
        
            rersults_AUC_ROC += result_metrics['rersults_AUC_ROC']
            rersults_AUC_PR += result_metrics['rersults_AUC_PR']

        
        for i in test_result_metrics.keys():
            test_result_metrics[i] = test_result_metrics[i]/(batch_idx+1)

        if online==True:
            wandb.log(test_result_metrics)

            wandb.log({'Test Epoch' : epoch,
                        'test_loss' : test_loss ,
                        'Test AUC_ROC': rersults_AUC_ROC/(batch_idx+1),
                        'Test AUC_PR': rersults_AUC_PR/(batch_idx+1)})
        else:
            if epoch % 100 == 0:
                print('-'*100)
                print('-- Validation ste:')
                print(f"Loss: {test_loss}")
                print(f"Average AUC ROC: {round(rersults_AUC_ROC/(batch_idx+1),2)}\
                    Average AUC PR: {round(rersults_AUC_PR/(batch_idx+1),2)}")
    return test_loss,test_loss_dict


############################### traning istruction Focal loss and FAMO ###############################
def train_FLAMO(model, dl_train,loss_fn,optimizer,weight_opt, scheduler,epoch,online, device='cpu'):
    model.train()
    rersults_AUC_ROC = 0
    rersults_AUC_PR = 0
    
    for batch_idx, (data) in enumerate(dl_train):
        training_data = data['subject'].to(device)

        # first we need to zero the gradient, otherwise PyTorch would accumulate them
        optimizer.zero_grad() 

        ##### training #####
        output = model(training_data)

        ##### loss with CN #####
        if len(output.keys()) == 7 :
            loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN = BinaryCrossEntropyCN_FL(output,data,device)

            if online:
                wandb.log({"Loss_AD" : loss_ad,
                            "Loss_FTD": loss_FTD,
                            "Loss_LMCI" : loss_LMCI,
                            "Loss_EMCI" : loss_EMCI,
                            "Loss_MCI" : loss_MCI,
                            "Loss_PD" : loss_pd,
                            "loss_CN" : loss_CN})

            loss_f = loss_fn(loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN)
            losses = torch.stack((loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN))

        ##### loss without CN #####
        else:
            loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd = BinaryCrossEntropy_FL(output,data,device)

            if online:
                wandb.log({"Loss_AD" : loss_ad,
                            "Loss_FTD": loss_FTD,
                            "Loss_LMCI" : loss_LMCI,
                            "Loss_EMCI" : loss_EMCI,
                            "Loss_MCI" : loss_MCI,
                            "Loss_PD" : loss_pd})

            loss_f = loss_fn(loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd)
            losses = torch.stack((loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd))
        
        ##### backward #####
        loss = weight_opt.backward(losses=losses, 
                            # shared_parameters = list(model.shared)
                            )
        optimizer.step()

        ##### update FAMO weights #####
        with torch.no_grad():
            new_output = model(training_data)
            if len(output.keys()) == 7 :
                loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN = BinaryCrossEntropy_CN(new_output,data,device)
                new_losses = torch.stack((loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN))
            else:
                loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd = BinaryCrossEntropy(new_output,data,device)

                new_losses = torch.stack((loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd))
            weight_opt.update(new_losses.detach())
        ####################
        
        result_metrics = utils.stat(output,data,device)
        
        rersults_AUC_ROC += result_metrics['rersults_AUC_ROC']
        rersults_AUC_PR += result_metrics['rersults_AUC_PR']

        if batch_idx == 0 :
            train_result_metrics = result_metrics
        else:
            for i in train_result_metrics.keys():
                train_result_metrics[i] += result_metrics[i]
                
    if scheduler:
        scheduler.step()

    for i in train_result_metrics.keys():
        train_result_metrics[i] = train_result_metrics[i]/(batch_idx+1)

    if online:
        wandb.log(train_result_metrics)
        wandb.log({'Train Epoch' : epoch,
                                    'Loss' : loss_f.item(),
                                    'Lr' : scheduler.optimizer.param_groups[0]['lr'],
                                    'Train AUC_ROC': rersults_AUC_ROC/(batch_idx+1),
                                    'Train AUC_PR': rersults_AUC_PR/(batch_idx+1)})
            
    #if epoch % 10 == 0:
    #    print('#'*100)
    #    print('-'*45+f' Epoch:{epoch} '+'-'*45)
    #    print(f"Training set: Average AUC ROC: {round(rersults_AUC_ROC/(batch_idx+1),2)}\
    #        Average AUC PR: {round(rersults_AUC_PR/(batch_idx+1),2)}")
    return loss_f.item()

############################### testing istruction ###############################
def f(mydict):
    return dict((k+"_test",f(v) if hasattr(v,'keys') else v) for k,v in mydict.items())

def test_FLAMO(model, dl_test,epoch,loss_fn,weight_opt,online, device='cpu'):
    model.eval()

    test_loss = 0
    rersults_AUC_PR = 0
    rersults_AUC_ROC = 0
    
    test_loss_AD = 0
    test_loss_PD = 0
    test_loss_MCI = 0
    test_loss_LMCI = 0
    test_loss_EMCI = 0
    test_loss_FTD = 0
    test_loss_CN = 0

    with torch.no_grad():
        for batch_idx, (data) in enumerate(dl_test):
            training_data = data['subject'].to(device)
            target_AD = data['AD'].to(device)
            target_PD = data['PD'].to(device)
            target_MCI = data['MCI'].to(device)
            target_LMCI = data['LMCI'].to(device)
            target_EMCI = data['EMCI'].to(device)
            target_FTD = data['FTD'].to(device)
            target_CN = data['CN'].to(device)

            output = model(training_data)
            test_loss_AD = focal_loss(F.binary_cross_entropy(output['AD'].squeeze(1), target_AD.float(),reduction='none'),output['AD'].squeeze(1),target_AD.float()).mean().item()  # sum up batch loss
            test_loss_PD = focal_loss(F.binary_cross_entropy(output['PD'].squeeze(1), target_PD.float(),reduction='none'),output['PD'].squeeze(1),target_PD.float()).mean().item()  # sum up batch loss
            test_loss_MCI = focal_loss(F.binary_cross_entropy(output['MCI'].squeeze(1), target_MCI.float(),reduction='none'),output['MCI'].squeeze(1),target_MCI.float()).mean().item()  # sum up batch loss
            test_loss_LMCI = focal_loss(F.binary_cross_entropy(output['LMCI'].squeeze(1), target_LMCI.float(),reduction='none'),output['LMCI'].squeeze(1),target_EMCI.float()).mean().item()  # sum up batch loss
            test_loss_EMCI = focal_loss(F.binary_cross_entropy(output['EMCI'].squeeze(1), target_EMCI.float(),reduction='none'),output['EMCI'].squeeze(1),target_LMCI.float()).mean().item()  # sum up batch loss
            test_loss_FTD = focal_loss(F.binary_cross_entropy(output['FTD'].squeeze(1), target_FTD.float(),reduction='none'),output['FTD'].squeeze(1),target_FTD.float()).mean().item()  # sum up batch loss

            if len(output.keys()) == 7 :
                test_loss_CN = focal_loss(F.binary_cross_entropy(output['CN'].squeeze(1), target_CN.float(),reduction='none'),output['CN'].squeeze(1),target_CN.float()).mean().item()  # sum up batch loss

                if online:
                    wandb.log({"test_Loss_AD" : test_loss_AD,
                        "test_Loss_FTD": test_loss_FTD,
                        "test_Loss_LMCI" : test_loss_LMCI,
                        "test_Loss_EMCI" : test_loss_EMCI,
                        "test_Loss_MCI" : test_loss_MCI,
                        "test_Loss_PD" : test_loss_PD,
                        "test_Loss_CN" : test_loss_CN})
                
                test_loss = loss_fn(test_loss_AD,test_loss_FTD,test_loss_LMCI,test_loss_EMCI,test_loss_MCI,test_loss_PD,test_loss_CN)
                test_loss_dict = {"AD" : test_loss_AD,"FTD": test_loss_FTD,"LMCI" : test_loss_LMCI,"EMCI" : test_loss_EMCI,"MCI" : test_loss_MCI,"PD" : test_loss_PD, "CN" :test_loss_CN}
            
            else:
                if online:
                    wandb.log({"test_Loss_AD" : test_loss_AD,
                        "test_Loss_FTD": test_loss_FTD,
                        "test_Loss_LMCI" : test_loss_LMCI,
                        "test_Loss_EMCI" : test_loss_EMCI,
                        "test_Loss_MCI" : test_loss_MCI,
                        "test_Loss_PD" : test_loss_PD})
                
                test_loss = loss_fn(test_loss_AD,test_loss_FTD,test_loss_LMCI,test_loss_EMCI,test_loss_MCI,test_loss_PD)
                test_loss_dict = {"AD" : test_loss_AD,"FTD": test_loss_FTD,"LMCI" : test_loss_LMCI,"EMCI" : test_loss_EMCI,"MCI" : test_loss_MCI,"PD" : test_loss_PD}
            
            
            result_metrics = utils.stat(output,data,device)

            if batch_idx == 0 :
                test_result_metrics = f(result_metrics)
            else:
                for i in test_result_metrics.keys():
                    test_result_metrics[i] += f(result_metrics)[i]
        
            rersults_AUC_ROC += result_metrics['rersults_AUC_ROC']
            rersults_AUC_PR += result_metrics['rersults_AUC_PR']

        
        for i in test_result_metrics.keys():
            test_result_metrics[i] = test_result_metrics[i]/(batch_idx+1)

        if online:
            wandb.log(test_result_metrics)

            wandb.log({'Test Epoch' : epoch,
                        'test_loss' : test_loss ,
                        'Test AUC_ROC': rersults_AUC_ROC/(batch_idx+1),
                        'Test AUC_PR': rersults_AUC_PR/(batch_idx+1)})
        
    #if epoch % 10 == 0:
    #    print('-'*100)
    #    print(f"Test set:     Average AUC ROC: {round(rersults_AUC_ROC/(batch_idx+1),2)}\
    #        Average AUC PR: {round(rersults_AUC_PR/(batch_idx+1),2)}")
    return test_loss,test_loss_dict