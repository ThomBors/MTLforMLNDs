#!/usr/bin/env python

from sklearn.metrics import roc_curve,precision_recall_curve,auc,roc_auc_score
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  Dataset
from torcheval.metrics import BinaryAUROC,BinaryAUPRC


############################################################################################
####################### data preparation ###################################################
############################################################################################
class AD_Dataset(Dataset):
    def __init__(self, feature_data,target):
        self.feature_data = feature_data
        self.target = target

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        subject = self.feature_data.iloc[idx]
        diagnosis = self.target[idx]

        subject = np.array([subject])
        diagnosis = np.array([diagnosis])

        subject = subject.astype('float32')

        subject = torch.from_numpy(subject)
        diagnosis = torch.from_numpy(diagnosis)


        return subject, diagnosis


############################################################################################
####################### training ###########################################################
############################################################################################
def train(model, dl_train,optimizer, scheduler,epoch, device='cpu',online=True):
    model.train()
    rersults_AUC_ROC = 0
    rersults_AUC_PR = 0

    loss_fn = nn.BCELoss()
    for batch_idx, (data, target) in enumerate(dl_train):
        data, target = data.to(device), target.to(device)
        
        # first we need to zero the gradient, otherwise PyTorch would accumulate them
        optimizer.zero_grad() 
        ##### training #####
        output = model(data)
        loss = loss_fn(output.squeeze(1),target.float())
        loss.backward()
        optimizer.step()
        ####################

        # stats
        AUC_ROC = BinaryAUROC()
        AUC_ROC.update(output.squeeze(), target.squeeze())
        AUC_ROC = AUC_ROC.compute()

        AUC_PR = BinaryAUPRC()
        AUC_PR.update(output.squeeze(), target.squeeze())
        AUC_PR = AUC_PR.compute()

        rersults_AUC_ROC += AUC_ROC.item()
        rersults_AUC_PR += AUC_PR.item()

    if scheduler != None:
        scheduler.step()
    if online:
        wandb.log({'Train Epoch' : epoch,
                             'Loss' : loss.item(),
                            'LR' : scheduler.optimizer.param_groups[0]['lr'],
                             'Train AUC_ROC': rersults_AUC_ROC/(batch_idx+1),
                             'Train AUC_PR': rersults_AUC_PR/(batch_idx+1)})
        
    
    if epoch % 10 == 0:
        print(f"{'#'*50} {epoch} {'#'*50}")
        print(f"Training set: Average AUC ROC: {round(rersults_AUC_ROC/(batch_idx+1),2)}\
            Average AUC PR: {round(rersults_AUC_PR/(batch_idx+1),2)}")
    return loss.item()

############################################################################################
####################### testing ############################################################
############################################################################################
def test(model, dl_test,epoch, device='cpu',online=True):
    model.eval()
    test_loss = 0
    rersults_AUC_ROC = 0
    rersults_AUC_PR = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dl_test):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = F.binary_cross_entropy(output.squeeze(1), target.float()).item()  # sum up batch loss
            # stats
            AUC_ROC = BinaryAUROC()
            AUC_ROC.update(output.squeeze(), target.squeeze())
            AUC_ROC = AUC_ROC.compute()

            AUC_PR = BinaryAUPRC()
            AUC_PR.update(output.squeeze(), target.squeeze())
            AUC_PR = AUC_PR.compute()

            rersults_AUC_ROC += AUC_ROC.item()
            rersults_AUC_PR += AUC_PR.item()
    if online:
        wandb.log({'Test Epoch' : epoch,
               'test_loss' : test_loss,
                'Test AUC_ROC': rersults_AUC_ROC/(batch_idx+1),
                'Test AUC_PR': rersults_AUC_PR/(batch_idx+1)})
    
    if epoch % 10 == 0:
        print(f"{'-'*104}")
        print(f"Test set:     Average AUC ROC: {round(rersults_AUC_ROC/(batch_idx+1),2)}\
            Average AUC PR: {round(rersults_AUC_PR/(batch_idx+1),2)}")
    
    return test_loss


############################################################################################
####################### evaluation #########################################################
############################################################################################
def get_metric_and_best_threshold_from_roc_curve(y_test,y_pred,show_result):
    num_pos_class = sum([i ==1 for i in y_test ])
    num_neg_class = sum([i ==0 for i in y_test ])
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    precision, recall, thr = precision_recall_curve(y_test, y_pred)
    AUC_pr = auc(recall, precision)

    AUC_roc = roc_auc_score(y_test, y_pred)
    tp = tpr * num_pos_class
    fp = fpr *num_pos_class
    tn = (1 - fpr) * num_neg_class
    fn = (1 - tpr) * num_pos_class
    acc = (tp + tn) / (num_pos_class + num_neg_class)
    f1 = (2*tp) / (2*tp+fp+fn)
    best_threshold = thresholds[np.argmax(f1)]

    if show_result:
        plt.plot(thresholds[1:],f1[1:]) # ROC curve = TPR vs FPR
        plt.vlines(thresholds[np.argmax(f1)],ymin=min(f1),ymax=max(f1),colors='red',linestyles='dotted')
        plt.title("f1 in thresholds")
        plt.xlabel("Thresholds")
        plt.ylabel("f1")
        plt.show()
    print(f"best threshold: {round(best_threshold,4)} with Accuracy: {round(max(acc),6)} and F1: {round(max(f1),6)}")
    print(f"thresholds independends metrics: AUC PR: {AUC_pr}, AUC Roc: {AUC_roc}")

    return {"best_threshold":best_threshold,"f1_score":max(f1).item(),"AUC_pr":AUC_pr,"AUC_Roc":AUC_roc}

################# plot distribution of disease #################
def plot_distribution_ds(df_target):
    fg = sns.displot(df_target,stat='percent', height=3.5, aspect=1.25)

    for ax in fg.axes.ravel():
        
        # add annotations
        for c in ax.containers:

            # custom label calculates percent and add an empty string so 0 value bars don't have a number
            labels = [f'{w:0.1f}%' if (w := v.get_height()) > 0 else '' for v in c]

            ax.bar_label(c, labels=labels, label_type='edge', fontsize=8, rotation=90, padding=2)
        
        ax.margins(y=0.2)
    plt.title(f"percent of disease in training ste")
    plt.show()
