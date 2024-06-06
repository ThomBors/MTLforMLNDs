#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import torch.nn.functional as F

################# cmpute BCELoss loss foe each targhet #################
def BinaryCrossEntropy_CN(output,target,device):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """
    target_AD = target['AD'].to(device)
    target_PD = target['PD'].to(device)
    target_MCI = target['MCI'].to(device)
    target_LMCI = target['LMCI'].to(device)
    target_EMCI = target['EMCI'].to(device)
    target_FTD = target['FTD'].to(device)
    target_CN = target['CN'].to(device)

    loss_function = nn.BCELoss()

    loss_ad = loss_function(output['AD'].squeeze(1),target_AD.float())
    loss_FTD = loss_function(output['FTD'].squeeze(1),target_FTD.float())
    loss_LMCI = loss_function(output['LMCI'].squeeze(1),target_LMCI.float())
    loss_EMCI = loss_function(output['EMCI'].squeeze(1),target_EMCI.float())
    loss_MCI = loss_function(output['MCI'].squeeze(1),target_MCI.float())
    loss_pd = loss_function(output['PD'].squeeze(1),target_PD.float())
    loss_CN = loss_function(output['CN'].squeeze(1),target_CN.float())

    return loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd,loss_CN

def BinaryCrossEntropy(output,target,device):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """
    target_AD = target['AD'].to(device)
    target_PD = target['PD'].to(device)
    target_MCI = target['MCI'].to(device)
    target_LMCI = target['LMCI'].to(device)
    target_EMCI = target['EMCI'].to(device)
    target_FTD = target['FTD'].to(device)

    loss_function_ad = nn.BCELoss()
    loss_function_FTD = nn.BCELoss()
    loss_function_LMCI = nn.BCELoss()
    loss_function_EMCI = nn.BCELoss()
    loss_function_MCI = nn.BCELoss()
    loss_function_pd = nn.BCELoss()


    loss_ad = loss_function_ad(output['AD'].squeeze(1),target_AD.float())
    loss_FTD = loss_function_FTD(output['FTD'].squeeze(1),target_FTD.float())
    loss_LMCI = loss_function_LMCI(output['LMCI'].squeeze(1),target_LMCI.float())
    loss_EMCI = loss_function_EMCI(output['EMCI'].squeeze(1),target_EMCI.float())
    loss_MCI = loss_function_MCI(output['MCI'].squeeze(1),target_MCI.float())
    loss_pd = loss_function_pd(output['PD'].squeeze(1),target_PD.float())

    return loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd

def BinaryCrossEntropy_FL(output,target,device):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """
    target_AD = target['AD'].to(device)
    target_PD = target['PD'].to(device)
    target_MCI = target['MCI'].to(device)
    target_LMCI = target['LMCI'].to(device)
    target_EMCI = target['EMCI'].to(device)
    target_FTD = target['FTD'].to(device)

    
      # You can adjust this value if needed
    loss_fn = torch.nn.BCELoss(reduction='none')

    
    loss_ad =  focal_loss(loss_fn(output['AD'].squeeze(1), target_AD.float()),output['AD'].squeeze(1), target_AD.float())
    loss_FTD = focal_loss(loss_fn(output['FTD'].squeeze(1), target_FTD.float()) , output['FTD'].squeeze(1), target_FTD.float())
    loss_LMCI = focal_loss(loss_fn(output['LMCI'].squeeze(1), target_LMCI.float()) , output['LMCI'].squeeze(1), target_LMCI.float())
    loss_EMCI = focal_loss(loss_fn(output['EMCI'].squeeze(1), target_EMCI.float()) , output['EMCI'].squeeze(1), target_EMCI.float())
    loss_MCI = focal_loss(loss_fn(output['MCI'].squeeze(1), target_MCI.float()) , output['MCI'].squeeze(1), target_MCI.float())
    loss_PD = focal_loss(loss_fn(output['PD'].squeeze(1), target_PD.float()) , output['PD'].squeeze(1), target_PD.float())


    return loss_ad.mean(),loss_FTD.mean(),loss_LMCI.mean(),loss_EMCI.mean(),loss_MCI.mean(),loss_PD.mean()

def BinaryCrossEntropyCN_FL(output,target,device):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """
    target_AD = target['AD'].to(device)
    target_PD = target['PD'].to(device)
    target_MCI = target['MCI'].to(device)
    target_LMCI = target['LMCI'].to(device)
    target_EMCI = target['EMCI'].to(device)
    target_FTD = target['FTD'].to(device)
    target_CN = target['CN'].to(device)
    
      # You can adjust this value if needed
    loss_fn = torch.nn.BCELoss(reduction='none')

    
    loss_ad =  focal_loss(loss_fn(output['AD'].squeeze(1), target_AD.float()),output['AD'].squeeze(1), target_AD.float())
    loss_FTD = focal_loss(loss_fn(output['FTD'].squeeze(1), target_FTD.float()) , output['FTD'].squeeze(1), target_FTD.float())
    loss_LMCI = focal_loss(loss_fn(output['LMCI'].squeeze(1), target_LMCI.float()) , output['LMCI'].squeeze(1), target_LMCI.float())
    loss_EMCI = focal_loss(loss_fn(output['EMCI'].squeeze(1), target_EMCI.float()) , output['EMCI'].squeeze(1), target_EMCI.float())
    loss_MCI = focal_loss(loss_fn(output['MCI'].squeeze(1), target_MCI.float()) , output['MCI'].squeeze(1), target_MCI.float())
    loss_PD = focal_loss(loss_fn(output['PD'].squeeze(1), target_PD.float()) , output['PD'].squeeze(1), target_PD.float())
    loss_CN = focal_loss(loss_fn(output['CN'].squeeze(1), target_CN.float()) , output['CN'].squeeze(1), target_CN.float())


    return loss_ad.mean(),loss_FTD.mean(),loss_LMCI.mean(),loss_EMCI.mean(),loss_MCI.mean(),loss_PD.mean(),loss_CN.mean()


def focal_loss(ce_loss,prediction, target,gamma = 2,alpha = -1):
    pt = prediction * target + (1 - prediction) * (1 - target)
    loss = ce_loss * ((1 - pt) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return  loss 

class UniformWeighting(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self):
        super(UniformWeighting, self).__init__()


    def forward(self, *x):
        final_loss = 0
        for  loss in x:
            final_loss += loss

        return final_loss

class WeightingLoss(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self,weight_AD=1.25,weight_PD=1.75,weight_FTD=1.75,weight_MCI=.25,weight_EMCI=.25,weight_LMCI=.25,weight_CN=2.5):
        super(WeightingLoss, self).__init__()
        self.weight_AD = weight_AD
        self.weight_FTD = weight_FTD
        self.weight_PD = weight_PD
        self.weight_MCI = weight_MCI
        self.weight_LMCI = weight_LMCI
        self.weight_EMCI = weight_EMCI
        self.weight_CN = weight_CN

    def forward(self, loss_ad,loss_FTD,loss_LMCI,loss_EMCI,loss_MCI,loss_pd):
        


        final_loss = loss_ad*self.weight_AD+loss_FTD*self.weight_FTD+loss_LMCI*self.weight_LMCI+loss_EMCI*self.weight_EMCI+loss_MCI*self.weight_MCI+loss_pd*self.weight_PD

        return final_loss

class  MeanLosses(Module):
    """
    Mean of Losses
    """

    def __init__(self):
        super(MeanLosses, self).__init__()


    def forward(self, *x):
        final_loss = 0  # Initialize final_loss to 1
        for loss in x:
            final_loss += loss
        return final_loss / len(x)

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=6):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

if __name__ == '__main__':
    awl = AutomaticWeightedLoss(6)
    print(awl.parameters())
