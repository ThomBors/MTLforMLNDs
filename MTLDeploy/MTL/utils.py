#!/usr/bin/env python

from sklearn.metrics import confusion_matrix, accuracy_score,f1_score,roc_curve,roc_auc_score,precision_score,recall_score,precision_recall_curve,auc,balanced_accuracy_score
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np


import torch
from torch.utils.data import Dataset
from torcheval.metrics import BinaryAUROC,BinaryAUPRC
import logging
from loss_function import UniformWeighting

################# create custum data set class #################
class AD_Dataset(Dataset):
    def __init__(self, feature_data,AD,PD,MCI,LMCI,EMCI,FTD,CN):
        self.feature_data = feature_data
        self.AD = AD
        self.PD = PD
        self.MCI = MCI
        self.LMCI = LMCI
        self.EMCI = EMCI
        self.FTD = FTD
        self.CN = CN

    def __len__(self):
        return len(self.feature_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        
        AD = self.AD[idx]
        PD = self.PD[idx]
        MCI = self.MCI[idx]
        LMCI = self.LMCI[idx]
        EMCI = self.EMCI[idx]
        FTD = self.FTD[idx]
        CN = self.CN[idx]
        subject = self.feature_data.iloc[idx]

        subject = np.array([subject])
        AD = np.array([AD])
        PD = np.array([PD])
        MCI = np.array([MCI])
        LMCI = np.array([LMCI])
        EMCI = np.array([EMCI])
        FTD = np.array([FTD])
        CN = np.array([CN])


        subject = subject.astype('float32')

        subject = torch.from_numpy(subject)
        AD = torch.from_numpy(AD)
        PD = torch.from_numpy(PD)
        MCI = torch.from_numpy(MCI)
        LMCI = torch.from_numpy(LMCI)
        EMCI = torch.from_numpy(EMCI)
        FTD = torch.from_numpy(FTD)
        CN = torch.from_numpy(CN)

        # Return the subject of the dataset
        sample = {'subject':subject, 'AD': AD, 'PD': PD, 'MCI':MCI,'LMCI':LMCI,'EMCI':EMCI,'FTD':FTD,'CN':CN}

        return sample

################# compute results statiscics #################
def stat(output,target,device):
    target_AD = target['AD'].to(device)
    target_PD = target['PD'].to(device)
    target_MCI = target['MCI'].to(device)
    target_LMCI = target['LMCI'].to(device)
    target_EMCI = target['EMCI'].to(device)
    target_FTD = target['FTD'].to(device)


    results = {}

    # stats
    AUC_ROC_AD = torch.empty((1)).to(device)
    AUC_PR_AD = torch.empty((1)).to(device)

    AUC_ROC_PD = torch.empty((1)).to(device)
    AUC_PR_PD = torch.empty((1)).to(device)

    AUC_ROC_MCI = torch.empty((1)).to(device)
    AUC_PR_MCI = torch.empty((1)).to(device)

    AUC_ROC_LMCI = torch.empty((1)).to(device)
    AUC_PR_LMCI = torch.empty((1)).to(device)

    AUC_ROC_EMCI = torch.empty((1)).to(device)
    AUC_PR_EMCI = torch.empty((1)).to(device)

    AUC_ROC_FTD = torch.empty((1)).to(device)
    AUC_PR_FTD = torch.empty((1)).to(device)


    for d in output.items():

        # ROc
        
        AUC_roc = BinaryAUROC()

        AUC_roc.update(d[1].squeeze(), torch.squeeze(target_AD))
        AUC_ROC_AD = torch.cat((AUC_ROC_AD, AUC_roc.compute().reshape(1).to(device)), dim = 0) 
        AUC_roc.update(d[1].squeeze(), torch.squeeze(target_PD))
        AUC_ROC_PD = torch.cat((AUC_ROC_PD, AUC_roc.compute().reshape(1).to(device)), dim = 0)
        AUC_roc.update(d[1].squeeze(), torch.squeeze(target_MCI))
        AUC_ROC_MCI = torch.cat((AUC_ROC_MCI, AUC_roc.compute().reshape(1).to(device)), dim = 0)
        AUC_roc.update(d[1].squeeze(), torch.squeeze(target_LMCI))
        AUC_ROC_LMCI = torch.cat((AUC_ROC_LMCI, AUC_roc.compute().reshape(1).to(device)), dim = 0)
        AUC_roc.update(d[1].squeeze(), torch.squeeze(target_EMCI))
        AUC_ROC_EMCI = torch.cat((AUC_ROC_EMCI, AUC_roc.compute().reshape(1).to(device)), dim = 0)
        AUC_roc.update(d[1].squeeze(), torch.squeeze(target_FTD))
        AUC_ROC_FTD = torch.cat((AUC_ROC_FTD, AUC_roc.compute().reshape(1).to(device)), dim = 0)

        # PR
        AUC_pr = BinaryAUPRC()

        AUC_pr.update(d[1].squeeze(), torch.squeeze(target_AD))
        AUC_PR_AD = torch.cat((AUC_PR_AD, AUC_pr.compute().reshape(1).to(device)), dim = 0)
        AUC_pr.update(d[1].squeeze(), torch.squeeze(target_PD))
        AUC_PR_PD = torch.cat((AUC_PR_PD, AUC_pr.compute().reshape(1).to(device)), dim = 0)
        AUC_pr.update(d[1].squeeze(), torch.squeeze(target_MCI))
        AUC_PR_MCI = torch.cat((AUC_PR_MCI, AUC_pr.compute().reshape(1).to(device)), dim = 0)
        AUC_pr.update(d[1].squeeze(), torch.squeeze(target_LMCI))
        AUC_PR_LMCI = torch.cat((AUC_PR_LMCI, AUC_pr.compute().reshape(1).to(device)), dim = 0)
        AUC_pr.update(d[1].squeeze(), torch.squeeze(target_EMCI))
        AUC_PR_EMCI = torch.cat((AUC_PR_EMCI, AUC_pr.compute().reshape(1).to(device)), dim = 0)
        AUC_pr.update(d[1].squeeze(), torch.squeeze(target_FTD))
        AUC_PR_FTD = torch.cat((AUC_PR_FTD, AUC_pr.compute().reshape(1).to(device)), dim = 0)


    results['rersults_AUC_ROC_AD'] = torch.median(AUC_ROC_AD)
    results['rersults_AUC_ROC_PD'] = torch.median(AUC_ROC_PD)
    results['rersults_AUC_ROC_MCI'] = torch.median(AUC_ROC_MCI)
    results['rersults_AUC_ROC_LMCI'] = torch.median(AUC_ROC_LMCI)
    results['rersults_AUC_ROC_EMCI'] = torch.median(AUC_ROC_EMCI)
    results['rersults_AUC_ROC_FTD'] = torch.median(AUC_ROC_FTD)

    results['rersults_AUC_PR_AD'] = torch.median(AUC_PR_AD)
    results['rersults_AUC_PR_PD'] = torch.median(AUC_PR_PD)
    results['rersults_AUC_PR_MCI'] = torch.median(AUC_PR_MCI)
    results['rersults_AUC_PR_LMCI'] = torch.median(AUC_PR_LMCI)
    results['rersults_AUC_PR_EMCI'] = torch.median(AUC_PR_EMCI)
    results['rersults_AUC_PR_FTD'] = torch.median(AUC_PR_FTD)


    results['rersults_AUC_PR'] = np.median([results['rersults_AUC_PR_AD'].item(),
                                    results['rersults_AUC_PR_PD'].item(),
                                    results['rersults_AUC_PR_MCI'].item(),
                                    results['rersults_AUC_PR_LMCI'].item(),
                                    results['rersults_AUC_PR_EMCI'].item(),
                                    results['rersults_AUC_PR_FTD'].item()])
    
    results['rersults_AUC_ROC'] = np.median([results['rersults_AUC_ROC_AD'].item(),
                                    results['rersults_AUC_ROC_PD'].item(),
                                    results['rersults_AUC_ROC_MCI'].item(),
                                    results['rersults_AUC_ROC_LMCI'].item(),
                                    results['rersults_AUC_ROC_EMCI'].item(),
                                    results['rersults_AUC_ROC_FTD'].item()])
    
    logging.getLogger().setLevel(logging.ERROR)
    return results

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

################# find best trhesold and parameters #################
def get_metric_and_best_threshold(y_test,y_pred,show_result=True,device='cpu',metric='f1'):
    """
    get trheshold that give the best F1 score 
    """

    result = {}
    for x in y_pred.keys():
        test = torch.cat(y_test[x]).cpu().numpy()
        pred = torch.cat(y_pred[x],dim=2).squeeze().cpu().numpy()
        num_pos_class = sum(test==1)
        num_neg_class = sum(test==0)
        fpr, tpr, thresholds = roc_curve(test, pred)
        tp = tpr * num_pos_class
        fp = fpr *num_pos_class
        tn = (1 - fpr) * num_neg_class
        fn = (1 - tpr) * num_pos_class
        acc = (tp + tn) / (num_pos_class + num_neg_class)
        f1 = (2*tp) / (2*tp+fp+fn)
        Bacc = ((tp/(tp+fn))+(tn/(tn+fp)))/2
        
        if metric == 'f1':
            best_threshold = thresholds[np.argmax(f1)]
        elif metric == 'acc':
            best_threshold = thresholds[np.argmax(acc)]
        elif metric == 'Bacc':
            best_threshold = thresholds[np.argmax(Bacc)]

        result[x] = {'fpr':fpr,'tpr':tpr,'tp':tp,'tn':tn,'acc':np.amax(acc),'thresholds':best_threshold}
        
        if show_result:
            print(f"best threshold for {x}: {round(thresholds[np.argmax(f1)],4)} with Accuracy: {round(max(acc),6)} and F1 score {round(max(f1),6)}")
     
    return result
########################### final stat ##########################
def Final_metrics(y_pred,y_test_metrix,y_test_CN,best_threshold,DataSetName):

    #### produce metrics
    result = {}
    result['data'] = {}
    result['accuracy'] = {}
    result['f1_score'] = {}
    result['AUC_ROC'] = {}
    result['AUC_PR'] = {}
    result['precision_score'] = {}
    result['recall_score'] = {}
    result['Balanced_acc'] = {}

    for x in y_pred.keys():
        prediction_thr = [1 if i > best_threshold[x]['thresholds'] else 0 for i in y_pred[x]]

        result['accuracy'][x] = accuracy_score(torch.cat(y_test_metrix[x]).numpy(),prediction_thr)
        result['f1_score'][x] = f1_score(torch.cat(y_test_metrix[x]).numpy(),prediction_thr)
        result['AUC_ROC'][x] = roc_auc_score(torch.cat(y_test_metrix[x]).numpy(),prediction_thr)
        fpr, tpr, thresholds = precision_recall_curve(torch.cat(y_test_metrix[x]).numpy(),prediction_thr)
        # Sort fpr and tpr arrays based on fpr values
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]

        result['AUC_PR'][x] = auc(fpr_sorted, tpr_sorted)
        result['precision_score'][x] = precision_score(torch.cat(y_test_metrix[x]).numpy(),prediction_thr)
        result['recall_score'][x] = recall_score(torch.cat(y_test_metrix[x]).numpy(),prediction_thr)
        result['Balanced_acc'][x] = balanced_accuracy_score(torch.cat(y_test_metrix[x]).numpy(),prediction_thr)
        result['data'][x] = DataSetName

    # compute metrics for CN as exclusion
    prediction_thr = []
    for i in range(len(y_test_CN)):
        re = 0
        for x in ['AD','PD','FTD','MCI','LMCI','EMCI']:
            if y_pred[x][i]>best_threshold[x]['thresholds']:
                re +=1
        if re==0:
            prediction_thr.append(1)
        else:
            prediction_thr.append(0)
    result['accuracy']['CN_EX'] = accuracy_score(y_test_CN,prediction_thr)
    result['f1_score']['CN_EX'] = f1_score(y_test_CN,prediction_thr)
    result['AUC_ROC']['CN_EX'] = roc_auc_score(y_test_CN,prediction_thr)
    fpr, tpr, thresholds = precision_recall_curve(y_test_CN,prediction_thr)
    try:
        result['AUC_PR']['CN_EX'] = auc(fpr, tpr)
    except:
        result['AUC_PR']['CN_EX'] = None
    result['precision_score']['CN_EX'] = precision_score(y_test_CN,prediction_thr)
    result['recall_score']['CN_EX'] = recall_score(y_test_CN,prediction_thr)
    result['Balanced_acc'][x] = balanced_accuracy_score(y_test_CN,prediction_thr)

    return result


################# Create target #################
def Create_target(y):
    y_AD = []
    y_MCI = []
    y_EMCI = []
    y_LMCI = []
    y_FTD = []
    y_PD = []
    y_CN = []
    for i in y:
        if i == 'AD':
            y_AD.append(1)
            y_MCI.append(1)
            y_EMCI.append(1)
            y_LMCI.append(1)
            y_FTD.append(0)
            y_PD.append(0)
            y_CN.append(0)

        elif i == 'FTD':
            y_AD.append(0)
            y_MCI.append(1)
            y_EMCI.append(1)
            y_LMCI.append(1)
            y_FTD.append(1)
            y_PD.append(0)
            y_CN.append(0)
        
        elif i == 'PD':
            y_AD.append(0)
            y_MCI.append(0)
            y_EMCI.append(0)
            y_LMCI.append(0)
            y_FTD.append(0)
            y_PD.append(1)
            y_CN.append(0)
        
        elif i == 'LMCI':
            y_AD.append(0)
            y_MCI.append(1)
            y_EMCI.append(1)
            y_LMCI.append(1)
            y_FTD.append(0)
            y_PD.append(0)
            y_CN.append(0)

        elif i == 'MCI':
            y_AD.append(0)
            y_MCI.append(1)
            y_EMCI.append(1)
            y_LMCI.append(1)
            y_FTD.append(0)
            y_PD.append(0)
            y_CN.append(0)
        
        elif i == 'EMCI':
            y_AD.append(0)
            y_MCI.append(1)
            y_EMCI.append(1)
            y_LMCI.append(0)
            y_FTD.append(0)
            y_PD.append(0)
            y_CN.append(0)

        else:
            y_AD.append(0)
            y_MCI.append(0)
            y_EMCI.append(0)
            y_LMCI.append(0)
            y_FTD.append(0)
            y_PD.append(0)
            y_CN.append(1)

    return y_AD,y_PD,y_MCI,y_EMCI,y_LMCI,y_FTD,y_CN

## Reference: https://github.com/davidtvs/pytorch-lr-finder/blob/14abc0b8c3edd95eefa385c2619028e73831622a/torch_lr_finder/lr_finder.py



def _total_num_neurons(model):
        """Determines the overall number of neurons in a provided PyTorch model.

        Args:
            model (torch.nn.Module): A PyTorch model object.

        Returns:
            int: Total number of neurons in the model.
        """
        total_neurons = 0
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                if isinstance(module, torch.nn.Linear):
                    total_neurons += module.in_features * module.out_features
                elif isinstance(module, torch.nn.Conv2d):
                    total_neurons += module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
        return total_neurons