#!/usr/bin/env python

from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import json

import torch
from torch.utils.data import Dataset
from torcheval.metrics import BinaryAUROC,BinaryAUPRC


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
def get_metric_and_best_threshold_from_roc_curve(y_test,y_pred,show_result=True,device='cpu'):
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
        best_threshold = thresholds[np.argmax(f1)]

        result[x] = {'fpr':fpr,'tpr':tpr,'tp':tp,'tn':tn,'f1':np.amax(f1),'thresholds':best_threshold}
        if show_result:
            print(f"best threshold for {x}: {round(thresholds[np.argmax(f1)],4)} with Accuracy: {round(max(acc),6)} and F1 score {round(max(f1),6)}")
     
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
            y_MCI.append(0)
            y_EMCI.append(0)
            y_LMCI.append(0)
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


################# number of neuron #################
def _total_num_neurons(model):
        """Determines the overall number of neurons in a provided PyTorch model.

        Args:
            model (torch.nn.Module): A PyTorch model object.

        Returns:
            int: Total number of neurons in the model.
        """
        total_neurons = 0
        total_weighted_layers = 0
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                total_weighted_layers += 1
                if isinstance(module, torch.nn.Linear):
                    total_neurons += module.in_features * module.out_features
                elif isinstance(module, torch.nn.Conv2d):
                    total_neurons += module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]
        return total_neurons,total_weighted_layers

def _save_nueron_per_perfomance(file_path,disease,architecture,total_neurons,total_layers,metrics,avg,r):
    """Save model results and number of neuron

    Returns:
        csv file
    """
    data = {'disease':disease,
            "architecture":architecture,
            "total_neurons":total_neurons,
            "total_layers":total_layers,
            "metrics":metrics,
            "avg": avg,
            "random_state" : r
            }
    try:
        with open(file_path, 'r+') as file:
            # Load existing JSON data
            existing_data = json.load(file)
            # Append new data
            existing_data.append(data)
            # Move the file pointer to the beginning
            file.seek(0)
            # Write the updated JSON data
            json.dump(existing_data, file, indent=4)
            file.truncate()
    except FileNotFoundError:
        with open(file_path, 'w') as file:
            # If the file doesn't exist, create a new one and write the data
            json.dump([data], file, indent=4)