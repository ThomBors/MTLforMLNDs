import numpy as np


class EarlyStopper:
    """
    Early arrest criterion

    Early stopping criterion that observes the trend of validation loss 
    if the training does not result in a decrease in validation loss, then it is stopped

    patience: number of consecutive epochs to wait if the validation loss does not decrease 
    min_delta: delta applied to the validation loss to overcome the variability of epochs (self.min_validation_loss + self.min_delta)

    """
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    


class EarlyStopperMTL:
    """
    Early arrest criterion

    Early stopping criterion that observes the trend of validation loss of each task 
    if the training does not result in a decrease in validation loss, then it freeze the weight of the training task

    patience: number of consecutive epochs to wait if the validation loss does not decrease 
    min_delta: delta applied to the validation loss to overcome the variability of epochs (self.min_validation_loss + self.min_delta)

    """
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta

        self.freez ={"AD" : 0,
                    "FTD": 0,
                    "LMCI" : 0,
                    "EMCI" : 0,
                    "MCI" : 0,
                    "PD" : 0,
                    "CN" :0}

        self.counter = {"AD" : 0,
                        "FTD": 0,
                        "LMCI" : 0,
                        "EMCI" : 0,
                        "MCI" : 0,
                        "PD" : 0,
                        "CN" :0}
        
        self.min_validation_loss = {"AD" : np.inf,
                                    "FTD": np.inf,
                                    "LMCI" : np.inf,
                                    "EMCI" : np.inf,
                                    "MCI" : np.inf,
                                    "PD" : np.inf,
                                    "CN" :np.inf}
    

    def early_stop(self,model, validation_dict):
        for disese,val_loss in validation_dict.items():
            if val_loss < self.min_validation_loss[disese]:
                self.min_validation_loss[disese] = val_loss
                self.counter[disese] = 0
            elif val_loss > (self.min_validation_loss[disese] + self.min_delta):
                self.counter[disese] += 1
                if self.counter[disese] >= self.patience:
                    if self.freez[disese] == 0 :
                        for param in model.task_specific_models[disese].parameters():
                            param.requires_grad = False
                        self.freez[disese] =+1
                        print(f"freeze task latyer: {disese}")