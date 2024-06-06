#!/usr/bin/env python

from MTL.CombinedModel import run_model


def mainMTL():
    exp_num = 1
    for selected_gender in [['M','F'],]:
        for freeze in ['Hybrid','Clean','Freeze_tasks','NtFreeze']:
            for r in [23, 362, 191, 80, 769, 328, 204, 281, 841, 31]:
                run_model(path_roi='../../data/MRI_rois_20211114//MRI_rois_20211114.csv',
                    path_colnames='../../data/MRI_Features/MRI_Features.csv',
                    path_result='exp'+ str(exp_num),
                    result_folder = 'MTL/experiments_mtl',
                    random_stat = r,
                    selected_gender = selected_gender,
                    selected_diagnosis = ['CN','AD','PD','LMCI','EMCI','MCI','FTD'],
                    network_design = 'CombinOptMTL',
                    training_algortim = 'FAMO',
                    freeze = freeze,
                    learning_rate = 0.001,
                    LR_schedurler = 'StepLR',
                    epochs = 500,
                    step_size = 100,
                    gamma = 0.5,
                    batch_size = 256,
                    weight_decay = 0.00025,
                    project_wandb = "MTL_experiment",
                    online = (r == 23),
                    save_resultsCSV=True,
                    save_model=True)
                
                exp_num +=1


if __name__ == "__mainMTL__":
    mainMTL()
