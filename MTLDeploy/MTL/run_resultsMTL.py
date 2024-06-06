#!/usr/bin/env python

from MTL.CombinedModel_TestGender import run_modelCV

def mainMTL():
    exp_num = 200
    for LR_schedurler in ['StepLR']:#,'ConstantLR']:
        for selected_gender in [['M'],['F']]:#['FAMO','Classic','focal_loss']:
            for freeze in ['Hybrid']:#['Clean','Freeze_tasks','NtFreeze','Hybrid']:
                for r in [23, 362, 191, 80, 769, 328, 204, 281, 841, 31]:
                    run_modelCV(path_roi='../../data/MRI_rois_20211114//MRI_rois_20211114.csv',
                        path_colnames='../../data/MRI_Features/MRI_Features.csv',
                        path_result='exp'+ str(exp_num),
                        result_folder = 'MTL/Results_mtl_Gender',
                        random_stat = r,
                        selected_gender = selected_gender,
                        selected_diagnosis = ['CN','AD','PD','LMCI','EMCI','MCI','FTD'],
                        network_design = 'CombinOptMTL',
                        training_algortim = 'FAMO',
                        freeze = freeze,
                        save_resultsCSV=True)
                    
                    exp_num +=1


if __name__ == "__mainMTL__":
    mainMTL()
