import pandas as pd
from sklearn.model_selection import train_test_split

def createTVT(path_roi = '../../data/MRI_rois_20211114//MRI_rois_20211114.csv',
              path_colnames = '../../data/MRI_Features/MRI_Features.csv',
              selected_diagnosis = ['CN','AD','PD','LMCI','EMCI','MCI','FTD'],
              selected_gender = ['M','F'],random_seed = 23):
    
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
    X_train_val,X_test, y_train_val,y_test = train_test_split(df_CNAD,df_CNAD['SD003'], 
                                                        test_size=0.1, 
                                                        random_state=23,
                                                        stratify=df_CNAD[['SD001','class']])

    #4.2# Test/validation split with stratification on Study and class
    test_validation_ratio = 0.1 / (1 - 0.1)  # Ratio of test size to train/validation size
    X_train,X_validation,y_train, y_validation = train_test_split(X_train_val,y_train_val, 
                                                        test_size=test_validation_ratio, 
                                                        random_state=random_seed,
                                                        stratify=X_train_val[['SD001','class']])
    
    X_train = X_train[fs_feature_names]
    X_validation = X_validation[fs_feature_names]
    X_test = X_test[fs_feature_names]

    return X_train,y_train,X_validation,y_validation,X_test,y_test