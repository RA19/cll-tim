'''
File: 2_data2ML_discreteTime_ensemble_RAfeatures.py
File Created: 2020-10-07
Author: Adrian Gabriel Zucco (adrian.gabriel.zucco@regionh.dk)
-----
Last Modified: 2020-10-07 12:27:17 pm
Modified By: Adrian Gabriel Zucco (adrian.gabriel.zucco@regionh.dk>)
'''

# %%
import datetime
import numpy as np
import pandas as pd
import sklearn
import shap
import scipy.stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import pickle
import re
import numbers
from scipy import stats
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool, cv
# from pycox.evaluation import EvalSurv

# Custom Functions
import sys
#sys.path.append('../')




def to_predict_DT(to_predict_data, time_cuts, time_column_name):

    if isinstance(time_cuts, numbers.Integral):
        time_cuts = np.arange(start=1, stop=time_cuts + 1, step=1)

    # Create dataframe of the time cuts
    t_df = pd.DataFrame({time_column_name: time_cuts})
    # Expand the data by duplicating rows according to the ammount of time cuts
    dup_pred = to_predict_data.iloc[np.arange(
        len(to_predict_data)).repeat(t_df.shape[0])]
    # Match length of both dataframes
    t_df = pd.concat([t_df] * to_predict_data.shape[0], ignore_index=True)
    # Concatenante dataframes
    to_pred_with_time = pd.concat(
        [dup_pred.reset_index(drop=True), t_df], axis=1)

    return(to_pred_with_time)


def make_range(x): return tuple(np.arange(start=1, stop=x + 1, step=1))


def train_discrete_time(dataframe, time_column, event_column, event_name, divide_time_by=1, max_time=None):

    OC = dataframe.copy()
    # Divide the time data to transform e.g if data is on days and desired column is on weeks then use 7
    OC["discretised"] = OC[time_column].div(divide_time_by)
    OC["range_time"] = OC["discretised"].apply(make_range)

    # Expand time column
    OC_DT = OC.explode("range_time")

    # Create column of target values only using 1 when an event occured
    OC_DT["DT_target"] = np.array(
        (OC_DT["discretised"] < OC_DT["range_time"]) & (OC_DT[event_column] == event_name)).astype(int)
    
    # Filter rows after censoring or event if any
    result = OC_DT[(OC_DT["DT_target"] == 1) | (
        OC_DT.range_time < OC_DT.discretised)]
    result["range_time"] = result["range_time"].astype("int")

    # Remove rows up to max_time
    if max_time != None:
        result = result[result.range_time < max_time + 1]

    return result





## GLOBAL OPTIONS
np.set_printoptions(precision=4)
pp = pprint.PrettyPrinter(indent=4)

# def DataGen():
# %% Load Parameters

# %% Load Parameters
# ################### RUN PARAMETERS ################################
cross_val_set = "cv5_60"
time_col_outcomes = "diff_start2end"
time_col_feat = "Days_fromstart"
target_col_name = "DT_target"
save_models = True
idx_col = "index"
events_col = "last_event"
event_name = "Death"
period_in_days = 7
max_time = 12
Outcome_Name='dead'
#####################################################################




with open('outcomes_censor1510_base_feat_splits.p','rb') as fp:
    OC=pickle.load(fp)
    
# Hospitlized only
OC=OC[OC['first_event']=='Admission']  
OC.reset_index(drop=True,inplace=True)
OC['AbsoluteMonth']=OC['Admission_dttm'].dt.year.astype(str)+'_'+OC['Admission_dttm'].dt.month.astype(str)

OC['AbsoluteMonth'][OC['AbsoluteMonth']=='2020_3']=1
OC['AbsoluteMonth'][OC['AbsoluteMonth']=='2020_4']=2
OC['AbsoluteMonth'][OC['AbsoluteMonth']=='2020_5']=3
OC['AbsoluteMonth'][OC['AbsoluteMonth']=='2020_6']=4
OC['AbsoluteMonth'][OC['AbsoluteMonth']=='2020_7']=5
OC['AbsoluteMonth'][OC['AbsoluteMonth']=='2020_8']=6


   
   
metaFolds = OC[['Key_Patient', 'diff_start2end', 'last_event', 'test_set', 'train80_val20',
           'cv3', 'cv5', 'cv3_60', 'cv5_60','dead','AbsoluteMonth']]
metaFolds = metaFolds.drop_duplicates(subset=[
    'Key_Patient'], keep='first')



# %% LOAD DATA
# Meta data
# meta = pd.read_feather("outcomes_base_feat_splits.feather")
# meta=meta[np.logical_and(meta.HOSP_BYCOVID_STAY_NUMBER<2,meta.HOSP_BYCOVID_STAY_NUMBER>0)]
# meta=meta.drop_duplicates(['Key_Patient']).reset_index(drop=True)
# meta_noage=meta.drop(['age'],axis=1).reset_index(drop=True)



with open('COVIMUN_META_Matrix_N.p','rb') as fp:
    meta=pickle.load(fp)

# Combine the meta data with the rolling model information
meta_filt = metaFolds.merge(meta, left_on=['Key_Patient'], 
                       right_on=['Key_Patient'], how='left')
        
# meta['KP']=meta.Key_Patient.copy()
# meta['Key_Patient']=meta.Key_Patient_Days
# del meta['Key_Patient_Days']

with open('COVIMUN_Matrix_N.p','rb') as fp:
    data=pickle.load(fp)

   # del data['Key_Patient_x'],data['Key_Patient_y'],data['date']

columns_trained=data.columns
columns_trained=columns_trained[columns_trained!='Key_Patient']
columns_trained=columns_trained[columns_trained!='Key_Patient_Days']

Thold=0.80
print(data.shape)
#Removing Null Columns
null_columns=data.isna().sum()/np.shape(data)[0]
to_del=null_columns[null_columns>Thold]
data.drop(to_del.index, axis=1, inplace=True)
print(data.shape)
columns_trained=data.columns
columns_trained=columns_trained[columns_trained!='Key_Patient']
columns_trained=columns_trained[columns_trained!='Key_Patient_Days']


wide_target_folds=meta_filt.copy()
        
# ROLLING TIME ADAPTATION
wide_target_folds["start2end_rolling"] = wide_target_folds[time_col_outcomes] - \
    wide_target_folds[time_col_feat].astype("int")
time_col = "start2end_rolling"


# Divide the time into the needed period
# ! change name of this and add as parameter to traing_discrete_time
wide_target_folds["range_time"] = wide_target_folds[time_col].div(
    period_in_days)


# Remove any possible error of negative days
wide_target_folds = wide_target_folds[wide_target_folds[time_col] > 0]
wide_target_folds=wide_target_folds.reset_index(drop=True)

# Censors INDICATORS
# Post Discharge Censor
# Post_Discharge_Censor=np.logical_or(wide_target_folds['DaysSinceHosp_END']<=0,wide_target_folds['DaysSinceHosp_END'].isnull()).values
# wide_target_folds['Post_Discharge_Censor']=Post_Discharge_Censor==0

# Non Hospitalized Censor
# Remove not hospitalised patients
wide_target_folds['Non_Hosp_Censor']=wide_target_folds["Days"].notnull()==0
# wide_target_folds=wide_target_folds[Non_Hosp_Censor==0]


# Binary Lack of Follow Up Censor
wide_target_folds['BinaryFollowUp_Censor']=np.logical_and(wide_target_folds['start2end_rolling']<(max_time*period_in_days),wide_target_folds[Outcome_Name]==0)
wide_target_folds[target_col_name]=wide_target_folds[Outcome_Name]



# Turn the data into DT format for training (ToTrain_AUG)
target_folds = train_discrete_time(
    wide_target_folds, time_col, events_col, event_name, period_in_days, max_time)
target_folds=target_folds.reset_index(drop=True)



target_folds_withFeatures = target_folds.merge(data, left_on=['Key_Patient','Key_Patient_Days','Days','Days_fromstart',
                                            'Days_in_hosp','in_hosp','Days_in_ICU','in_ICU'], right_on=[
    'Key_Patient','Key_Patient_Days','Days','Days_fromstart',
                                            'Days_in_hosp','in_hosp','Days_in_ICU','in_ICU'], how='left').reset_index(drop=True)
target_folds_withFeatures=target_folds_withFeatures.reset_index(drop=True)


ToTrain_AUG=target_folds_withFeatures[np.append('AbsoluteMonth',np.append('range_time',columns_trained))]
ToTrain_AUG_Info=target_folds_withFeatures.drop(columns_trained,axis=1)


# Turn the data into DT format for evaluation (ToEval_AUG)
Pred_Aug=to_predict_DT(wide_target_folds, max_time, "RT")
Pred_Aug['range_time']=Pred_Aug['RT'].copy()
del Pred_Aug['RT']



Pred_Aug_withFeatures = Pred_Aug.merge(data, left_on=['Key_Patient','Key_Patient_Days','Days','Days_fromstart',
                                            'Days_in_hosp','in_hosp','Days_in_ICU','in_ICU'], right_on=[
    'Key_Patient','Key_Patient_Days','Days','Days_fromstart',
                                            'Days_in_hosp','in_hosp','Days_in_ICU','in_ICU'], how='left').reset_index(drop=True)
Pred_Aug_withFeatures=Pred_Aug_withFeatures.reset_index(drop=True)

ToEval_AUG=Pred_Aug_withFeatures[np.append('AbsoluteMonth',np.append('range_time',columns_trained))]
ToEval_AUG_Info=Pred_Aug_withFeatures.drop(columns_trained,axis=1)




################# PREPARING DATA FOR GA ######################################
# Format: Dictionary with: 'TrainData','EvalData','TrainTargets',',EvalTargets','TrainCVFoldIndexes']
# IMPORTANT: TrainFoldIndexes should have no nans, and have the same number of rows as TrainData
# IMPORTANT: 
    



InData={}
idx_train = np.squeeze(ToTrain_AUG_Info.train80_val20.values == 0)
idx_val = np.squeeze(ToTrain_AUG_Info.train80_val20.values == 1)
InData['ForTrain_Features']=ToTrain_AUG.copy().reset_index(drop=True)
ToTrain_AUG_Info['Binary_Targets']=ToTrain_AUG_Info[target_col_name].reset_index(drop=True)
ToTrain_AUG_Info['General_Targets']=ToTrain_AUG_Info[target_col_name].reset_index(drop=True)
ToTrain_AUG_Info['CV_Fold_Indexes']=ToTrain_AUG_Info.cv5_60.reset_index(drop=True)
ToTrain_AUG_Info['idx_train']=idx_train
ToTrain_AUG_Info['idx_val']=idx_val
InData['ForTrain_Info']=ToTrain_AUG_Info.copy().reset_index(drop=True)





idx_train = np.squeeze(ToEval_AUG_Info.train80_val20.values == 0)
idx_val = np.squeeze(ToEval_AUG_Info.train80_val20.values == 1)
InData['ForEval_Features']=ToEval_AUG.copy().reset_index(drop=True)
ToEval_AUG_Info['Binary_Targets']=ToEval_AUG_Info[target_col_name].reset_index(drop=True)
ToEval_AUG_Info['General_Targets']=ToEval_AUG_Info[target_col_name].reset_index(drop=True)
ToEval_AUG_Info['CV_Fold_Indexes']=ToEval_AUG_Info.cv5_60.reset_index(drop=True)
ToEval_AUG_Info['idx_train']=idx_train
ToEval_AUG_Info['idx_val']=idx_val
InData['ForEval_Info']=ToEval_AUG_Info.copy().reset_index(drop=True)




with open('COVIMUN_GA_Data_Weekly_N_80_Hosp.p','wb') as fp:
    pickle.dump(InData,fp,protocol=pickle.HIGHEST_PROTOCOL)
    

InData={}
idx_train = np.squeeze(ToTrain_AUG_Info.test_set.values == 0)
idx_val = np.squeeze(ToTrain_AUG_Info.test_set.values == 1)
InData['ForTrain_Features']=ToTrain_AUG.copy().reset_index(drop=True)
ToTrain_AUG_Info['Binary_Targets']=ToTrain_AUG_Info[target_col_name].reset_index(drop=True)
ToTrain_AUG_Info['General_Targets']=ToTrain_AUG_Info[target_col_name].reset_index(drop=True)
ToTrain_AUG_Info['CV_Fold_Indexes']=ToTrain_AUG_Info.cv5.reset_index(drop=True)
ToTrain_AUG_Info['idx_train']=idx_train
ToTrain_AUG_Info['idx_val']=idx_val
InData['ForTrain_Info']=ToTrain_AUG_Info.copy().reset_index(drop=True)



idx_train = np.squeeze(ToEval_AUG_Info.test_set.values == 0)
idx_val = np.squeeze(ToEval_AUG_Info.test_set.values == 1)
InData['ForEval_Features']=ToEval_AUG.copy().reset_index(drop=True)
ToEval_AUG_Info['Binary_Targets']=ToEval_AUG_Info[target_col_name].reset_index(drop=True)
ToEval_AUG_Info['General_Targets']=ToEval_AUG_Info[target_col_name].reset_index(drop=True)
ToEval_AUG_Info['CV_Fold_Indexes']=ToEval_AUG_Info.cv5.reset_index(drop=True)
ToEval_AUG_Info['idx_train']=idx_train
ToEval_AUG_Info['idx_val']=idx_val
InData['ForEval_Info']=ToEval_AUG_Info.copy().reset_index(drop=True)


with open('COVIMUN_GA_Data_Weekly_For_Test_N_80_Hosp.p','wb') as fp:
    pickle.dump(InData,fp,protocol=pickle.HIGHEST_PROTOCOL) 
    
    
#    
