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
import feather
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

from scipy import stats
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool, cv
# from pycox.evaluation import EvalSurv

# Custom Functions
import sys
sys.path.append('../')
import useful_funcs_testing as uf

## GLOBAL OPTIONS
np.set_printoptions(precision=4)
pp = pprint.PrettyPrinter(indent=4)


# %% Load Parameters
# ################### RUN PARAMETERS ################################
cross_val_set = "cv5_60"
time_col = "start2end"
target_col_name = "DT_target"
save_models = True
idx_col = "Key_Days"
events_col = "last_event"
event_name = "Death"
period_in_days = 30
max_time = 90
Outcome_Name='dead'
#####################################################################


# %% LOAD DATA
# Meta data
meta = pd.read_feather("outcomes_base_feat_splits.feather")
meta=meta[np.logical_and(meta.HOSP_BYCOVID_STAY_NUMBER<2,meta.HOSP_BYCOVID_STAY_NUMBER>0)]
meta=meta.drop_duplicates(['Key_Patient']).reset_index(drop=True)
meta_noage=meta.drop(['age'],axis=1).reset_index(drop=True)


# Pre-hospitalisation data
source_folder = "L:/LovbeskyttetMapper/COVIMUN/Data/EHR/encoded_features/RudiAllFeatures/"
df_PreHosp = pickle.load(open(source_folder + "PreHosp_ALL.pkl", "rb"))
df_PreHosp = df_PreHosp.add_suffix('_PH')
df_PreHosp = df_PreHosp.rename(columns={'Key_Patient_PH': 'Key_Patient'})

# Post-hospitalisation data
df_PostHosp = pickle.load(open(source_folder + "PostHosp_ALL.pkl", "rb"))
df_PostHosp['Key_Days'] = df_PostHosp['Key_Patient'].copy()
df_PostHosp[['Days', 'Key_Patient']
            ] = df_PostHosp['Key_Patient'].str.split('_', expand=True)

# Merge both
data = df_PostHosp.merge(df_PreHosp, left_on=['Key_Patient'], right_on=[
                       'Key_Patient'], how='left')

# Release some RAM
del df_PreHosp
del df_PostHosp


# %%
folder_name = "ensembe_featRA"
FeatureList = np.loadtxt(
    'L:/LovbeskyttetMapper/COVIMUN/Scripts/ML_pipeline/trained_models/{}/columns_to_predict_DT2.txt'.format(folder_name), dtype='str', delimiter='\n')
# columns_trained = np.loadtxt('./trained_models/{}/columns_to_predict_MC.txt'.format(folder_name), dtype = 'str')

columns_trained=np.append(['Key_Patient','Days','Key_Days'],FeatureList)
columns_trained=columns_trained[columns_trained!='age']
columns_trained=columns_trained[columns_trained!='ClassFeat']
# %% Explore
print("\n ################# \n")
print(data.info())
# print(data.columns[:12])
# for pair in enumerate(data.columns[:30]):
#     print(pair) 

# ? Using Rudi features
data = data[columns_trained]



# Combine the meta data with the rolling model information
wide_target_folds = meta.merge(data, left_on=['Key_Patient'], right_on=[
    'Key_Patient'], how='left').reset_index(drop=True)
wide_target_folds=wide_target_folds.reset_index(drop=True)


wide_target_folds=wide_target_folds[wide_target_folds.Days.notnull()]
# ROLLING TIME ADAPTATION
wide_target_folds["start2end_rolling"] = wide_target_folds.diff_start2end - \
    wide_target_folds.Days.astype("int")
time_col = "start2end_rolling"

# Divide the time into the needed period
# ! change name of this and add as parameter to traing_discrete_time
wide_target_folds["range_time"] = wide_target_folds[time_col].div(
    period_in_days)

wide_target_folds = wide_target_folds[wide_target_folds[time_col] > 0]
wide_target_folds=wide_target_folds.reset_index(drop=True)
wide_target_folds['DaysInHosp']=wide_target_folds.Days.astype(float)
wide_target_folds['DaysInHosp'][wide_target_folds.HOSP_BYCOVID_STAY_NUMBER==0]=np.nan
wide_target_folds['DaysSinceHosp_END']=wide_target_folds.Days.astype(float)-(wide_target_folds.Discharge_dttm-wide_target_folds.Admission_dttm).astype('timedelta64[D]').astype(float)
wide_target_folds['DaysInHosp'][wide_target_folds['DaysSinceHosp_END']>0]=np.nan
wide_target_folds['DaysSinceHosp_END'][wide_target_folds['DaysSinceHosp_END']<0]=np.nan
wide_target_folds['TToDeath']=(wide_target_folds['Death_dttm']-wide_target_folds['first_event_dttm']).astype('timedelta64[D]').astype(float)
wide_target_folds['DaysInICU']=wide_target_folds.Days.astype(float)-(wide_target_folds.ICU_STAY_START_dttm-wide_target_folds.Admission_dttm).astype('timedelta64[D]').astype(float)
wide_target_folds['DaysSinceICU_END']=wide_target_folds.Days.astype(float)-(wide_target_folds.ICU_STAY_END_dttm-wide_target_folds.Admission_dttm).astype('timedelta64[D]').astype(float)
wide_target_folds['DaysInICU'][wide_target_folds['DaysSinceICU_END']>0]=np.nan
wide_target_folds['DaysInICU'][wide_target_folds['DaysInICU']<0]=np.nan
wide_target_folds['DaysSinceICU_END'][wide_target_folds['DaysSinceICU_END']<0]=np.nan


# Censors INDICATORS
# Post Discharge Censor
Post_Discharge_Censor=np.logical_or(wide_target_folds['DaysSinceHosp_END']<=0,wide_target_folds['DaysSinceHosp_END'].isnull()).values
wide_target_folds['Post_Discharge_Censor']=Post_Discharge_Censor==0
# wide_target_folds=wide_target_folds[Post_Discharge_Censor==0]
# Non Hospitalized Censor
# Remove not hospitalised patients
wide_target_folds['Non_Hosp_Censor']=wide_target_folds["Days"].notnull()==0
# wide_target_folds=wide_target_folds[Non_Hosp_Censor==0]


# Binary Lack of Follow Up Censor
wide_target_folds['BinaryFollowUp_Censor']=np.logical_and(wide_target_folds['start2end_rolling']<max_time,wide_target_folds[Outcome_Name]==0)
wide_target_folds[target_col_name]=wide_target_folds.dead

# Turn the data into DT format for training (ToTrain_AUG)
target_folds = uf.train_discrete_time(
    wide_target_folds, time_col, events_col, event_name, period_in_days, max_time)
target_folds=target_folds.reset_index(drop=True)

ToTrain_AUG=target_folds[np.append('range_time',columns_trained)]
ToTrain_AUG=ToTrain_AUG.drop(['Key_Patient','Key_Days'],axis=1)

columns_to_remove=columns_trained[columns_trained!='Key_Patient']
columns_to_remove=columns_to_remove[columns_to_remove!='Key_Days']
columns_to_remove=columns_to_remove[columns_to_remove!='Days']
ToTrain_AUG_Info=target_folds.drop(columns_to_remove,axis=1)


# Turn the data into DT format for evaluation (ToEval_AUG)
Pred_Aug=uf.to_predict_DT(wide_target_folds, max_time,period_in_days, "RT")
Pred_Aug['range_time']=Pred_Aug['RT'].copy()
del Pred_Aug['RT']
ToEval_AUG=Pred_Aug[np.append('range_time',columns_trained)]
ToEval_AUG=ToEval_AUG.drop(['Key_Patient','Key_Days'],axis=1)
ToEval_AUG_Info=Pred_Aug.drop(columns_to_remove,axis=1)




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






# # Post Discharge Censor
# Post_Discharge_Censor=np.logical_or(wide_target_folds['DaysSinceHosp_END']<=0,wide_target_folds['DaysSinceHosp_END'].isnull()).values
# wide_target_folds['Post_Discharge_Censor']=Post_Discharge_Censor==0

# # Non Hospitalized Censor
# # Remove not hospitalised patients
# wide_target_folds['Non_Hosp_Censor']=wide_target_folds["Days"].notnull()==0

# # Binary Lack of Follow Up Censor
# wide_target_folds['No_Full_FollowUp_Censor']=np.logical_and(wide_target_folds['start2end_rolling']<max_time,wide_target_folds[Outcome_Name]==0)
# wide_target_folds[target_col_name]=wide_target_fo

################# GA PARAMETERS ##############################################


GA_Params={
'NClasses':1 ,# Number of Classes
'CMode':'DiscreteTime',#'MultiClass'
'No_of_Features':np.shape(InData['ForTrain_Features'])[1], # Size of Entire Feature Set
'Gran':50, # Granularity level for possible parameter values in hyperparameter opt
'FeatSize':np.shape(InData['ForTrain_Features'])[1],# Maximum Number of Features Per Algo
'EnsSize':4, # Size of Ensemble
'popsize':100, # Initial Population Size for GA (Larger better, but more expensive)
'NGEN':25, # Number of GA Iterations
'CVType': 'Given', # Options include 'None' 'Given','RbyF' (Given will use InData['TrainCVFoldIndexes'])
'CV_RbyF': '1_10', # R is number of repetitions, F is number of folds. for 10CV simple do 1_10
'OptMetric':np.array(['MCC']), # Metric to Optimize # FIXED TO MCC AT MOMENT
'Fitness_set': 'Val', # CV,Val, or CV_Val
'ForTrain_Censors': ['Post_Discharge_Censor','Non_Hosp_Censor'],
'ForEval_Censors': ['Post_Discharge_Censor','Non_Hosp_Censor','BinaryFollowUp_Censor'],
'Alg_n_jobs': 1
}

from timeit import default_timer as timer
import GA_Functions
tic=timer()
################# RUNNING GA ################################################
Final_Population,Top10=GA_Functions.RunGA(GA_Params,InData)
print(timer()-tic)

10 cores=340s
112 147 584 (245 with 2 cores)
885 378 308
"""
# %%
binary_metrics = []
survival_metrics = []
Fold_preds = {}
predicted_probs = {}
target_values = {}
binary_targets = []
hospitalised = []
models = {}
PredData=np.zeros((1,np.shape(target_folds)[1]+2))

models_to_train = ["catboost"]
# models_to_train = ["cbm_gdt"]
# models_to_train = ["lgb_rf"]
# models_to_train = ["lgb_gbt", "lgb_rf"]

for val_fold in range(folds[0], folds[1]+1):
    print("\n### Fold ", val_fold)
    
    preds_for_fold = []

    for model_type in models_to_train:
        print("\n\n################# Training model: ",
            model_type, " #################")

        print("# Parsing train and validation set...")
        # Augment data in discrete time way
        idx_train = target_folds.loc[:, cross_val_set] != val_fold
        y_train = target_folds[idx_train.values].loc[:,
                                                     target_col_name].astype("int")
        X_train = tomodel.loc[idx_train.values]

        idx_val = target_folds.loc[:, cross_val_set] == val_fold
        y_val = target_folds[idx_val].loc[:, target_col_name]
        X_val = tomodel.loc[idx_val.values]

        print("# Initiating training...")

        if 'cbm' in model_type:
            model = CatBoostClassifier(**all_params[model_type])
            model.fit(X_train, y_train, cat_features, eval_set=[
                      (X_val, y_val)], plot=False)#, use_best_model=True)

        if 'lgb' in model_type:
            model = lgb.LGBMClassifier(**all_params[model_type])
            model.fit(X_train, y_train, eval_set=[
                      (X_val, y_val)], early_stopping_rounds=None, verbose=10)

        if 'catboost' in model_type:
  
            trainpool = Pool(X_train, y_train, cat_features)
            valpool = Pool(X_val, y_val, cat_features)
            
            
            # Training
            model = CatBoostClassifier(**all_params[model_type])
            model.fit(trainpool, eval_set = valpool)
            model.shrink(model.best_iteration_+1)            


            # Add model to dictionary
        models[model_type + "_" + str(val_fold)] = model

        # Format data to predict at all time points
        idx_val_wide = wide_target_folds.loc[:, cross_val_set] == val_fold
        
        # Get the sample size
        sample_size = tomodel_wide[idx_val_wide.values].shape[0]

        X_val_long = uf.to_predict_DT(
            tomodel_wide[idx_val_wide.values], max_time,period_in_days, "range_time")

        X_val_long_target_folds = uf.to_predict_DT(
            wide_target_folds[idx_val_wide.values], max_time,period_in_days, "range_time")
        
        # TODO index by variables better using fold order
        # hospitalised.append(features_filt[idx_val_wide.values].hospitalised.values == 1)

        # y_val_prob = model.predict_proba(X_val_long)
        y_val_prob = model.predict(X_val_long,prediction_type='Probability')
        y_val_prob = y_val_prob[:, 1]
        y_val_long=X_val_long_target_folds.loc[:, target_col_name]
        predicted_probs[model_type + "_" + str(val_fold)] = y_val_prob
        
        PredData=np.row_stack((PredData,np.column_stack((y_val_prob,
                                                         y_val_long.values,
                                                         np.int64(y_val_prob>0.5),
                                                         X_val_long_target_folds.values))))


        PredData_df_Sub=pd.DataFrame(data=np.column_stack((y_val_prob,y_val_long.values,np.int64(y_val_prob>0.5),X_val_long_target_folds.values)),
                                 columns=np.append(['Prob','Pred','Outcome'],X_val_long_target_folds.columns))


        PredData_df_Sub=X_val_long_target_folds[['Key_Days','Days','dead','start2end_rolling','DT_target']]
        PredData_df_Sub['Prob']=y_val_prob
        PredData_df_Sub['range_time']=X_val_long['range_time']
        YO=PredData_df_Sub.groupby(['Key_Days','Days','dead','start2end_rolling','DT_target'])['Prob'].apply(lambda x: x.product()).reset_index()
        YO=YO[np.logical_and(YO['start2end_rolling']<max_time,YO['DT_target']==0)==False]


        FeatEncV2.EvalMetrics(YO.Prob.astype(float),YO.dead.astype(int),0.5)
        EvaResyBinary=FeatEncV2.EvalMetrics(YO.Prob.astype(float),YO.dead.astype(int),0.5)
        print(EvaResyBinary)

        # For the meta-learner
        preds_for_fold.append(model.predict_proba(X_val)[:, 1])

        print("\n# Fold ", val_fold, "  model  ", model_type, " PERFORMANCE ON VALIDATION SET")



        ###### BINARY METRICS
        range_timeZ=np.arange(start=1, stop=max_time + 1, step=period_in_days).argsort()+1
        for eval_at_t in range_timeZ:
        # ! fixed to get metrics at an specific time generalise to other time steps
        # for eval_at_t in range(3, 4):
            prob_threshold = 0.5
            y_val_t = uf.binary_time_target(
                wide_target_folds[idx_val_wide], "range_time", events_col, event_name, eval_at_t)
            bin_metric = uf.EvalPred_at_time(
                y_val_prob, sample_size, len(range_timeZ), eval_at_t, y_val_t, prob_threshold)
            bin_metric["Val_fold"] = val_fold
            bin_metric["Prob_threshold"] = prob_threshold
            bin_metric["best_iteration"] = model.best_iteration_
            bin_metric["model_type"] = model_type

            binary_metrics.append(bin_metric)
            binary_targets.append(y_val_t)

        pp.pprint(bin_metric)

    # Out of the model loop
    Fold_preds[val_fold] = [np.vstack(preds_for_fold), y_val.values]

predicted_prob=y_val_prob
samples_size=sample_size

time_steps=len(range_timeZ)
time_to_eval=eval_at_t
binary_target=y_val_t
threshold=prob_threshold
indexes=np.empty(shape=0) 
as_dict=False

cum_incidence_df = 1 - \
    uf.survival_df(predicted_prob, samples_size, time_steps)

if indexes.shape[0] != 0:
    cum_incidence_df = cum_incidence_df.iloc[:, indexes]
    binary_target = binary_target[indexes]

nocens = ~np.isnan(binary_target)

# -1 since Python index start at 0
Truthy=binary_target[nocens]
Preddy=cum_incidence_df.iloc[time_to_eval, :].values[nocens]
try:
    EvaResy_at_T = EvalPred(
        cum_incidence_df.iloc[time_to_eval, :].values[nocens], binary_target[nocens], threshold, as_dict)
except ValueError:
    if as_dict:
        EvaResy_at_T = dict()
    else:
        EvaResy_at_T = pd.DataFrame()
    print("Problem at time ", time_to_eval)

EvaResy_at_T["Time"] = [time_to_eval]
EvaResy_at_T["Censored"] = [np.sum(~nocens)]



    

# Formatting Results
PredData=PredData[1:,:]
PredData_df=pd.DataFrame(data=PredData,columns=np.append(['Prob','Pred','Outcome'],X_val_long_target_folds.columns))

PredData_df=PredData_df[np.logical_and(PredData_df['start2end_rolling']<max_time,PredData_df['DT_target']==0)==False]


PredData_df['Days']=PredData_df['Days'].astype(int)
YO=PredData_df.groupby(['Key_Days','Days','dead'])['Prob'].apply(lambda x: x.product()).reset_index()
DaytoDay_Eval=FeatEncV2.EvalMetrics(YO.Prob.astype(float),YO.dead.astype(int),0.5)

# TP            1631.000s
# FN             291.000
# FP            1188.000
# TN            3231.000
# MCC              0.536
# PrecPPV          0.579
# SensRecall       0.849
# SpecTNR          0.731
# NPR              0.917
# PRAUC            0.756
# ROC              0.877
# dtype: float64
DaytoDay_Eval['Day']=-1
for n  in range(1,PredData_df.Days.max()):
    CResult=FeatEncV2.EvalMetrics(YO.Prob.astype(float)[YO.Days==n],
                                  YO.dead.astype(int)[YO.Days==n],0.5)
    CResult['Day']=n
    DaytoDay_Eval=DaytoDay_Eval.append(CResult)
    
DaytoDay_Eval.reset_index()
DaytoDay_Eval=DaytoDay_Eval[1:19]
DaytoDay_Eval.mean()

# TP            26.859649
# FN             7.350877
# FP            24.736842
# TN            61.280702
# MCC            0.634772
# PrecPPV        0.708930
# SensRecall     0.832649
# SpecTNR        0.820737
# NPR            0.909263
# PRAUC          0.747158
# ROC            0.865702
# Day           29.000000
# dtype: float64

# %%
binary_metrics_df = pd.concat(binary_metrics)
binary_metrics_df.reset_index(drop=True)

# %%
binary_metrics_df[(binary_metrics_df.Val_fold == 4) & (binary_metrics_df.Time < 10)]

# %%
ax1 = binary_metrics_df[binary_metrics_df.Val_fold == 4].set_index(
    'Time')[["MCC","PRAUC", "F1", "AUC"]].plot()
ax1.set_ylim(0, 1)

# %%
surv_metrics_df = pd.concat(survival_metrics)
surv_metrics_df.reset_index(drop=True)

# %%
allmetrics = pd.merge(binary_metrics_df, surv_metrics_df)
# allmetrics.to_csv("./quick_results/all_metrics_90days.csv", index=False)
allmetrics

# %% Meta Learner
# I Tried this but it did not work very well I guess because it is not taking advantage of the discrete time approach
# Maybe adding the DT variable would help
meta_model_save = False
metaX_all = []
metay_all = []

for valfold, metaX_y in Fold_preds.items():
    metaX_all.append(metaX_y[0])
    metay_all.append(metaX_y[1])

metaX_all = np.hstack(metaX_all).T
metay_all = np.hstack(metay_all)

model_LR = sklearn.linear_model.LogisticRegression(solver='liblinear')
model_LR.fit(metaX_all, metay_all)


# %% Save models
folder_name = "featRA_models"

if not os.path.exists("./trained_models/{}".format(folder_name)):
    os.makedirs("./trained_models/{}".format(folder_name))

if save_models:
    for model_type, model in models.items():
        if 'lgb' in model_type:
            model.booster_.save_model("./trained_models/{folder}/{folder}-{model}.txt".format(
                folder=folder_name, model=model_type))#, num_iteration=model.best_iteration)

        if 'cbm' in model_type:
            model.save_model("./trained_models/{folder}/{folder}-{model}.cbm".format(
                folder=folder_name, model=model_type), format="cbm")
    
    if meta_model_save:
        pickle.dump(model_LR, open("./trained_models/{folder}/{folder}-{model}.pkl".format(
                folder=folder_name, model="LRmeta"), 'wb'))
        

# Save features names
np.savetxt("./trained_models/{folder}/columns_to_predict.txt".format(
    folder=folder_name), tomodel.columns, fmt='%s')


# %%
