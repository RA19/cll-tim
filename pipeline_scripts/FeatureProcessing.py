# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:57:15 2021

@author: RAGI0001
"""


import numpy as np

import pandas as pd
from scipy import stats
import sklearn
# import lifelines
# from lifelines import KaplanMeierFitter
import shap
import sklearn
from catboost import CatBoostClassifier, Pool, cv
import seaborn as sns
# import matplotlib.pyplot as plt
import scipy.stats
import os
# import graphviz
import datetime

import FeatEncV3

'''
File: 1_feature_engineering.py
File Created: 2020-12-01
Author: Adrian Gabriel Zucco (adrian.gabriel.zucco@regionh.dk)
'''

# %%

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from functools import reduce
import pickle
import re
import sys

# Custom Functions
import data_proc_funcs

# import dask as dd
# from dask.dataframe import from_pandas


def FeatureReduce_And_Join(CurrDataSource,windows_Labels,Thold):
    CurrFeat={}
    
    for w in windows_Labels:
        
        data=CurrDataSource[w].copy()
        print(data.shape)
        # Removing Null Columns
        null_columns=data.isna().sum()/np.shape(data)[0]
        to_del=null_columns[null_columns>Thold]
        data.drop(to_del.index, axis=1, inplace=True)
        print(data.shape)
        data.columns=[str(col)+'_'+w for col in data.columns]
        data['Key_Patient']=data['Key_Patient'+'_'+w].copy()
        del data['Key_Patient'+'_'+w]
        CurrFeat[w]=data.copy()
        
    FeatMat=CurrFeat[windows_Labels[0]].copy()
    for s in range(1,4):
        FeatMat = pd.merge(FeatMat, CurrFeat[windows_Labels[s]].copy(),
                                  how="outer", left_on='Key_Patient', right_on='Key_Patient')
    FeatMat.reset_index(inplace=True,drop=True)
    print(FeatMat.shape)
    return FeatMat



def serialize_dates_long(df, start_col, end_col, normalise = True):
    """
    Function to expand vertically a date range given a start and end date
    """
    if normalise:
        dates = [pd.date_range(r[0], r[1], normalize=True).to_series() 
        for r in df[[start_col, end_col]].values]
    
    lens = [len(x) for x in dates]

    df_long = pd.DataFrame( {col:np.repeat(df[col].values, lens) for col in df.columns})
    df_long.insert(1, 'date', np.concatenate(dates))

    return df_long

# %% LOAD DATA

with open('outcomes_censor1510_base_feat_splits.p','rb') as fp:
    OC=pickle.load(fp)
    

# %% ##################### PARAMETERS
censortime = '31-08-2020'

# %% Make table of all follow up days
OC_startend = OC[["Key_Patient", "first_event", "first_event_dttm",
                  "last_event", "last_event_dttm", "diff_start2end"]].drop_duplicates()
OC_startend_long = serialize_dates_long(
    OC_startend, 'first_event_dttm', 'last_event_dttm')

# Not expanding the first positive samples
# OC_startend_long = OC_startend.copy()
# OC_startend_long["date"] = OC_startend["first_event_dttm"].dt.floor("D")

OC_startend_long["Days_fromstart"] = OC_startend_long.groupby(
    "Key_Patient")["date"].rank(method="first", ascending=True).astype("int")

# %% Annotate days in hospital

OC_hosp = OC[["Key_Patient", "Admission_dttm",
              "Discharge_dttm"]].drop_duplicates().dropna()
OC_hosp_long = serialize_dates_long(
    OC_hosp, 'Admission_dttm', 'Discharge_dttm')
OC_hosp_long["Days_in_hosp"] = OC_hosp_long.groupby(
    ["Key_Patient", "Admission_dttm"])["date"].rank(method="first", ascending=True).astype("int")
OC_hosp_long["in_hosp"] = 1

# %% Annotate days in ICU

OC_ICU = OC[["Key_Patient", "ICU_STAY_START_dttm",
             "ICU_STAY_END_dttm"]].drop_duplicates().dropna()
# OC_ICU[ (OC_ICU["ICU_STAY_END_dttm"] - OC_ICU["ICU_STAY_START_dttm"]).dt.total_seconds() > 0 ]
OC_ICU_long = serialize_dates_long(
    OC_ICU, "ICU_STAY_START_dttm", "ICU_STAY_END_dttm")
OC_ICU_long["Days_in_ICU"] = OC_ICU_long.groupby(["Key_Patient", "ICU_STAY_START_dttm"])[
    "date"].rank(method="first", ascending=True).astype("int")
OC_ICU_long["in_ICU"] = 1

# %% Combine all tables
OC_s2e_hosp = pd.merge(OC_startend_long[["Key_Patient", "date", "Days_fromstart"]],
         OC_hosp_long[["Key_Patient", "date", "Days_in_hosp", "in_hosp"]], how="left")
OC_s2e_hosp_ICU = pd.merge(OC_s2e_hosp, OC_ICU_long[["Key_Patient", "date", "Days_in_ICU", "in_ICU"]], how="left")

OC_s2e_hosp

#### MEM CLEANING
# del OC_startend
del OC_startend_long
del OC_hosp_long
del OC_ICU_long

#%% Filter out the rolling days for the patients that were only positive
# TODO check that ICU overlaps with in_hosp
OC_firstpos_rollhosp = OC_s2e_hosp_ICU[(OC_s2e_hosp_ICU["Days_fromstart"] == 1.0) | 
    (OC_s2e_hosp_ICU["in_hosp"].notnull()) |
    (OC_s2e_hosp_ICU["in_ICU"].notnull())]

# %%
OC_firstpos_rollhosp.iloc[:,3:] = OC_firstpos_rollhosp.iloc[:,3:].fillna(0).astype(int)

#################### FEATURE ENGINEERING

#################### BASIC FEATURES
# %%
basic_feat = OC[['Key_Patient', 'sex_at_birth', 'age', 'BMI']
                ].drop_duplicates(subset=['Key_Patient'], keep='first')
basic_feat['sex_at_birth'] = pd.factorize(basic_feat['sex_at_birth'])[0]


windows=np.array([7,30,365,365*20])
windows_Labels=np.array(['W_7d','W_30d','W_1y','W_20y'])
    
with open('../AdmMed_daily_feat.p','rb') as fp:
    AdmMed_daily_feat=pickle.load(fp)

AdmMed_Feats=FeatureReduce_And_Join(AdmMed_daily_feat,windows_Labels,0.95)

with open('../OrdMed_daily_feat.p','rb') as fp:
    OrdMed_daily_feat=pickle.load(fp)

OrdMed_Feats=FeatureReduce_And_Join(OrdMed_daily_feat,windows_Labels,0.95)

with open('../Lab_daily_feat.p','rb') as fp:
    Lab_daily_feat=pickle.load(fp)

Lab_Feats=FeatureReduce_And_Join(Lab_daily_feat,windows_Labels,0.95)


with open('../Lab_daily_Cult_feat.p','rb') as fp:
    Lab_daily_Cult_feat=pickle.load(fp)

LabCult_Feats=FeatureReduce_And_Join(Lab_daily_Cult_feat,windows_Labels,0.95)

with open('../Vitals_daily_feat.p','rb') as fp:
    Vitals_daily_feat=pickle.load(fp)

Vitals_Feats=FeatureReduce_And_Join(Vitals_daily_feat,windows_Labels,0.95)

with open('../VitalsCategorical_daily_feat.p','rb') as fp:
    VitalsCategorical_daily_feat=pickle.load(fp)

VitalsCategor_Feats=FeatureReduce_And_Join(VitalsCategorical_daily_feat,windows_Labels,0.95)


with open('../Diagnosis_daily_feat.p','rb') as fp:
    Diagnosis_daily_feat=pickle.load(fp)

Diagnosis_Feats=FeatureReduce_And_Join(Diagnosis_daily_feat,windows_Labels,0.95)



with open('../Diagnosis_Hospital_daily_feat.p','rb') as fp:
    Diagnosis_Hospital_daily_feat=pickle.load(fp)

Diagnosis_Hospital_Feats=FeatureReduce_And_Join(Diagnosis_Hospital_daily_feat,windows_Labels,0.95)
Diagnosis_Hospital_Feats.columns=[str(col)+'_Hosp' for col in Diagnosis_Hospital_Feats.columns]
Diagnosis_Hospital_Feats['Key_Patient']=Diagnosis_Hospital_Feats['Key_Patient'+'_Hosp'].copy()
del Diagnosis_Hospital_Feats['Key_Patient'+'_Hosp']



with open('../Diagnosis_Outpatient_daily_feat.p','rb') as fp:
    Diagnosis_Outpatient_daily_feat=pickle.load(fp)

Diagnosis_Outpatient_Feats=FeatureReduce_And_Join(Diagnosis_Outpatient_daily_feat,windows_Labels,0.95)
Diagnosis_Outpatient_Feats.columns=[str(col)+'_Outpatient' for col in Diagnosis_Outpatient_Feats.columns]
Diagnosis_Outpatient_Feats['Key_Patient']=Diagnosis_Outpatient_Feats['Key_Patient'+'_Outpatient'].copy()
del Diagnosis_Outpatient_Feats['Key_Patient'+'_Outpatient']


with open('../ComScore.p','rb') as fp:
    ComScore_Feats=pickle.load(fp)


with open('../SmokingDrinking_daily_feat.p','rb') as fp:
    SmokingDrinking_daily_feat=pickle.load(fp)

SmokingDrinking_Feats=FeatureReduce_And_Join(SmokingDrinking_daily_feat,windows_Labels,0.95)


df_Matrix=[AdmMed_Feats,
             OrdMed_Feats,
             Lab_Feats,
             LabCult_Feats,
             Vitals_Feats,
             VitalsCategor_Feats,
             Diagnosis_Feats,
             Diagnosis_Hospital_Feats,
             Diagnosis_Outpatient_Feats,
             ComScore_Feats,
             SmokingDrinking_Feats]
df_Matrix=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df_Matrix) 

df_Matrix['Key_Patient_Days']=df_Matrix['Key_Patient'].copy()
df_Matrix[['Key_Patient','Days']]=df_Matrix['Key_Patient'].str.split('_',expand=True)

df_Matrix = pd.merge(df_Matrix, basic_feat, how="left",on='Key_Patient')
                       
OC_firstpos_rollhosp['Key_Patient_Days']= OC_firstpos_rollhosp['Key_Patient']+'_'+OC_firstpos_rollhosp['Days_fromstart'].astype(str)

df_Matrix = pd.merge(OC_firstpos_rollhosp, df_Matrix, how="left",on='Key_Patient_Days')


df_Matrix[['Key_Patient','Days']]=df_Matrix['Key_Patient_Days'].str.split('_',expand=True)
del df_Matrix['Key_Patient_x'],df_Matrix['date']

META=df_Matrix[['Days','Key_Patient','Key_Patient_Days', 'Days_fromstart', 'Days_in_hosp', 'in_hosp',
       'Days_in_ICU', 'in_ICU']].copy()

with open('COVIMUN_Matrix_N.p','wb') as fp:
    pickle.dump(df_Matrix,fp,protocol=pickle.HIGHEST_PROTOCOL)
    
    
with open('COVIMUN_META_Matrix_N.p','wb') as fp:
    pickle.dump(META,fp,protocol=pickle.HIGHEST_PROTOCOL)
    
    
Keys=pd.DataFrame()
Keys['Key_Patient']=META.Key_Patient.unique()

Death = pd.merge(Keys, OC[['Key_Patient','dead']], how="left",on='Key_Patient')
Death=Death.drop_duplicates()
META= pd.merge(META, Death, how="left",on='Key_Patient')

dead=META.dead.values
Dead=(df_Matrix[dead==1]).reset_index(drop=True)
NotDead=(df_Matrix[dead==0]).reset_index(drop=True)
#
#MannWhit=np.zeros((df_Matrix.shape[1],6))-9999
#MannWhit=pd.DataFrame(MannWhit)
#for n in range(24029,df_Matrix.shape[1]):
#    print(n)
#    if df_Matrix.dtypes[n]!='str':
#        if df_Matrix.dtypes[n]!='object':
#            if df_Matrix.columns[n]!='date':
#                Dead_dist=Dead[Dead.columns[n]].dropna().values
#                Alive_dist=NotDead[NotDead.columns[n]].dropna().values
#                if np.isnan(Alive_dist.mean())==False:
#                    if np.isnan(Dead_dist.mean())==False:
#                        if Alive_dist.mean()!=Dead_dist.mean():
#                            MannWhit[2][n],MannWhit[1][n]=scipy.stats.mannwhitneyu(Dead_dist,Alive_dist,alternative='two-sided')
#                            MannWhit[0][n]=Dead.columns[n]
#                            MannWhit[3][n]=np.nanmean(Dead_dist)
#                            MannWhit[4][n]=np.nanmean(Alive_dist)
#                            MannWhit[5][n]=len(Dead_dist)  +len(Alive_dist)
#                        else:
#                            MannWhit[0][n]=Dead.columns[n]
#                            MannWhit[3][n]=np.nanmean(Dead_dist)
#                            MannWhit[4][n]=np.nanmean(Alive_dist)
#                            MannWhit[5][n]=len(Dead_dist)  +len(Alive_dist)
#                            
#         
#MannWhit['Feature_Name']=MannWhit[0].values
#MannWhit['pvalue']     =MannWhit[1].values
#MannWhit['u-statistic']  =MannWhit[2].values
#MannWhit['Mean Value (For Patients (Dead))']    =MannWhit[3].values
#MannWhit['Mean Value (For Patients (Alive))'] =MannWhit[4].values
#MannWhit['Sample Size (Patient Days)'] =MannWhit[5].values
#del MannWhit[0],MannWhit[1],MannWhit[2],MannWhit[3],MannWhit[4],MannWhit[5]
#
#plt.close()
#plt.figure()
#
#FName='Active Days with 8 or more Unique Prescriptions (ATC level 2) (Ordered Med.)_W_20y'
#sns.distplot(Dead[FName].dropna().values,kde=False)
#sns.distplot(NotDead[FName].dropna().values,kde=False)
#plt.title(FName)

