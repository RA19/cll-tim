# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 21:02:36 2020

@author: RAGI0001
"""


from scipy import stats

import seaborn as sns

import scipy.stats
import os
# import graphviz
import datetime


import FeatEncV3


import pyarrow
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from functools import reduce
import pickle
import re
import sys

os.chdir("/home/projects/reg_00012/people/rudagi/")
#sys.path.append('/home/projects/reg_00012/data/Ldrive_mirror/Scripts/useful_funcs/')

# Custom Functions
#from useful_funcs.data_proc_funcs import *
import data_proc_funcs

def ComScore_ICD10(CurrMatrix):
    Vascular=np.array(['DI872','DI890','DI652B', 'DI652', 'DI652A'   ,'DI800','DI801','DI802', 'DI803E', 'DT817C', 'DI26', 'DT817D'   ,'DI74'   
    'DI70'   ,'DI739A'   ,'DG458', 'DG459', 'DI63', 'DI64'  ,'DI21', 'DI22' , 'DI20'   ,'DI24', 'DI25'   , 'KFNG' ,'KAAL', 'KFNA',  
    'KFNC', 'KFNE', 'KFNF', 'KFNG', 'KPAE', 'KPAF', 'KPAH', 'KPAP', 'KPAQ', 'KPBE', 'KPBF', 'KPBH', 'KPBQ', 'KPCE', 'KPCH', 'KPCQ', 'KPCP', 'KPDE', 'KPDF', 'KPDH', 'KPDQ', 'KPDP', 'KPGH', 
     'KPEE', 'KPEF', 'KPEH', 'KPEN', 'KPEP', 'KPEQ', 'KPFE', 'KPFH', 'KPEQ' ])
    
    Upper_GI=np.array(['DK25','DK26','DK27','DK28' , 'DK227'   ,'DK85', 'DK860', 'DK861'   
    'DC15', 'DC16'   ,'DK251', 'DK252', 'DK255', 'DK256',   'DK261', 'DK262', 'DK265', 'DK266',   'DK271', 'DK272', 'DK275', 'DK276',   
    'DK281', 'DK282', 'DK285', 'DK286' , 'DK921'   ,'DK920'])
    
    Endocrine=np.array(['DE660F','DE660G','DE660H','DE662','DE660','DE660B','DE661','DE05','C50','C73','C74','E271','E272','E273','E274'])

    Com_Vascular=CurrMatrix.groupby(['Key_Patient']).apply(lambda x: np.int32(len(list(set(x['ICD10']) & set(Vascular)))>0))
    Com_Upper_GI=CurrMatrix.groupby(['Key_Patient']).apply(lambda x: np.int32(len(list(set(x['ICD10']) & set(Upper_GI)))>0))
    Com_Endocrine=CurrMatrix.groupby(['Key_Patient']).apply(lambda x: np.int32(len(list(set(x['ICD10']) & set(Endocrine)))>0))
    
    return Com_Vascular+Com_Upper_GI+Com_Endocrine


def ComScore_ATC(CurrMatrix):
    Vascular_Meds=np.array(['B01AA','B01AA','B01AC', 'B01AE', 'B01AF' ])
    Upper_GI_Meds=np.array(['A02BA','A02BC'])  
    Endocrine_Meds=np.array(['A10B','A10A','H03B','H03C']) 
    Com_Vascular=CurrMatrix.groupby(['Key_Patient']).apply(lambda x: np.int32(len(list(set(x['ATC_lev5']) & set(Vascular_Meds)))>0))
    Com_Upper_GI=CurrMatrix.groupby(['Key_Patient']).apply(lambda x: np.int32(len(list(set(x['ATC_lev5']) & set(Upper_GI_Meds)))>0))
    Com_Endocrine=CurrMatrix.groupby(['Key_Patient']).apply(lambda x: np.int32(len(list(set(x['ATC_lev4']) & set(Endocrine_Meds)))>0))
    
    return Com_Vascular+Com_Upper_GI+Com_Endocrine



def WindowGeneral(CurrData,TimeZeroColumn,LookBack):
    
    CurrData['ORDER_START_TIME']=(CurrData[TimeZeroColumn]-CurrData['ORDER_START_TIME']).astype('timedelta64[D]').astype(float)

    
    # Removing Medications that start after time-zero
    CurrData=CurrData[CurrData['ORDER_START_TIME']>=0]

    ######################################################
    # Applying Window
    CurrData=CurrData[CurrData['ORDER_START_TIME']<LookBack]
  
    return CurrData

def WindowMeds(MedMat_Clean_OC,dti_A,TimeZeroColumn,LookBack):
    dti=(MedMat_Clean_OC[TimeZeroColumn]-dti_A).astype('timedelta64[D]').astype(float)
    MedMat_Clean_OC['ORDER_START_TIME']=dti-MedMat_Clean_OC['ORDER_START_TIME']
    MedMat_Clean_OC['ORDER_END_TIME']=dti-MedMat_Clean_OC['ORDER_END_TIME']
     
     
    # Removing Medications that start after time-zero
    MedMat_Clean_OC=MedMat_Clean_OC[MedMat_Clean_OC['ORDER_START_TIME']>=0]
    # Setting Medicatoin end time to time-zero for medications that end after timezero.
    MedMat_Clean_OC['ORDER_END_TIME'][MedMat_Clean_OC['ORDER_END_TIME']<0]=0 # clipping end times at time =0.
    MedMat_Clean_OC=MedMat_Clean_OC.drop([TimeZeroColumn],axis=1).reset_index(drop=True)
    ######################################################
    
    # Applying Window
    MedMat_Clean_OC['ORDER_START_TIME'][MedMat_Clean_OC['ORDER_START_TIME']>LookBack]=LookBack
    MedMat_Clean_OC=MedMat_Clean_OC[MedMat_Clean_OC['ORDER_END_TIME']<LookBack]
    
    
    
    # Re-calculate Durations and Cumulative Dosages 
    MedMat_Clean_OC['Durations']=MedMat_Clean_OC['ORDER_START_TIME']-MedMat_Clean_OC['ORDER_END_TIME']
    
    # Calculating Cumulative Dosage Per Prescription Window
    MedMat_Clean_OC['CumDosage_PerWindow']=(MedMat_Clean_OC['Durations'])*(MedMat_Clean_OC['Dose'])
    MedMat_Clean_OC.sort_values("ORDER_START_TIME",inplace=True)
    MedMat_Clean_OC.reset_index(inplace=True,drop=True)
    return MedMat_Clean_OC



def WindowDiags(MedMat_Clean_OC,dti_A,TimeZeroColumn,LookBack):
    dti=(MedMat_Clean_OC[TimeZeroColumn]-dti_A).astype('timedelta64[D]').astype(float)
    MedMat_Clean_OC['ORDER_START_TIME']=dti-MedMat_Clean_OC['ORDER_START_TIME']
    MedMat_Clean_OC['ORDER_END_TIME']=dti-MedMat_Clean_OC['ORDER_END_TIME']
     
     
    # Removing Medications that start after time-zero
    MedMat_Clean_OC=MedMat_Clean_OC[MedMat_Clean_OC['ORDER_START_TIME']>=0]
    # Setting Medicatoin end time to time-zero for medications that end after timezero.
    MedMat_Clean_OC['ORDER_END_TIME'][MedMat_Clean_OC['ORDER_END_TIME']<0]=0 # clipping end times at time =0.
    MedMat_Clean_OC=MedMat_Clean_OC.drop([TimeZeroColumn],axis=1).reset_index(drop=True)
    ######################################################
    
    # Applying Window
    MedMat_Clean_OC['ORDER_START_TIME'][MedMat_Clean_OC['ORDER_START_TIME']>LookBack]=LookBack
    MedMat_Clean_OC=MedMat_Clean_OC[MedMat_Clean_OC['ORDER_END_TIME']<LookBack]
    
    
    
    # Re-calculate Durations
    MedMat_Clean_OC['Durations']=MedMat_Clean_OC['ORDER_START_TIME']-MedMat_Clean_OC['ORDER_END_TIME']
    
    MedMat_Clean_OC.sort_values("ORDER_START_TIME",inplace=True)
    MedMat_Clean_OC.reset_index(inplace=True,drop=True)
    return MedMat_Clean_OC
    



def PlotMeds(SubOrd,SubAdm):
    SubOrd['ATC']=SubOrd.ATC_lev7.copy()
    SubAdm['ATC']=SubAdm.ATC_lev7.copy()
    
    ATCies_Ord=SubOrd.ATC.unique()
    ATCies_Adm=SubAdm.ATC.unique()
    ATCNames=np.unique(np.append(ATCies_Ord,ATCies_Adm))
    Len_atc7=len(ATCNames)
        
    St='ORDER_START_TIME'
    Et='ORDER_END_TIME'
    
    x=SubOrd.copy()
    x[Et]=-np.round(x[Et],0).astype(int)
    x[St]=-np.round(x[St],0).astype(int)
    NotNan=x[St].isnull()==0
    NanET=x[Et].isnull()
    x[Et][NanET]=x[St][NanET]
    s1=abs(x[St].min().astype(int))
    RefMin=abs(np.nanmin(x[St]).astype(int))
    Maty=np.zeros((Len_atc7,abs(np.nanmin(x[St]).astype(int))+abs(np.nanmax(x[Et]).astype(int))))
    for atc in ATCNames:
        w=ATCNames==atc
           
        sties=(x[St][x.ATC==atc]+s1)[NotNan].values.astype(int)
        eties=(x[Et][x.ATC==atc]+s1)[NotNan].values.astype(int)
        for ncnt in range(len(eties)):
        
            Maty[w,sties[ncnt]:eties[ncnt]]=Maty[w,sties[ncnt]:eties[ncnt]]+1
    
    M1=np.shape(Maty)[0]
    M2=np.shape(Maty)[1]
    
    
      
    St='ORDER_START_TIME'
    Et='ORDER_END_TIME'
    
    x=SubAdm.copy()
    x[Et]=-np.round(x[Et],0).astype(int)
    x[St]=-np.round(x[St],0).astype(int)
    NotNan=x[St].isnull()==0
    NanET=x[Et].isnull()
    x[Et][NanET]=x[St][NanET]
    s1=abs(x[St].min().astype(int))
    
    Maty2=np.zeros((Len_atc7,M2))
    for atc in ATCNames:
        w=ATCNames==atc
           
        sties=(x[St][x.ATC==atc]+s1)[NotNan].values.astype(int)
        eties=(x[Et][x.ATC==atc]+s1)[NotNan].values.astype(int)
        for ncnt in range(len(eties)):
        
            Maty2[w,sties[ncnt]:eties[ncnt]]=Maty2[w,sties[ncnt]:eties[ncnt]]+1
    
    Maty2=np.fliplr(Maty2)
    
    
    
    import matplotlib.pyplot as plt
    plt.figure()
    ax=sns.heatmap(Maty,xticklabels=np.arange(-RefMin,-RefMin+np.shape(Maty)[1],step=1),cmap='Greens')
    ax.set_yticklabels(ATCNames,rotation='horizontal')
    ax.set_xticklabels(np.arange(-RefMin,-RefMin+np.shape(Maty)[1],step=1))
    if len(x)!=0:
        plt.scatter(np.argwhere(Maty2==1)[:,1],np.argwhere(Maty2==1)[:,0],marker="s",s=10,c="black")
    

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

basepath = "/home/projects/reg_00012/data/Ldrive_mirror/Data/EHR/SP_files_15102020/"

OC = pd.read_feather(basepath + "outcome_tables/outcomes_censor1510_base_feat_splits.feather")

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




# # FEATURES
# OrdMed = pd.read_feather(basepath + "clean_tables/"+"OrdineretMedicin.feather")
# AdmMed = pd.read_feather(basepath + "clean_tables/" + "AdministreretMedicin.feather")
# Lab = pd.read_feather(basepath + "clean_tables/"+"AlleProvesvar_filt.feather")
# Diag = pd.read_feather(basepath + "clean_tables/" + "Aktive_problemliste_diagnoser.feather")
# Diag_full = pd.read_feather(basepath + "clean_tables/" + "Diagnoses_combined.feather")
# Vitals = pd.read_feather(basepath + "clean_tables/" + "VitaleVaerdier.feather")
# Smokes = pd.read_feather(basepath + "clean_tables/" + "smoking_drinking.feather")

# ### META DATA
# meta_diag = pd.read_csv("L:\LovbeskyttetMapper\COVIMUN\Data/Metadata/diagnose_kode.tsv", sep="\t")

import datetime
######################3 WINDOWING & Feature Generation ###########################3
TimeZeroColumn='date'


windows=np.array([7,30,365,365*20])
windows_Labels=np.array(['W_7d','W_30d','W_1y','W_20y'])
################################################################
##                       Administered MEDICINES
#################################################################
AdmMed = pd.read_feather(basepath + "clean_tables/" + "AdministreretMedicin.feather")

AdmMed['ORDER_START_TIME']=AdmMed['TAKEN_TIME'].copy()
AdmMed['ORDER_END_TIME']=AdmMed['TAKEN_TIME']+datetime.timedelta(hours=24)
dti_Adm=AdmMed['ORDER_END_TIME'].max() # CHANGE
del AdmMed

AdmMed_Clean = pd.read_feather('/home/projects/reg_00012/people/rudagi/AdmMed_Clean.feather')
#AdmMed_Clean=AdmMed_Clean[0:1000].copy()

AdmMed_daily = pd.merge(OC_firstpos_rollhosp, AdmMed_Clean,
                          how="left", left_on='Key_Patient', right_on='Key_Patient')

AdmMed_daily['Key_Patient']=AdmMed_daily['Key_Patient']+'_'+AdmMed_daily['Days_fromstart'].astype(str)
#
#AdmMed_daily_W={}
#df_AM_W={}
#for w in range(len(windows)):
#    ForFeat=WindowMeds(AdmMed_daily.copy(),dti_Adm,TimeZeroColumn,windows[w])
#    df_AM=FeatEncV3.Enc_Med(ForFeat,' (Administered Med.)')
#    df_AM.reset_index(inplace=True,drop=True)
#    cols=df_AM.select_dtypes(include=[np.float64]).columns
#    df_AM[cols]=df_AM[cols].astype(pd.SparseDtype(float))
#    df_AM.reset_index(inplace=True,drop=True)
#    df_AM_W[windows_Labels[w]]=df_AM.copy()
#    
#with open('AdmMed_daily_feat.p','wb') as fp:
#    pickle.dump(df_AM_W,fp,protocol=pickle.HIGHEST_PROTOCOL)
#    
# with open('AdmMed_daily_feat.p','rb') as fp:
#     data=pickle.load(fp)
    



################################################################
##                       ORDERED MEDICINES
#################################################################
OrdMed = pd.read_feather(basepath + "clean_tables/"+"OrdineretMedicin.feather")
dti_Ord=OrdMed['ORDER_END_TIME'].max() # CHANGE
del OrdMed

OrdMed_Clean = pd.read_feather('/home/projects/reg_00012/people/rudagi/OrdMed_Clean.feather')
#OrdMed_Clean=OrdMed_Clean[0:1000].copy()
OrdMed_daily = pd.merge(OC_firstpos_rollhosp, OrdMed_Clean,
                          how="left", left_on='Key_Patient', right_on='Key_Patient')

OrdMed_daily['Key_Patient']=OrdMed_daily['Key_Patient']+'_'+OrdMed_daily['Days_fromstart'].astype(str)
#
#OrdMed_daily_W={}
#df_OM_W={}
#for w in range(len(windows)):
#    ForFeat=WindowMeds(OrdMed_daily.copy(),dti_Ord,TimeZeroColumn,windows[w])
#    df_OM=FeatEncV3.Enc_Med(ForFeat,' (Ordered Med.)')
#    df_OM.reset_index(inplace=True,drop=True)
#    cols=df_OM.select_dtypes(include=[np.float64]).columns
#    df_OM[cols]=df_OM[cols].astype(pd.SparseDtype(float))
#    df_OM.reset_index(inplace=True,drop=True)
#    df_OM_W[windows_Labels[w]]=df_OM.copy()
#    
#with open('OrdMed_daily_feat.p','wb') as fp:
#    pickle.dump(df_OM_W,fp,protocol=pickle.HIGHEST_PROTOCOL)

#
#################################################################
###                       Laboratory
##################################################################
#Lab = pd.read_feather(basepath + "clean_tables/"+"AlleProvesvar.feather")
##Lab_T=Lab[0:1000].copy()
#Lab_T['ORD_VALUE_num'][pd.to_numeric(Lab_T.ORD_VALUE,errors='coerce').isnull()]=np.nan
#Lab_T['ORD_VALUE_num']=Lab_T['ORD_VALUE_num'].astype(float)
#Lab_T['ORDER_START_TIME']=Lab_T['SPECIMN_TAKEN_TIME'].copy()
#
#Lab_daily = pd.merge(OC_firstpos_rollhosp, Lab_T, how="left",
#                         left_on='Key_Patient', right_on='Key_Patient')
#
#Lab_daily['Key_Patient']=Lab_daily['Key_Patient']+'_'+Lab_daily['Days_fromstart'].astype(str)
#
#
#df_Lab_W={}
#for w in range(len(windows)):
#    ForFeat=WindowGeneral(Lab_daily.copy(),TimeZeroColumn,windows[w])
#    df_Lab=FeatEncV3.Enc_Lab(ForFeat,' (Lab)')
#    df_Lab.reset_index(inplace=True,drop=True)
#    cols=df_Lab.select_dtypes(include=[np.float64]).columns
#    df_Lab[cols]=df_Lab[cols].astype(pd.SparseDtype(float))
#    df_Lab.reset_index(inplace=True,drop=True)
#    df_Lab_W[windows_Labels[w]]=df_Lab.copy()
#    
#with open('Lab_daily_feat.p','wb') as fp:
#    pickle.dump(df_Lab_W,fp,protocol=pickle.HIGHEST_PROTOCOL)
#
#
#
#################################################################
###                       Laboratory NPU17191
##################################################################
#Lab_daily_Cult=Lab_daily[Lab_daily['COMPONENT'].str.contains("NPU17191",case=False,na=False)].copy()
#
#dfy_CULT_W={}
#for w in range(len(windows)):
#    ForFeat=WindowGeneral(Lab_daily_Cult.copy(),TimeZeroColumn,windows[w])
#    dfy_CULT=FeatEncV3.NFeatCalc(ForFeat,'COMPONENT','','ORDER_START_TIME','ORDER_START_TIME','','Time_Basic')
#    dfy_CULT.reset_index(inplace=True,drop=True)
#    cols=dfy_CULT.select_dtypes(include=[np.float64]).columns
#    dfy_CULT[cols]=dfy_CULT[cols].astype(pd.SparseDtype(float))
#    dfy_CULT.reset_index(inplace=True,drop=True)
#    dfy_CULT_W[windows_Labels[w]]=dfy_CULT.copy()
#    
#with open('Lab_daily_Cult_feat.p','wb') as fp:
#    pickle.dump(dfy_CULT_W,fp,protocol=pickle.HIGHEST_PROTOCOL)
#
#
#
#
#################################################################
###                       VITALS
##################################################################
#
#Vitals = pd.read_feather(basepath + "clean_tables/" + "VitaleVaerdier.feather")
##Vitals_T=Vitals[0:1000].copy()
#Vitals_T['NumericValue'][pd.to_numeric(Vitals_T.NumericValue,errors='coerce').isnull()]=np.nan
#Vitals_T['ORD_VALUE_num']=Vitals_T['NumericValue'].astype(float)
#Vitals_T['ORDER_START_TIME']=Vitals_T['RECORDED_TIME'].copy()
#Vitals_T['EXTERNAL_NAME']=Vitals_T['DISPLAYNAME'].copy()
#Vit_daily = pd.merge(OC_firstpos_rollhosp, Vitals_T, how="left",
#                    left_on='Key_Patient', right_on='Key_Patient')        
#                    
#Vit_daily['Key_Patient']=Vit_daily['Key_Patient']+'_'+Vit_daily['Days_fromstart'].astype(str)
#
#
#df_Vitals_W={}
#for w in range(len(windows)):
#    ForFeat=WindowGeneral(Vit_daily.copy(),TimeZeroColumn,windows[w])
#    df_Vitals=FeatEncV3.Enc_Lab(ForFeat,' (Vitals)')
#    df_Vitals.reset_index(inplace=True,drop=True)
#    cols=df_Vitals.select_dtypes(include=[np.float64]).columns
#    df_Vitals[cols]=df_Vitals[cols].astype(pd.SparseDtype(float))
#    df_Vitals.reset_index(inplace=True,drop=True)
#    df_Vitals_W[windows_Labels[w]]=df_Vitals.copy()
#    
#with open('Vitals_daily_feat.p','wb') as fp:
#    pickle.dump(df_Vitals_W,fp,protocol=pickle.HIGHEST_PROTOCOL)
#
#
#
#################################################################
###                       VITALS CATEGORICAL
##################################################################
#
#Vit_daily_Categ=Vit_daily[Vit_daily['NumericValue'].isnull()]
#Vit_daily_Categ=Vit_daily_Categ[Vit_daily_Categ['MEAS_VALUE_clean'].isnull()==0]
#Vit_daily_Categ['CategVal']=Vit_daily_Categ['DISPLAYNAME']+'_'+Vit_daily_Categ['MEAS_VALUE_clean'].str.split(';').str[0]
#
#df_VitalsC_W={}
#for w in range(len(windows)):
#    ForFeat=WindowGeneral(Vit_daily_Categ.copy(),TimeZeroColumn,windows[w])
#    df_VitalsC=FeatEncV3.Enc_VitalsC(ForFeat,' (Vitals)')
#    df_VitalsC.reset_index(inplace=True,drop=True)
#    cols=df_VitalsC.select_dtypes(include=[np.float64]).columns
#    df_VitalsC[cols]=df_VitalsC[cols].astype(pd.SparseDtype(float))
#    df_VitalsC.reset_index(inplace=True,drop=True)
#    df_VitalsC_W[windows_Labels[w]]=df_VitalsC.copy()
#    
#with open('VitalsCategorical_daily_feat.p','wb') as fp:
#    pickle.dump(df_VitalsC_W,fp,protocol=pickle.HIGHEST_PROTOCOL)
#
#
#


################################################################
##                       Diagnosis
#################################################################


Diag_full = pd.read_feather(basepath + "clean_tables/" + "Diagnoses_combined.feather")
Diag_full['CURRENT_ICD10_LIST']=Diag_full['SKSKode'].copy()
#Diag_full=Diag_full[0:1000].reset_index()
dti_Diag=Diag_full['SlutTid'].max() # CHANGE
del Diag_full

Diag_T=pd.read_feather('/home/projects/reg_00012/people/rudagi/Diag_Clean.feather')


Diag_daily = pd.merge(OC_firstpos_rollhosp, Diag_T,
                          how="left", left_on='Key_Patient', right_on='Key_Patient')

# All Diagnoses
Diag_daily['Key_Patient']=Diag_daily['Key_Patient']+'_'+Diag_daily['Days_fromstart'].astype(str)
df_Diag_W={}
for w in range(len(windows)):
    ForFeat=WindowDiags(Diag_daily.copy(),dti_Diag,TimeZeroColumn,windows[w])
    df_Diag=FeatEncV3.Enc_Diag(ForFeat,' ( Diag.)')
    df_Diag.reset_index(inplace=True,drop=True)
    cols=df_Diag.select_dtypes(include=[np.float64]).columns
    df_Diag[cols]=df_Diag[cols].astype(pd.SparseDtype(float))
    df_Diag.reset_index(inplace=True,drop=True)
    df_Diag_W[windows_Labels[w]]=df_Diag.copy()
    with open('Diagnosis_daily_feat_'+windows_Labels[w]+'.p','wb') as fp:
        pickle.dump(df_Diag,fp,protocol=pickle.HIGHEST_PROTOCOL)
with open('Diagnosis_daily_feat.p','wb') as fp:
    pickle.dump(df_Diag_W,fp,protocol=pickle.HIGHEST_PROTOCOL)

# Outpatient Diagnosis
Diag_daily_Outpatient = Diag_daily[Diag_daily['BesogsType']=='Ambulant']
df_Diag_W={}
for w in range(len(windows)):
    ForFeat=WindowDiags(Diag_daily_Outpatient.copy(),dti_Diag,TimeZeroColumn,windows[w])
    df_Diag=FeatEncV3.Enc_Diag(ForFeat,' ( Outpatient Diag.)')
    df_Diag.reset_index(inplace=True,drop=True)
    cols=df_Diag.select_dtypes(include=[np.float64]).columns
    df_Diag[cols]=df_Diag[cols].astype(pd.SparseDtype(float))
    df_Diag.reset_index(inplace=True,drop=True)
    df_Diag_W[windows_Labels[w]]=df_Diag.copy()
    with open('Diagnosis_Outpatient_daily_feat_'+windows_Labels[w]+'.p','wb') as fp:
        pickle.dump(df_Diag,fp,protocol=pickle.HIGHEST_PROTOCOL)    
with open('Diagnosis_Outpatient_daily_feat.p','wb') as fp:
    pickle.dump(df_Diag_W,fp,protocol=pickle.HIGHEST_PROTOCOL)

# Hospitalized Diagnosis
Diag_daily_Hospital = Diag_daily[Diag_daily['BesogsType']=='Indlagt']

df_Diag_W={}
for w in range(len(windows)):
    ForFeat=WindowDiags(Diag_daily_Hospital.copy(),dti_Diag,TimeZeroColumn,windows[w])
    df_Diag=FeatEncV3.Enc_Diag(ForFeat,' ( Hospital Diag.)')
    df_Diag.reset_index(inplace=True,drop=True)
    cols=df_Diag.select_dtypes(include=[np.float64]).columns
    df_Diag[cols]=df_Diag[cols].astype(pd.SparseDtype(float))
    df_Diag.reset_index(inplace=True,drop=True)
    df_Diag_W[windows_Labels[w]]=df_Diag.copy()
    with open('Diagnosis_Hospital_daily_feat_'+windows_Labels[w]+'.p','wb') as fp:
        pickle.dump(df_Diag,fp,protocol=pickle.HIGHEST_PROTOCOL)     
with open('Diagnosis_Hospital_daily_feat.p','wb') as fp:
    pickle.dump(df_Diag_W,fp,protocol=pickle.HIGHEST_PROTOCOL)





ComICD10=pd.DataFrame(ComScore_ICD10(Diag_daily))
ComICD10['ICD10_Comorbidity_Points']=ComICD10[0].copy()
del ComICD10[0]

AdmMed_daily_1yr=WindowMeds(AdmMed_daily.copy(),dti_Adm,TimeZeroColumn,365)
OrdMed_daily_1yr=WindowMeds(OrdMed_daily.copy(),dti_Ord,TimeZeroColumn,365)


ComATC_Adm=pd.DataFrame(ComScore_ATC(AdmMed_daily_1yr))
ComATC_Adm['Adm_Comorbidity_Points']=ComATC_Adm[0].copy()
del ComATC_Adm[0]
ComATC_Ord=pd.DataFrame(ComScore_ATC(OrdMed_daily_1yr))
ComATC_Ord['Ord_Comorbidity_Points']=ComATC_Ord[0].copy()
del ComATC_Ord[0]

ComScore=pd.merge(ComATC_Ord,ComATC_Adm,on='Key_Patient',how='outer')
ComScore=pd.merge(ComScore,ComICD10,on='Key_Patient',how='outer')
ComScore['Adm_ICD_Score']=ComScore['Adm_Comorbidity_Points']+ComScore['ICD10_Comorbidity_Points']
ComScore['Ord_ICD_Score']=ComScore['Ord_Comorbidity_Points']+ComScore['ICD10_Comorbidity_Points']
ComScore['Adm_Ord_ICD_Score']=ComScore['Adm_Comorbidity_Points']+ComScore['Ord_Comorbidity_Points']+ComScore['ICD10_Comorbidity_Points']






#
#
## Med Visualization
#SubOrd=OrdMed_Clean_daily_Windowed[OrdMed_Clean_daily_Windowed['Days_fromstart']==5]
#SubAdm=AdmMed_Clean_daily_Windowed[AdmMed_Clean_daily_Windowed['Days_fromstart']==5]
#
#PlotMeds(SubOrd,SubAdm)

