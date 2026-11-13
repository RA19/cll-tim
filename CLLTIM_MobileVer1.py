# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:57:16 2019

@author: Rudi Agius
CLL-TIM Mobile Version v1.0 - 14-11-2024
https://www.nature.com/articles/s41467-019-14225-8
This version only considers CLL Baseline, Laboratory and Infection variables


"""

# Location Folders
CLLTIM_ScriptsDataFolder="CLLTIMData" # Where CLL-TIM Scripts and Data are Located
CLLTIM_OutputFolder="CLLTIMOutput"  # Where  you want your CLL-TIM Reports Located 
PatientCSV="PatientFiles\SynthPat01.csv" # Your PatientFile

# NOTE: your patient reports will be named according to 'Patient' Column in your patient file, and not the name of your Patient CSV file.



# Day of Prediction for CLL-TIM
PPointInd="FromPatientFile" # default Prediction Point taken from Patient File , but you can set as "Today".

import pickle
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from datetime import date
import seaborn as sns
from PyPDF2 import PdfFileMerger
import sklearn
import xgboost
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
warnings.filterwarnings(module='sklearn*', action='ignore', category=RuntimeWarning)
warnings.filterwarnings(module='sklearn*', action='ignore', category=UserWarning)


# SCALING AND NAN TO MEAN FUNCTIONS
def Nan2Mean(b):
    a = np.copy(b)
    # Nan to mean and Scaling
    col_mean = np.nanmean(a, axis=0)
    col_mean[np.isnan(col_mean)] = 0;
    if np.sum(~np.isfinite(col_mean)) > 0:
        col_cnt = 0
        for c in col_mean:
            if np.logical_or(str(c) == '-inf', str(c) == 'inf'):
                miny = np.nanmin(a[np.isfinite(a[:, col_cnt]), col_cnt])
                maxy = np.nanmax(a[np.isfinite(a[:, col_cnt]), col_cnt])
                Listy = np.argwhere(np.logical_and(~np.isfinite(a[:, col_cnt]), ~np.isnan(a[:, col_cnt])))
                for l in range(np.shape(Listy)[0]):
                    if str(a[Listy[l][0], col_cnt]) == '-inf':
                        a[Listy[l][0], col_cnt] = miny
                    elif str(a[Listy[l][0], col_cnt]) == 'inf':
                        a[Listy[l][0], col_cnt] = maxy
            col_cnt = col_cnt + 1
    col_mean = np.nanmean(a, axis=0)
    col_mean[np.isnan(col_mean)] = 0;
    # Find indicies that you need to replace
    inds = np.where(np.isnan(a))
    a[inds] = np.take(col_mean, inds[1])

    return a, col_mean


def Nan2Mean_useColmean(b, col_mean):
    a = np.copy(b)
    inds = np.where(np.isnan(a))
    a[inds] = np.take(col_mean, inds[1])
    return a




# Closing any open figures
plt.close('all')





##############################################################################
# READING PATIENT FILE AND CLL TIM DATA
##############################################################################

P01=pd.read_csv(PatientCSV,dtype = 'str')
patient_id =  P01['Patient'][0]

# loading translation variables
RCLabList=['na','alb','alp','inr','b2m','wbc','plat','rbc','hbm','urat','baso','trans','psa','smudge','amy','pblympho','mic','mcv','prom','crp','gluc','crea','creak','hema',
        'egfr','eos','ldh','neutro','mch','mchc','blast','myel','metamyel']
LabList_Orig=['NA+','ALB','ALP','INR','B2M','LEUK','THR','NPU01960','HAEM','NPU03688','BAS','NPU03607','NPU08669','NPU17597','NPU19653','LYM','NPU26631','NPU01944','NPU03974','CRP','GLUC','CRE','NPU19656','NPU01961',
        'EGFR','EOS','LDH','NEU','NPU02320','NPU02321','NPU03972','NPU03976','NPU03978']
RCBaselineList=['binet','ighv','ecog','famcll','agereg','Beta2m','cd38','zap70','sex','13q','tri12','11q','17p']
BaselineList_Orig=['binet','umut','WHOPERFORMANCE','FAMCLL','Age','beta2m','CD38','ZAP70','Gender','del13','tri12','del11','del17']

# Loading Dates to Int Conversion
Date2Int = pd.read_csv(CLLTIM_ScriptsDataFolder+'\Date2Int.csv', sep=',', float_precision='round_trip')


###############################################################################
# Data Loading
###############################################################################
TransList = np.load(CLLTIM_ScriptsDataFolder+'\TrainsListing.npy')
ToRemo = np.squeeze(np.argwhere(np.logical_and(P01['Variable_Category'] == 'lab', pd.isnull(P01['Value_of_Test']))))
P01 = P01.drop(ToRemo).reset_index()
P01.loc[P01.Value_of_Test.isnull(),'Value_of_Test']='NA'

# Date Conversions
DiagDate =np.argwhere(Date2Int['F1'] == P01['Diag_Date'][0])[0][0]

# Setting Predictoin Point either at PPoint in Patient File or either as today's date
today = date.today()
todayF1= today.strftime('%Y-%m-%d')
if PPointInd=="Today":
    PPoint = np.argwhere(Date2Int['F1'] == todayF1)[0][0]   
elif PPointInd=="FromPatientFile":
    PPoint =np.argwhere(Date2Int['F1'] == P01['Prediction_Point'][0])[0][0]

# Loading Feature Decoding Information and Training Data
LabbiesNames = pd.read_csv(CLLTIM_ScriptsDataFolder+'\LabTestFeatStruc.csv', sep=',',float_precision='round_trip')
FeatNamesMatrix = np.load(CLLTIM_ScriptsDataFolder+'\FeatNamesMatrix_7288.npy')
FeatListNames = np.load(CLLTIM_ScriptsDataFolder+'\FeatListNames_7288.npy')


# CLL-TIM DATA LOADING
Collect_CVData = np.load(CLLTIM_ScriptsDataFolder+'\_Collect_CVData.npy')
Collect_ModelNames = np.load(CLLTIM_ScriptsDataFolder+'\_Collect_ModelNames.npy')
CurrEnsSize = 28
ToKeep = np.load(CLLTIM_ScriptsDataFolder+'\_28_ToKeep.npy')
Collect_CVData_Chosen = Collect_CVData[ToKeep, :]
Collect_ModelNames_Chosen = Collect_ModelNames[ToKeep, :]
Collect_FeatIndies = np.load(CLLTIM_ScriptsDataFolder+'\CLL_TIM_FeatIndexes.npy',allow_pickle=True)
N_Chosen = len(ToKeep)
ALLFeats = np.load(CLLTIM_ScriptsDataFolder+'\ALLFeats_.npy')
FuncFeatNames = pd.read_csv(CLLTIM_ScriptsDataFolder+'\FeatureFunctionalNames.csv', sep='\t', header=None)
with open(CLLTIM_ScriptsDataFolder+"/vecgetcolmean.pkl","rb") as f:
    vecgetcolmeanALL=pickle.load(f)
f.close()

with open(CLLTIM_ScriptsDataFolder+"/vec_kmeans.pkl","rb") as f:
    X_Train_summaries=pickle.load(f)
f.close()

with open(CLLTIM_ScriptsDataFolder+"/vec_scaler.pkl","rb") as f:
    scalerfits=pickle.load(f)
f.close()


with open(CLLTIM_ScriptsDataFolder+"/vec_mx_all.pkl","rb") as f:
    vec_mx=pickle.load(f)
f.close()
 
with open(CLLTIM_ScriptsDataFolder+"/vec_mn_all.pkl","rb") as f:
    vec_mn=pickle.load(f)
f.close()

FAllInds = 0
for nn in range(0, CurrEnsSize):
    FAllInds = np.append(FAllInds, Collect_FeatIndies.item()[Collect_CVData_Chosen[nn, 0]])
FAInds = np.unique(FAllInds)




###############################################################################
# Setting Up Patient Feature Vector
###############################################################################
CT_FNM = FeatNamesMatrix[FAInds, :]
CT_FNM_Baseline = CT_FNM[CT_FNM[:, 1] == 'Baseline', :]
CT_FNM_Labka = np.copy(LabbiesNames)
CT_FNM_BC = CT_FNM[CT_FNM[:, 2] == 'BC', :]

###############################################################################
# Feature Extraction
###############################################################################


################################################################################
# Extracting baseline variables and creating baseline features

Baseline_P = np.zeros((1, 32))
# Going through Each of the Baseline Features Features and checking if they are in Patient Data
bcnt = -1
for CurrBVar in CT_FNM_Baseline[:, 0]:
    CB = CurrBVar.split('_')[0]
    bcnt = bcnt + 1

    CurrBVar_T = RCBaselineList[np.argwhere(CB == np.array(BaselineList_Orig))[0][0]]  # translated baseline name
    ConVal = -99
    if np.sum(CurrBVar_T == P01['Variable_Name']) > 0:

        Where = np.argwhere(CurrBVar_T == P01['Variable_Name'])[0][0]

        if 'binet' in CB:

            if '_A' in CurrBVar:
                if P01['Value_of_Test'][Where] == '1':
                    ConVal = 1
                else:
                    ConVal = 0
            if '_B' in CurrBVar:
                if P01['Value_of_Test'][Where] == '2':
                    ConVal = 1
                else:
                    ConVal = 0

            if '_C' in CurrBVar:
                if P01['Value_of_Test'][Where] == '3':
                    ConVal = 1
                else:
                    ConVal = 0


        elif 'Age' in CB:
            ConVal = np.int64(P01['Value_of_Test'][Where])/100

        elif 'WHOPERFORMANCE' in CB:

            if 'Can Walk' in CurrBVar:
                if P01['Value_of_Test'][Where] == '1': 
                    ConVal = 1
                else:
                    ConVal = 0
            if 'Good Condition' in CurrBVar:
                if P01['Value_of_Test'][Where] == '0':
                    ConVal = 1
                else:
                    ConVal = 0


        elif 'Gender' in CB:
            if P01['Value_of_Test'][Where] == '2': #female
                ConVal = 1
            elif P01['Value_of_Test'][Where] == '1': #male
                ConVal = 0
            else:
                ConVal = -99
        elif 'umut' in CB:
            if 'umut_1' in CurrBVar:

                if P01['Value_of_Test'][Where] == 'unmu':
                    ConVal = 1
                else:
                    ConVal = 0
            if 'umut_0' in CurrBVar:

                if P01['Value_of_Test'][Where] == 'mu':
                    ConVal = 1
                else:
                    ConVal = 0
            if 'umut_NA' in CurrBVar:

                if P01['Value_of_Test'][Where] == 'NA':
                    ConVal = 1
                else:
                    ConVal = 0
                    
        else: # includes del13,del11,tri12,del17
        
                
            if P01['Value_of_Test'][Where] == '0':
                ConVal = 1
            else:
                ConVal = 0
            if '_1' in CurrBVar:

                if P01['Value_of_Test'][Where] == '1':
                    ConVal = 1
                else:
                    ConVal = 0
                                    
            if '_NA' in CurrBVar:
                  if P01['Value_of_Test'][Where]=='nd':
                        ConVal=1
                  else:
                        ConVal=0
            if '_NA' in CurrBVar:
                  if P01['Value_of_Test'][Where]=='99':
                        ConVal=1
                  else:
                        ConVal=0
                        
    Baseline_P[0, bcnt] = ConVal


Baseline_P = np.squeeze(np.copy((Baseline_P)))
Baseline_P[Baseline_P == -99] = 0
if Baseline_P[2]==0:
    Baseline_P[3]=1
if np.logical_and(Baseline_P[4]==0,Baseline_P[5]==0):
    Baseline_P[6]=1
if np.logical_and(Baseline_P[7]==0,Baseline_P[8]==0):
    Baseline_P[9]=1
if np.logical_and(Baseline_P[10]==0,Baseline_P[11]==0):
    Baseline_P[12]=1
if np.logical_and(Baseline_P[15]==0,Baseline_P[16]==0):
    Baseline_P[17]=1

if np.logical_and(Baseline_P[21]==0,Baseline_P[22]==0):
    Baseline_P[23]=1
if np.logical_and(Baseline_P[24]==0,Baseline_P[25]==0):
    Baseline_P[26]=1

for n in range(np.shape(P01)[0]):
    if P01['Variable_Category'][n]=='lab':
        P01['Date_of_Test'][n] = np.argwhere(Date2Int['F1'] == P01['Date_of_Test'][n])[0][0]
        P01['Diag_Date'][n]
        P01['Date_of_Test'][n]
        
        
########## APPLYING UNIT CONVERSIONS #######################################

for tcnt in range(np.shape(P01)[0]):
    if P01.iloc[tcnt]['Variable_Category']=='lab':
        CurrDaty=P01.iloc[tcnt]['Date_of_Test']
        CurrVary=P01.iloc[tcnt]['Variable_Name']
        CurrValue=float(P01.iloc[tcnt]['Value_of_Test'])
        CurrUnit=P01.iloc[tcnt]['Units_of_Test']
        if CurrVary=='b2m':
            if CurrUnit=='mg_dL':
                P01.at[tcnt,'Value_of_Test']=str(CurrValue*10) # to mg_L
            elif CurrUnit=='mmol_L':
                P01.at[tcnt,'Value_of_Test']=str(CurrValue*1000000*0.0118) # to nmol_L to  mg_L
            elif CurrUnit=='nmol_L':
                P01.at[tcnt,'Value_of_Test']=str(CurrValue*0.0118) #  to  mg_L
     
                
        if CurrVary=='crp':
            if CurrUnit=='nmol_L':
                P01.at[tcnt,'Value_of_Test']=str(CurrValue/9.5238) # to mg_L
                
        if CurrVary=='hema':
            if CurrUnit=='percentage':
                P01.at[tcnt,'Value_of_Test']=str(CurrValue/100) # fraction
                
        if CurrVary=='hbm':
            if CurrUnit=='mg_dL':
                P01.at[tcnt,'Value_of_Test']=str((CurrValue/1000)*0.6206) # to g_dl to mmol_L
            if CurrUnit=='g_dL':
                P01.at[tcnt,'Value_of_Test']=str(CurrValue*0.6206) #to g_dl to mmol_L 
            if CurrUnit=="g_L":
                P01.at[tcnt,"Value_of_Test"]=str(CurrValue*0.1*0.6206) #g_L to g_dl to mmol_L 

        if CurrVary=="urat":
                if CurrUnit=="mcmol_L":
                    P01.at[tcnt,"Value_of_Test"]=str(CurrValue*0.001) # to mmol_L                   

        if CurrVary=="ldh":
                if CurrUnit=="mckat_L":
                    P01.at[tcnt,"Value_of_Test"]=str(CurrValue*60) # to ukat_l to U_L       

                
        if CurrVary=='trans':
            if CurrUnit=='g_L':
                P01.at[tcnt,'Value_of_Test']=str(CurrValue*12.5628) # mcmol_L
                
        if CurrVary=='neutro':
            if CurrUnit=='percentage':
                wr=np.argwhere(np.logical_and(P01['Variable_Name']=='wbc',P01['Date_of_Test']==CurrDaty)==1)[0][0]
                CurrWBC=float(P01.iloc[wr]['Value_of_Test'])
                P01['Variable_Name']=='wbc'
                P01.at[tcnt,'Value_of_Test']=str(CurrWBC*(CurrValue/100)) # to fraction to  10^9/L         
        
        if CurrVary=='baso':
            if CurrUnit=='percentage':
                wr=np.argwhere(np.logical_and(P01['Variable_Name']=='wbc',P01['Date_of_Test']==CurrDaty)==1)[0][0]
                CurrWBC=float(P01.iloc[wr]['Value_of_Test'])
                P01['Variable_Name']=='wbc'
                P01.at[tcnt,'Value_of_Test']=str(CurrWBC*(CurrValue/100)) # to fraction to 10^9/L        

                
        if CurrVary=="alp":
            if CurrUnit=="mckat_L":
                P01.at[tcnt,"Value_of_Test"]=str(CurrValue*60) # to ukat_l to U_L      

        if CurrVary=="amy":
            if CurrUnit=="mckat_L":
                P01.at[tcnt,"Value_of_Test"]=str(CurrValue*60.24) # to ukat_l to U_L      

        if CurrVary=="mchc":
            if CurrUnit=="g_L":
                P01.at[tcnt,"Value_of_Test"]=str(CurrValue*0.0621) # to g_L to mmolL   

        if CurrVary=="mch":
            if CurrUnit=="pg":
                P01.at[tcnt,"Value_of_Test"]=str(CurrValue*0.0621) # to pg to fmol      

        if CurrVary=="creak":
            if CurrUnit=="mckat_L":
                P01.at[tcnt,"Value_of_Test"]=str(CurrValue*60) # to ukat_l to U_L    
                

############################################################

# Adding beta2m at baseline
Entered=0
if np.sum(P01['Variable_Name']=='b2m')>0:
    b2ms=0
    Where = 'b2m' == P01['Variable_Name']
    VChunk = np.squeeze(np.column_stack((P01['Date_of_Test'][Where], P01['Value_of_Test'][Where])).astype(float))
    if np.sum(Where)==1:
        if abs(VChunk[0]-DiagDate)<=30:
            Entered=1
            b2ms=np.append(b2ms,VChunk[1])
    else:       
        for n in range(np.sum(Where)):
            
            if abs(VChunk[n,0]-DiagDate)<=30:
                Entered=1
                b2ms=np.append(b2ms,VChunk[n,1])
            
    if Entered==1:
        b2ms=np.nanmean(b2ms[1:])          
        if b2ms<4:
            Baseline_P[18]=1
        else:
            Baseline_P[19]=1
    
        
if np.logical_and(Baseline_P[18]==0,Baseline_P[19]==0):
    Baseline_P[20]=1
    
CT_FNM_Baseline[Baseline_P==1]
################################################################################

################################################################################
### Extracting Lab Variables and Creating Lab Features
Lab_P = np.empty((1, 120))
Lab_P[:] = np.nan

VV = np.copy(CT_FNM_Labka)

v = -1
lcnt = -1
for CL in CT_FNM_Labka[:, 0]:
    v = v + 1
    lcnt = lcnt + 1
    CurrLVar_T = RCLabList[np.argwhere(CL == np.array(LabList_Orig))[0][0]]  # translated baseline name
    ConVal = -99
    if np.sum(CurrLVar_T == P01['Variable_Name']) > 0:
        CurrFeatVal=np.nan
        ConVal = -99
        Where = CurrLVar_T == P01['Variable_Name']
        VChunk = np.squeeze(
            np.column_stack((P01['Date_of_Test'][Where], P01['Value_of_Test'][Where])).astype(float))

        if VV[v, 1] == 365:
            LBack = 365
        elif VV[v, 1] == 90:
            LBack = 90
        else:
            LBack = 365 * 20

        if np.size(VChunk) > 2:
            ToKeep1 = (PPoint - VChunk[:, 0]) >= 0  # No Future Points
            ToKeep2 = VChunk[:, 0] >= (PPoint - LBack)
        else:
            ToKeep1 = (PPoint - VChunk[0]) >= 0
            ToKeep2 = VChunk[0] >= (PPoint - LBack)

        ToKeep = np.logical_and(ToKeep1, ToKeep2)
        VChunk = VChunk[ToKeep, :]
        if len(VChunk) != 0:
            CLDates = VChunk[:, 0]
            CLValues = np.abs(VChunk[:, 1])
            Un_Dates_L = np.shape(CLDates)[0]
            CLDates=PPoint-CLDates.astype(int)
            
            if np.shape(VChunk)[0]>1:
                  indios=np.argsort(CLDates)
                  CLDates=CLDates[indios]
                  CLValues=CLValues[indios]         

            if VV[v, 2] == 'Instance':
                if VV[v, 3] == 'Cnt':
                    CurrFeatVal=np.shape(VChunk)[0] # we cnt identical  test on same day as multiple in here
                elif VV[v, 3] == 'latest':
                    CurrFeatVal = np.min(CLDates)
                elif VV[v, 3] == 'Mean':
                    CurrFeatVal = np.nanmean(CLDates)
            elif VV[v, 2] == 'nanmax':
                CurrFeatVal = np.nanmax(CLValues)
            elif VV[v, 2] == 'nanmin':
                CurrFeatVal = np.nanmin(CLValues)
            elif VV[v, 2] == 'nanmean':
                CurrFeatVal = np.nanmean(CLValues)
            elif VV[v, 2] == 'nanmedian':
                CurrFeatVal = np.nanmedian(CLValues)
            elif VV[v,2]=='nanstd':
                  if len(CLValues)!=1:
                        CurrFeatVal=np.nanstd(CLValues,ddof=1)   
                  else:
                        CurrFeatVal=0  
            elif VV[v,2]=='kurtosis':
                  if len(CLValues)!=1:
                        if len(np.unique(CLValues))==1:
                              CurrFeatVal=np.nan
                        else:                                    
                              CurrFeatVal=scipy.stats.kurtosis(CLValues,axis=0, fisher=False)       
                  else:
                        CurrFeatVal=np.nan
            elif VV[v,2]=='skewness':

                  if len(CLValues)!=1:
                        if len(np.unique(CLValues))==1:
                              CurrFeatVal=np.nan
                        else:
                              CurrFeatVal=scipy.stats.skew(CLValues)  
                  else:
                        CurrFeatVal=np.nan
            if  Un_Dates_L>=3:
                  if VV[v,2]=='kurtosis':
                        if len(np.unique(CLValues))==1:
                              CurrFeatVal=np.nan
                        else:
                              CurrFeatVal=scipy.stats.kurtosis(CLValues,axis=0, fisher=False)                
                  elif VV[v,2]=='skewness':
                        if len(np.unique(CLValues))==1:
                              CurrFeatVal=np.nan
                        else:
                              CurrFeatVal=scipy.stats.skew(CLValues)  
                              
                  elif VV[v,2]=='Slope1':
                  
                        if VV[v,3]=='coefa':
                              Slp1=np.polyfit(CLDates,CLValues,1)
                              CurrFeatVal=Slp1[0]
                        elif VV[v,3]=='coefb':
                              Slp1=np.polyfit(CLDates,CLValues,1)
                              CurrFeatVal=Slp1[1]
                                      
                  elif VV[v,2]=='Slope2':  
                        if VV[v,3]=='coefa':
  
                              Slp1=np.polyfit(CLDates,CLValues,2)
                              CurrFeatVal=Slp1[0]
                        elif VV[v,3]=='coefb':
                              
                              Slp1=np.polyfit(CLDates,CLValues,2)
                              CurrFeatVal=Slp1[1]
                        elif VV[v,3]=='coefc':
                              
                              Slp1=np.polyfit(CLDates,CLValues,2)
                              CurrFeatVal=Slp1[2]                              
                        
            if Un_Dates_L>=2:
                  if VV[v,2]=='kurtosis':
                        if len(np.unique(CLValues))==1:
                              CurrFeatVal=np.nan
                        else:
                              CurrFeatVal=scipy.stats.kurtosis(CLValues,axis=0, fisher=False)                      
                  elif VV[v,2]=='skewness':
                        if len(np.unique(CLValues))==1:
                              CurrFeatVal=np.nan
                        else:
                              CurrFeatVal=scipy.stats.skew(CLValues)  
                        
                        
            Lab_P[0,lcnt]=CurrFeatVal
            if CurrFeatVal>0:
                  stop=1

Lab_P = np.squeeze(np.copy(np.transpose(Lab_P)))
################################################################################



################################################################################
# Extracting Infection Dates and Applying Infection Features
################################################################################

for n in range(np.shape(P01)[0]):
    if P01['Variable_Category'][n]=='ii':
        P01['Date_of_Test'][n] = np.argwhere(Date2Int['F1'] == P01['Date_of_Test'][n])[0][0]
        P01['Diag_Date'][n]
        P01['Date_of_Test'][n]



InfecDates = np.unique(P01['Date_of_Test']['infec' == P01['Variable_Name']])

ToKeep1 = (PPoint - InfecDates) > 0  # No Future Points
Infec_90 = PPoint - InfecDates[np.logical_and(InfecDates >= (PPoint - 90), ToKeep1)]
Infec_365 = PPoint - InfecDates[np.logical_and(InfecDates >= (PPoint - 365), ToKeep1)]
Infec_Inf = PPoint - InfecDates[np.logical_and(InfecDates >= (PPoint - 365 * 20), ToKeep1)]

Infec_90 = np.int64(Infec_90.astype(None))
SInfec_90 = Infec_90[np.argsort(-Infec_90)]
Infec_365 = np.int64(Infec_365.astype(None))

SInfec_365 = Infec_365[np.argsort(-Infec_365)]
Infec_Inf = np.int64(Infec_Inf.astype(None))
SInfec_Inf = Infec_Inf[np.argsort(-Infec_Inf)]

# if no infections = zero
IFeats = np.zeros((12, 1))

if len(SInfec_Inf)>=3:
        Slp=np.polyfit(np.arange(1,len(SInfec_Inf)+1),SInfec_Inf,1)
        IFeats[0]=Slp[0] # slope1_coefa
  
if len(SInfec_90)>=3:
        Slp=np.polyfit(np.arange(1,len(SInfec_90)+1),SInfec_90,2)
        IFeats[1]=Slp[2] # slope2_coefc
  
if len(Infec_90>0):
        IFeats[2]=np.nanmean(Infec_90)
        IFeats[5]=np.nanmin(Infec_90)
        if len(Infec_Inf)!=1:
              if len(np.unique(CLValues))==1:
                    IFeats[7]=np.nan
              else:
                    IFeats[7]=scipy.stats.skew(Infec_90)  
        else:
              IFeats[7]=np.nan
if len(Infec_365)>0:
        IFeats[3]=np.nanmean(Infec_365)
        if len(Infec_365)!=1:
              IFeats[4]=np.nanstd(Infec_365,ddof=1)
        else:
              IFeats[4]=0
if len(Infec_Inf)>0:
        IFeats[6]=np.nanmax(Infec_Inf)
        if len(Infec_Inf)!=1:
              if len(np.unique(Infec_Inf))==1:
                    IFeats[8]=np.nan
              else:
                    IFeats[8]=scipy.stats.kurtosis(Infec_Inf,axis=0, fisher=False)
        else:
              IFeats[8]=np.nan
  
if len(SInfec_Inf)==2:
        Changes=SInfec_Inf[0]-SInfec_Inf[1]
        IFeats[9]=np.nanmin(Changes)
        IFeats[11]=np.nanmean(Changes)
elif len(SInfec_Inf)>2:
                    
        Changes=SInfec_Inf[0:-1]-SInfec_Inf[1:];
        IFeats[9]=np.nanmin(Changes)
        IFeats[11]=np.nanmean(Changes)
  
if len(SInfec_365)==2:
        Changes=SInfec_365[0]-SInfec_365[1]
        IFeats[10]=np.nanmean(Changes)
elif len(SInfec_365)>2:
                    
        Changes=SInfec_365[0:-1]-SInfec_365[1:];
        IFeats[10]=np.nanmean(Changes)
  
if len(Infec_90)==0:
        IFeats[1]=0
        IFeats[2]=0
        IFeats[5]=0
        IFeats[7]=0
elif len(Infec_90)==1:
        IFeats[7]=np.nan
        IFeats[1]=np.nan
elif len(Infec_90)==2:
        IFeats[1]=np.nan
        
if len(Infec_365)==0:
        IFeats[3]=0
        IFeats[4]=0
        IFeats[10]=0
     
elif len(Infec_365)==1:
        IFeats[10]=np.nan
       
if len(Infec_Inf)==0:
        IFeats[0]=0
        IFeats[6]=0
        IFeats[8]=0
        IFeats[9]=0
        IFeats[11]=0
elif len(Infec_Inf)==1:
        IFeats[8]=np.nan
        IFeats[0]=np.nan
        IFeats[9]=np.nan
        IFeats[11]=np.nan
elif len(Infec_Inf)==2:
        IFeats[0]=np.nan

Bw90=0
Bw365=0
BwInf=0
if np.sum(P01['Variable_Name']=='infec')!=0:
    if len(np.argwhere(P01['Variable_Name']=='infec'))==1:
          BOWInfec=pd.DataFrame(P01.iloc[np.argwhere(P01['Variable_Name']=='infec')[0][0]])
          BOWInfec=np.array(BOWInfec)
          if BOWInfec[7][0]=='0':
                if BOWInfec[6][0]>=(PPoint-90):
                      if (PPoint-BOWInfec[6][0])>0 :
                            Bw90=Bw90+1
                if BOWInfec[6][0]>=(PPoint-365):
                      if (PPoint-BOWInfec[6][0])>0 :
                            Bw365=Bw365+1
                if BOWInfec[6][0]>=(PPoint-365*20):
                      if (PPoint-BOWInfec[6][0])>0 :
                            BwInf=BwInf+1                    
else:
    
          BOWInfec=pd.DataFrame(P01.iloc[np.squeeze(np.argwhere(P01['Variable_Name']=='infec'))])
          BOWInfec=BOWInfec.reset_index()
          for sp in  range(0, np.shape(BOWInfec)[0]):
                
                if BOWInfec['Value_of_Test'][sp]=='0':
                      if BOWInfec['Date_of_Test'][sp]>=(PPoint-90):
                            if (PPoint-BOWInfec['Date_of_Test'][sp])>0 :
                                  Bw90=Bw90+1

                      if BOWInfec['Date_of_Test'][sp]>=(PPoint-365):
                            if (PPoint-BOWInfec['Date_of_Test'][sp])>0 :
                                  Bw365=Bw365+1

                      if BOWInfec['Date_of_Test'][sp]>=(PPoint-365*20):
                            if (PPoint-BOWInfec['Date_of_Test'][sp])>0 :
                                  BwInf=BwInf+1                              
                          
IFeats = np.squeeze(IFeats)
Infec_P = np.copy(np.transpose(IFeats))



################################################################################
#### CONCATENATING ALL FEATURES INTO 228  FEATURES IN ORDER REQUIRED
################################################################################
# note we do not use pathology and diagnosis codes in mobile version - hence np.zeros(64,)
OutputVal = np.concatenate((Baseline_P, np.zeros((64,)), Lab_P, Infec_P))  # Joining Feature Vectors (
OutputVal[32]=Bw90
OutputVal[33]=Bw365
OutputVal[34]=BwInf              
CheckFeats = np.column_stack((CT_FNM, OutputVal))

# hard coding Binet Missing as the median of Binet A
Bin=OutputVal[27] + OutputVal[28] + OutputVal[29]
if Bin==0:
    OutputVal[27]=1 

CheckFeats = np.column_stack((CT_FNM, OutputVal)) # USE THIS IF YOU WANT TO CHECK IF FEATURE VALUES MAKE SENSE WITH THE RAW DATA
# END OF FEATURE GENERATION #
################################################################################


   


################################################################################
# Generating 28 Base-learner Predictions
################################################################################
AllProbs = np.zeros((28, 1))
FinalFeats = np.zeros((1, 228))
RealFeats = np.zeros((1, 228))
RealFeatsU = np.zeros((1, 228))
FinalFeatsMat =np.zeros((28,228))-999


# Generating Base-learner Predictions
for nn in range(0, CurrEnsSize):
    print("Prediction for Baselearner " + str(nn))  
    # Loading Base Learner
    Curr_BL = pickle.load(open(CLLTIM_ScriptsDataFolder+'\CLL-TIM_BLN_' + str(nn) + '.sav', 'rb'))
    # Loading feature indices of base-learner
    FInds = Collect_FeatIndies.item()[Collect_CVData_Chosen[nn, 0]]
    fcnt = 0
    FeatsofBL = np.zeros((len(FInds))) - 99
    for ff in FInds:
        FeatsofBL[fcnt] = np.squeeze(np.argwhere(FAInds == ff))
        fcnt = fcnt + 1
    FeatsofBL = np.int64(FeatsofBL)
    VMatrix = np.row_stack((OutputVal, OutputVal))
    VMatrixU= VMatrix[:, FeatsofBL]
    VMatrix = VMatrix[:, FeatsofBL]

    if Collect_ModelNames_Chosen[nn, 1] != 'XGB':  # normalization if not XGBoost
        get_colmean = vecgetcolmeanALL[nn]
        VMatrix = Nan2Mean_useColmean(VMatrix, get_colmean)
        VMatrix= scalerfits[nn].transform(VMatrix)
        
        
    VMatrixNorm = (VMatrix - vec_mn[nn]) / (vec_mx[nn] - vec_mn[nn])
    Probs = Curr_BL.predict_proba(VMatrix)
    AllProbs[nn] = Probs[0, 1]

    if np.sum(Collect_ModelNames_Chosen[nn, 1] == np.array(['Elastic', 'LogisticRegression'])) > 0:
        X_train_summary=X_Train_summaries[nn]
        explainer = shap.KernelExplainer(Curr_BL.predict_proba, X_train_summary)
        shap_values = explainer.shap_values(VMatrix)
        shap_values = shap_values[0]

    elif np.sum(Collect_ModelNames_Chosen[nn, 1] == np.array(['XGB'])) > 0:
        explainer = shap.TreeExplainer(Curr_BL)
        shap_values = explainer.shap_values(VMatrix)
    elif np.sum(Collect_ModelNames_Chosen[nn, 1] == np.array(['RF','ExtraTreesClassifier'])) > 0:
        explainer = shap.TreeExplainer(Curr_BL)
        shap_values = explainer.shap_values(VMatrix)
        shap_values = shap_values[0]
        
    cnty = 0
    for s in TransList[FInds]:
        cind = np.int64(np.squeeze(np.argwhere(ALLFeats == s)))

        if cind.size > 0:
            FinalFeatsMat[nn,cind]=shap_values[0, cnty]
            FinalFeats[0, cind] = FinalFeats[0, cind] + shap_values[0, cnty]
            RealFeats[0, cind] = VMatrixNorm[0, cnty]
            RealFeatsU[0, cind] = VMatrixU[0,cnty]
            cnty = cnty + 1


 
            

        else:
            stop=1
            

ProbRisk = np.mean(AllProbs)  # Generating Mean Probabilistic Risk from CLL-TIM
ProbRisk=np.round(ProbRisk,2)
# END OF CLL-TIM Prediction



################################################################################
## Generation of Personalized Risk Factors
################################################################################
Toppies = ['', '', '', '', '', '', '', '', '', '']
ToppiesContribs = np.zeros((1, 10))
ToppiesRealVals = np.zeros((1, 10))
ToppiesRealValsU = np.zeros((1, 10))
TopInds = np.zeros((1, 10))
for n in range(np.shape(FinalFeats)[0]):

    Top = np.argsort(-1 * FinalFeats[n, :])
    TopVals = np.sort(-1 * FinalFeats[n, :])

    TopHR = np.array(FuncFeatNames[0][Top[0:5]]) 
    for rr in range(5):
        if FuncFeatNames[2][Top[rr]] == 1:
            if RealFeatsU[0, Top[rr]] == 0:
                TopHR[rr] = FuncFeatNames[1][Top[rr]]
        else:

            if np.isnan(RealFeatsU[0, Top[rr]]):
                TopHR[rr] = FuncFeatNames[1][Top[rr]]

    TopLR = np.array(FuncFeatNames[0][Top[-5:]])
    rrcnt = -1
    for rr in np.array([223, 224, 225, 226, 227]):
        rrcnt = rrcnt + 1
        if FuncFeatNames[2][Top[rr]] == 1:
            if RealFeatsU[0, Top[rr]] == 0:
                TopLR[rrcnt] = FuncFeatNames[1][Top[rr]]
        else:

            if np.isnan(RealFeatsU[0, Top[rr]]):
                TopLR[rrcnt] = FuncFeatNames[1][Top[rr]]

    Toppies = np.row_stack((Toppies, np.append(TopLR, TopHR)))
    ToppiesContribs = np.row_stack((ToppiesContribs, np.append(TopVals[-5:], TopVals[0:5])))
    ToppiesRealVals = np.row_stack((ToppiesRealVals, np.append(RealFeats[n, Top[-5:]], RealFeats[n, Top[0:5]])))
    ToppiesRealValsU = np.row_stack((ToppiesRealValsU, np.append(RealFeatsU[n, Top[-5:]], RealFeatsU[n, Top[0:5]])))
    TopInds = np.row_stack((TopInds, np.append(Top[-5:], Top[0:5])))

ToppiesRealVals = ToppiesRealVals[1:, :]
ToppiesRealValsU = ToppiesRealValsU[1:, :]
ToppiesContribs = ToppiesContribs[1:, :]
Toppies = Toppies[1:, :]
TopInds = np.int64(TopInds[1:, :])



################################################################################
## Generation of Reports
################################################################################

### Generating Personalized Risk Factors Visualization ####
p1 = ["silver", "olivedrab", "royalblue", "steelblue", "lightsalmon", "darkorange", 'firebrick', 'royalblue',
      'mediumspringgreen', 'aqua', 'saddlebrown', 'palevioletred', 'brown', 'olivedrab', 'black']
p2 = ['#1f77b4',
      '#ff7f0e',
      '#2ca02c',
      '#d62728',
      '#9467bd',
      '#8c564b',
      '#e377c2',
      '#7f7f7f',
      '#bcbd22',
      '#17becf']
pal = np.append(p1, p2)
import pylab as P
mks = ["o",
       "*",
       "^",
       "p",
       "s",
       "D",
       "h",
       "X",
       "v",
       '+']
for p in [0]:
    plt.close('all')
    plt.figure(dpi=1000)  #
    LW = 0.1
    plt.subplot(2, 1, 1)
    Flippy = np.flip(abs(ToppiesContribs[p, 5:]))
    Flippy = ToppiesContribs[p, 5:]
    HL = np.min(abs(ToppiesContribs[p, :])) - 0.0001
    plt.barh(0, Flippy[0], color=(1, 0, 0), height=0.3, edgecolor='black')

    P.arrow(0.001 + Flippy[0], 0, -Flippy[0] - HL - 0.01, 0, ec=(1, 0, 0), fc="black", width=0, head_width=0.1,
            head_length=HL)
    bbox = {'facecolor': 'lightgray', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
    plt.text(0, -0.2, Toppies[p, 5], {'ha': 'right', 'va': 'top', 'bbox': bbox}, rotation=0, color='black',
             fontsize=7)
    bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
    plt.text(0 - HL, 0, str(np.round(Flippy[0], 2)), {'ha': 'right', 'va': 'center', 'bbox': bbox}, rotation=0,
             color='black', fontsize=5)
    cnty = 0.2
    for n in range(1, 5):
       
        plt.barh(0, Flippy[n], left=np.sum(Flippy[0:n]), color=(1, cnty, cnty), height=0.3, edgecolor='black')
        P.arrow(0.001 + np.sum(Flippy[0:n]) + Flippy[n], 0, -Flippy[n] - HL - 0.01, 0, ec=(1, cnty, cnty),
                fc="black", width=0, head_width=0.1, head_length=HL)

        P.arrow(np.sum(Flippy[0:n]), -0.2, 0, (-0.2 * n), ec='black', fc="black", width=0.05, head_width=0.1,
                head_length=0)
        bbox = {'facecolor': 'lightgray', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
        plt.text(np.sum(Flippy[0:n]), (-0.2 * n) - 0.2, Toppies[p, n + 5],
                 {'ha': 'right', 'va': 'top', 'bbox': bbox}, rotation=0, color='black', fontsize=7)

        bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
        plt.text(np.sum(Flippy[0:n]) - HL, 0, str(np.round(Flippy[n], 2)),
                 {'ha': 'right', 'va': 'center', 'bbox': bbox}, rotation=0, color='black', fontsize=5)
        cnty = cnty + 0.2

    P.arrow(np.sum(Flippy), 0.37, -np.sum(Flippy) - 0.3, 0, ec='black', fc=(1, 0, 0), width=0.04, head_width=0.1,
            head_length=0.3, linewidth=0.1)
    plt.text(np.sum(Flippy) / 2, 0.5, 'Top High-Risk Contributors', {'ha': 'center', 'va': 'bottom'}, fontsize=5)

    Flippy = np.flip(abs(ToppiesContribs[p, 0:5]))
    plt.barh(0, Flippy[0], color=(0, 0, 1), height=0.3, edgecolor='black')
    HL = np.min(abs(ToppiesContribs[p, :])) - 0.0001
    P.arrow(0.001 + Flippy[0], 0, -Flippy[0] + HL - 0.01, 0, fc='black', ec=(0, 0, 1), head_width=0.1, width=0,
            head_length=HL, shape='full')
    
    bbox = {'facecolor': 'lightgray', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
    plt.text(0, +0.2, Toppies[p, n], {'ha': 'left', 'va': 'bottom', 'bbox': bbox}, rotation=0, color='black',
             fontsize=7)
    bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
    plt.text(0 + HL, 0, str(np.round(Flippy[0], 2)), {'ha': 'left', 'va': 'center', 'bbox': bbox}, rotation=0,
             color='black', fontsize=5)
    cnty = 0.2
    for n in range(1, 5):
       
 
        plt.barh(0, Flippy[n], left=np.sum(Flippy[0:n]), color=(cnty, cnty, 1), height=0.3, edgecolor='black')
        P.arrow(0.001 + Flippy[n] + np.sum(Flippy[0:n]), 0, -Flippy[n] + HL - 0.01, 0, fc='black',
                ec=(cnty, cnty, 1), width=0, head_width=0.1, head_length=HL, shape='full')
        P.arrow(np.sum(Flippy[0:n]), 0.2, 0, (0.2 * n), ec='black', fc="black", width=0.05, head_width=0.1,
                head_length=0)
        bbox = {'facecolor': 'lightgray', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
        plt.text(np.sum(Flippy[0:n]), (0.2 * n) + 0.2, Toppies[p, 4 - n],
                 {'ha': 'left', 'va': 'bottom', 'bbox': bbox}, rotation=0, color='black', fontsize=7)
        bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
        plt.text(np.sum(Flippy[0:n]) + HL, 0, str(np.round(Flippy[n], 2)),
                 {'ha': 'left', 'va': 'center', 'bbox': bbox}, rotation=0, color='black', fontsize=5)
        cnty = cnty + 0.2

    P.arrow(np.sum(Flippy), -0.37, -np.sum(Flippy) + 0.3, 0, ec='black', fc=(0, 0, 1), width=0.04, head_width=0.1,
            head_length=0.3, linewidth=0.1)
    plt.text(np.sum(Flippy) / 2, -0.5, 'Top Low-Risk Contributors', {'ha': 'center', 'va': 'top'}, fontsize=5)

    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')


Xinit = -70
if ProbRisk > 0.5:
    plt.text(Xinit, 1.5, 'Patient ' + str(patient_id)
             + ': Predicted as High-Risk of Infection or CLL Treatment in the next 2 years', color='red')

    if ProbRisk >= 0.58:
        nin = 0
        plt.text(Xinit, 1.2, 'Confidence Level is ' + str(np.round(ProbRisk, 3)) + ': High-Confidence')
    else:
        nin = 0
        plt.text(Xinit, 1.2, 'Confidence Level is ' + str(np.round(ProbRisk, 3)) + ': Low-Confidence')
else:
    plt.text(Xinit, 1.5, 'Patient ' + str(patient_id)
             + ': Predicted as Low-Risk of Infection or CLL Treatment in the next 2 years', color='blue')

    if ProbRisk < 0.28:
        nin = 0
        plt.text(Xinit, 1.2, 'Confidence Level is ' + str(np.round(ProbRisk, 3)) + ': High-Confidence')
    else:
        nin = 0
        plt.text(Xinit, 1.2, 'Confidence Level is ' + str(np.round(ProbRisk,3)) + ': Low-Confidence')

Yinit = 0.4
hh = 0.2
SF = 50

plt.barh(Yinit, SF * 0.28, left=Xinit + SF * 0, color='blue', height=hh, edgecolor='black')
plt.barh(Yinit, SF * 0.22, left=Xinit + SF * 0.28, color='lightblue', height=hh, edgecolor='black')
plt.barh(Yinit, SF * 0.08, left=Xinit + SF * 0.5, color='pink', height=hh, edgecolor='black')
plt.barh(Yinit, SF * 0.42, left=Xinit + SF * 0.58, color='red', height=hh, edgecolor='black')
if ProbRisk > 0.5:
    P.arrow(Xinit + SF * ProbRisk, Yinit + -(hh / 2), 0, hh + 0.3, ec='red', fc="red", width=0.01, head_width=0.5,
            head_length=0.05)
else:
    P.arrow(Xinit + SF * ProbRisk, Yinit + -(hh / 2), 0, hh + 0.3, ec='blue', fc="blue", width=0.01, head_width=0.5,
            head_length=0.05)
plt.text(Xinit + SF * 0.14, Yinit + -(hh / 2 + 0.05), 'High-Confidence', {'ha': 'center', 'va': 'top'}, fontsize=5,
         color='black')
plt.text(Xinit + SF * 0.45, Yinit + -(hh / 2 + 0.05), 'Low-Confidence ', {'ha': 'center', 'va': 'top'}, fontsize=5,
         color='black')
plt.text(Xinit + SF * 0.80, Yinit + -(hh / 2 + 0.05), 'High-Confidence ', {'ha': 'center', 'va': 'top'}, fontsize=5,
         color='black')
bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}

plt.text(Xinit + SF * 1, Yinit + (hh / 2 + 0.25), '1', {'ha': 'center', 'va': 'top'}, fontsize=8, color='black')
plt.text(Xinit + SF * 0.5, Yinit + (hh / 2 + 0.25), '0.5', {'ha': 'center', 'va': 'top'}, fontsize=8, color='black')
plt.text(Xinit + SF * 0, Yinit + (hh / 2 + 0.25), '0', {'ha': 'center', 'va': 'top'}, fontsize=8, color='black')

#
P.arrow(Xinit + SF * 0.5, Yinit + (hh / 2 + 0.07), SF * (0.5 - 0.05), 0, ec='red', fc="red", width=0.005,
        head_width=0.1, head_length=SF * 0.05)
P.arrow(Xinit + SF * 0.5, Yinit + (hh / 2 + 0.07), SF * (-0.5 + 0.05), 0, ec='blue', fc="blue", width=0.005,
        head_width=0.1, head_length=SF * 0.05)
P.arrow(Xinit + SF * 0.5, Yinit + (hh / 2 + 0.07), SF * 0, 0.05, ec='black', fc="black", width=0.005, head_width=0,
        head_length=0)
plt.text(Xinit + SF * ProbRisk, Yinit + 0.6, str(np.round(ProbRisk, 2)),
         {'ha': 'center', 'va': 'top', 'bbox': bbox}, fontsize=7, color='black')

bbox = {'facecolor': 'white', 'linewidth': LW, 'boxstyle': 'round,pad=0.2'}
plt.text(Xinit + SF * 0.9, Yinit + (hh / 2 + 0.2), 'High-Risk', {'ha': 'right', 'va': 'top', 'bbox': bbox},
         fontsize=5, color='Red')
plt.text(Xinit + SF * 0.1, Yinit + (hh / 2 + 0.2), 'Low-Risk', {'ha': 'left', 'va': 'top', 'bbox': bbox},
         fontsize=5, color='Blue')

plt.subplot(2, 1, 2)
plt.ylim(-1, 1)
plt.xticks([])
plt.yticks([])
plt.axis('off')
PCnt = 0
if ProbRisk > 0.5:
    plt.text(0, 1, 'Personalized High-Risk Factors for Patient: ' + str(patient_id), weight="bold")
    CurrXPos = 1
    for nnn in range(5, 10):
 
        CurrXPos = CurrXPos - 0.2
        PCnt = PCnt + 1
        if np.logical_or(FuncFeatNames[3][TopInds[0, nnn]] == 'B', np.isnan(RealFeatsU[0, TopInds[0, nnn]])):
            plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn])
        elif np.logical_and(FuncFeatNames[3][TopInds[0, nnn]] == 'C', RealFeatsU[0, TopInds[0, nnn]] == 0):
            plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn])
        else:
            plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str( np.round(ToppiesRealValsU[0, nnn], 1)))
else:

    plt.text(0, 1, 'Personalized Low-Risk Factors for Patient: ' + str(patient_id), weight="bold")
    CurrXPos = 1
    for nnn in range(4, -1, -1):
        CurrXPos = CurrXPos - 0.2
        PCnt = PCnt + 1
        if np.logical_or(FuncFeatNames[3][TopInds[0, nnn]] == 'B', np.isnan(RealFeatsU[0, TopInds[0, nnn]])):
           
            plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn])
        elif np.logical_and(FuncFeatNames[3][TopInds[0, nnn]] == 'C', RealFeatsU[0, TopInds[0, nnn]] == 0):
            plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn])
        else:
            plt.text(0, CurrXPos, str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(np.round(ToppiesRealValsU[0, nnn], 1)))

    
plt.savefig(CLLTIM_OutputFolder+"/"+patient_id+'.png', bbox_inches='tight')
plt.savefig(CLLTIM_OutputFolder+"/"+patient_id+'.pdf', bbox_inches='tight')
plt.close('all')


### Outputing personalized Risk Factors in terminal ###
PCnt = 0
if ProbRisk > 0.5:
    print('Patient ' + str(patient_id)
          + ': Predicted as High-Risk of Infection or CLL Treatment in the next 2-years: ' + str(ProbRisk))
    if ProbRisk >= 0.58:

        print('Note: CLL-TIM predicts this with High-Confidence')
    else:

        print('Note: CLL-TIM predicts this with Low-Confidence')

    print('')
    print('Personalized High-Risk Factors for Patient: ' + str(patient_id))
    for nnn in range(5, 10):
        PCnt = PCnt + 1
        if np.logical_or(FuncFeatNames[3][TopInds[0, nnn]] == 'B', np.isnan(RealFeatsU[0, TopInds[0, nnn]])):
            print(str(PCnt) + ': ' + Toppies[0, nnn])

        else:
            print(str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(ToppiesRealValsU[0, nnn]))


else:
    print('Patient ' + str(patient_id)
          + ': Predicted as Low-Risk of Infection or CLL Treatment in the next 2-years: '+ str(ProbRisk))

    if ProbRisk < 0.28:

        print('Note: CLL-TIM predicts this with High-Confidence')
    else:

        print('Note: CLL-TIM predicts this with Low-Confidence')
    print('')
    print('Personalized Low-Risk Factors for Patient: ' + str(patient_id))
    for nnn in range(4, -1, -1):

        PCnt = PCnt + 1
        if np.logical_or(FuncFeatNames[3][TopInds[0, nnn]] == 'B', np.isnan(RealFeats[0, TopInds[0, nnn]])):
            print(str(PCnt) + ': ' + Toppies[0, nnn])
        else:
            print(str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(ToppiesRealValsU[0, nnn]))


### Outputing personalized Risk Factors in text file ###
PReportFile=open(CLLTIM_OutputFolder+"/"+str(patient_id)+'_RiskFactors.txt','w')
PCnt = 0
PRFOut=''
if ProbRisk > 0.5:
    PReportFile.write('Patient ' + str(patient_id)
          + ': Predicted as High-Risk of Infection or CLL Treatment in the next 2-years\n')
    if ProbRisk >= 0.58:

        PReportFile.write('Note: CLL-TIM predicts this with High-Confidence\n')
    else:

        PReportFile.write('Note: CLL-TIM predicts this with Low-Confidence\n')

    PReportFile.write('\n')
    PReportFile.write('Personalized High-Risk Factors for Patient: ' + str(patient_id)+'\n')

else:
    PReportFile.write('Patient ' + str(patient_id)
          + ': Predicted as Low-Risk of Infection or CLL Treatment in the next 2-years'+'\n')

    if ProbRisk < 0.28:

        PReportFile.write('Note: CLL-TIM predicts this with High-Confidence'+'\n')
    else:

        PReportFile.write('Note: CLL-TIM predicts this with Low-Confidence'+'\n')
    PReportFile.write('\n')
    
PReportFile.write('\n')    
PCnt = 0    
PReportFile.write('Personalized High-Risk Factors for Patient: ' + str(patient_id)+'\n')
for nnn in range(5, 10):
    PCnt = PCnt + 1
    if np.logical_or(FuncFeatNames[3][TopInds[0, nnn]] == 'B', np.isnan(RealFeatsU[0, TopInds[0, nnn]])):
        PReportFile.write(str(PCnt) + ': ' + Toppies[0, nnn]+'\n')
        PRFOut=np.append(PRFOut,str(PCnt) + ': ' + Toppies[0, nnn]+'\n')
    else:
        PReportFile.write(str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(ToppiesRealValsU[0, nnn])+'\n' )
        PRFOut=np.append(PRFOut,str(PCnt)+ ': ' + Toppies[0, nnn] + ' with a value of ' + str(
            ToppiesRealValsU[0, nnn]) +'\n' )

PReportFile.write('\n')         
PCnt = 0
PReportFile.write('Personalized Low-Risk Factors for Patient: ' + str(patient_id)+'\n')
for nnn in range(4, -1, -1):

    PCnt = PCnt + 1
    if np.logical_or(FuncFeatNames[3][TopInds[0, nnn]] == 'B', np.isnan(RealFeatsU[0, TopInds[0, nnn]])):

        PReportFile.write(str(PCnt) + ': ' + Toppies[0, nnn]+'\n')
        PRFOut=np.append(PRFOut,str(PCnt) + ': ' + Toppies[0, nnn]+'\n')
    else:
        PReportFile.write(str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(
            ToppiesRealValsU[0, nnn]) +'\n')
        PRFOut=np.append(PRFOut,str(PCnt) + ': ' + Toppies[0, nnn] + ' with a value of ' + str(
            ToppiesRealValsU[0, nnn]) +'\n')

PReportFile.close()


# Generating probabilistic output of each baselearner
FFNtemp=np.copy(FinalFeatsMat)
FFNtemp[FFNtemp==-999]=0
Vmaxy=np.nanmax(FFNtemp)
Vminy=np.nanmin(FFNtemp)

fig=plt.figure(dpi=100,figsize=(15,5*3))
ax1=plt.subplot(3,1,1)
ax1.bar(np.arange(28),height=np.squeeze(AllProbs),align='edge')
ax1.hlines(0.5,0,28,ls='--')
ax1.hlines(0.58,0,28,colors='red',ls='--')
ax1.hlines(0.28,0,28,colors='green',ls='--')
ax1.hlines(ProbRisk,0,28,colors='blue')
ax1.set_xticklabels(['']*28)

# Generating heatmaps that show how much each top feature contirbuted to a given baselearner
# High Risk First
FinalFeatsSort=np.squeeze(np.argsort(FinalFeats))
IndVec=np.flipud(np.squeeze(FinalFeatsSort[-10:]))
SubSHAP=np.transpose(FinalFeatsMat[:,IndVec])
ylabelsy=FuncFeatNames[0].values[np.squeeze(IndVec)]
masky=SubSHAP==-999
SubSHAP[SubSHAP==-999]=0
ax3=plt.subplot(3,1,2,sharex=ax1)
f= sns.heatmap(SubSHAP[0:5,:],cmap='jet',cbar=None,center=0,square=False,linewidths=0.5,linecolor='grey',mask=masky[0:5,:],vmax=Vmaxy,vmin=Vminy,
        yticklabels=PRFOut[1:6])

# Low Risk Second
FinalFeatsSort=np.squeeze(np.argsort(FinalFeats))
IndVec=np.flipud(np.squeeze(FinalFeatsSort[0:10]))
SubSHAP=np.transpose(FinalFeatsMat[:,IndVec])
ylabelsy=FuncFeatNames[0].values[IndVec]
masky=SubSHAP==-999
SubSHAP[SubSHAP==-999]=0

ax2=plt.subplot(3,1,3,sharex=ax1)
g= sns.heatmap(SubSHAP[5:10,:],cmap='jet',cbar=None,center=0,square=False,linewidths=0.5,linecolor='grey',mask=masky[5:10,:],vmax=Vmaxy,vmin=Vminy,
        yticklabels=np.flipud(PRFOut[6:11]))


plt.savefig(CLLTIM_OutputFolder+ "/"+patient_id +'HeatmapsTop.png', bbox_inches='tight')
plt.savefig(CLLTIM_OutputFolder+"/" +patient_id+'HeatmapsTop.pdf', bbox_inches='tight')
plt.close('all')


# Merging PDFS
pdfs=[CLLTIM_OutputFolder+"/"+patient_id+'.pdf',CLLTIM_OutputFolder+"/"+patient_id+'HeatmapsTop.pdf']
merger=PdfFileMerger()
for pdf in pdfs:
    merger.append(pdf)
    
merger.write(CLLTIM_OutputFolder+"/"+patient_id+'_Report.pdf')


# END


