# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 08:06:09 2021

@author: RAGI0001
"""

import pandas as pd
import datetime
import numpy as np

def DiagICDGroup(DiagTemp):
    
    DiagTemp['isnumeric']=DiagTemp["ICD10"].str[2].str.isnumeric()

 
    icd10sub=pd.read_csv('/home/projects/reg_00012/people/rudagi/icd10sub.csv')
    # ToRemove=np.array(['B10','B20','B99','C50','C7A','C7B','D3A','D49','D78','E89','E36','F99',
    # 'H59','H95','E36','J95','L76','M95','M96','M99','N99','O09','P09','P84','R97','R99','T07','T14','T79',
    # 'Z16','Z17','Z18','Z66','Z67','Z68','9A0'])
    ToRemove=np.array(['C7A-C7A','C7B-C7B','D3A-D3A','9A0-9A0'])
    
    for t in ToRemove:
        icd10sub=icd10sub[icd10sub['Code Range']!=t]
    
    
    # ToRemove_DiagTemp=np.array(['DO992','DO993A','DO998','DVA10','DO993A1','DO993A3','DO995A','DO985','DO993B'
    #                         ,'DO996','DO992B','DUA35','DUH37','DO993B5','DO990A','DO992C','DO993B3','DVRA01'
    #                         ,'DVRA02A','DVRA10','DVRB02','DVRB51','DVRK01','DUA31','DUH35','DA979','DO991',
    #                         'DO994','DO984','DO993B2','DO991D','DO991B','DUA32','DUH33'])
    # for t in ToRemove_DiagTemp:
    #     DiagTemp=DiagTemp[DiagTemp['ICD10']!=t]
    
    
    DiagTemp['ICD10_Group']=''
    DiagTemp['ICD10_GroupDescription']=''
    cnt=-1
    for D in DiagTemp.ICD10:
        if cnt%1000==0:
          print(cnt)
        cnt=cnt+1
        if DiagTemp['isnumeric'][cnt]==False:
            DiagTemp['ICD10_Group'][cnt]=D
            DiagTemp['ICD10_GroupDescription'][cnt]=D
        else:
            Sub=icd10sub['Code Range'][icd10sub['Code Range'].str.contains(D[1])]
            SubNames=icd10sub['Section Description'][icd10sub['Code Range'].str.contains(D[1])]
            # print(D)
            Nummy=int(D[2:5])
            # print(Sub)
            for n in range(len(Sub)):
                if Sub.str.split('-').iloc[n][0].find(D[1])!=-1:
                    LB=int(Sub.str.split('-').iloc[n][0].split(D[1])[1]+'0')
                    if (n+1)!=len(Sub):
                        UB=int(Sub.str.split('-').iloc[n+1][0].split(D[1])[1]+'9')
                    else:
                        if len(Sub.str.split('-').iloc[n])==2:
                            UB=int(Sub.str.split('-').iloc[n][1].split(D[1])[1]+'9')
                        else:
                            UB=int(Sub.str.split('-').iloc[n][0].split(D[1])[1]+'9')               
                    if UB!=LB:
                        if np.logical_and(Nummy >= LB,Nummy <= UB):
                            
                            # print(D+' in ' +Sub.iloc[n] +' '+str(cnt))
                            DiagTemp['ICD10_Group'][cnt]=Sub.iloc[n]
                            DiagTemp['ICD10_GroupDescription'][cnt]=SubNames.iloc[n]
                            break
                    else:
                        if Nummy >= LB:
                            # print(D+' in ' +Sub.iloc[n]+' '+str(cnt))
                            DiagTemp['ICD10_Group'][cnt]=Sub.iloc[n]
                            DiagTemp['ICD10_GroupDescription'][cnt]=SubNames.iloc[n]
                            break    
                else:
                    UB=int(Sub.str.split('-').iloc[n][1].split(D[1])[1]+'9')
                    if Nummy <=UB:
                            # print(D+' in ' +Sub.iloc[n]+' '+str(cnt))
                            DiagTemp['ICD10_Group'][cnt]=Sub.iloc[n]
                            DiagTemp['ICD10_GroupDescription'][cnt]=SubNames.iloc[n]
                            break
      
        
    return DiagTemp

def MedCleaner(MedMat,dti):


    SameDayCollapse=1 # To COLLAPSE Instances ON SAME DAY, and take SUM value
    
    # Setting Time Relative to dti (reference date)
    MedMat['ORDER_START_TIME']=(MedMat['ORDER_START_TIME']-dti).astype('timedelta64[D]').astype(float)
    MedMat['ORDER_END_TIME']=(MedMat['ORDER_END_TIME']-dti).astype('timedelta64[D]').astype(float)  
    MedMat['Dose'][pd.to_numeric(MedMat.Dose,errors='coerce').isnull()]=np.nan
    MedMat['Dose']=MedMat['Dose'].astype(float)
    
     
    #  Step.1 COLLAPSE Instances ON SAME DAY, and take SUM value
    if SameDayCollapse==1:
        MedMat=MedMat.groupby(['Key_Patient','ORDER_START_TIME','ORDER_END_TIME','ATC'],as_index=False)['Dose'].sum().reset_index()
    
    
    # Step.2 Replacing dosage values of 1 with the median dosage for that same medication
    YO=MedMat.groupby(['ATC','Key_Patient'],group_keys=False).apply(lambda x: x['Dose'].replace(1,np.nanmedian(x['Dose'][x['Dose']!=1].values)))
    MedMat['DoseN']=YO.copy()
    # and Replacing dosage values of 2 with the 2 * median dosage for that same medication
    YO=MedMat.groupby(['ATC','Key_Patient'],group_keys=False).apply(lambda x: x['DoseN'].replace(2,2*np.nanmedian(x['DoseN'][x['DoseN']!=2].values)))
    MedMat['DoseN']=YO.copy()
    # and Replacing dosage values of 0.5 with the 0.5* median dosage for that same medication
    YO=MedMat.groupby(['ATC','Key_Patient'],group_keys=False).apply(lambda x: x['DoseN'].replace(0.5,0.5*np.nanmedian(x['DoseN'][x['DoseN']!=0.5].values)))
    MedMat['DoseN']=YO.copy()
    MedMat['DoseN'][MedMat['DoseN'].isnull()]=MedMat['Dose'][MedMat['DoseN'].isnull()] 
    MedMat['Dose']=MedMat['DoseN'].copy()
    MedMat=MedMat.drop(['DoseN'],axis=1).reset_index(drop=True)
    
    # Step. 3 Putting NaN dosages as median for that medication given the patient
    YO=MedMat.groupby(['ATC','Key_Patient'],group_keys=False).apply(lambda x: x['Dose'].fillna(np.nanmedian(x['Dose'][x['Dose'].isnull()==0].values)))
    MedMat['Dose']=YO.copy()
    # Step. 3 if still NaN, put as median for the population
    YO=MedMat.groupby(['ATC'],group_keys=False).apply(lambda x: x['Dose'].fillna(np.nanmedian(x['Dose'][x['Dose'].isnull()==0].values)))
    MedMat['Dose']=YO.copy()
    ######################################################  
    MedMatOrig=MedMat.copy()
    
                                 
                                    
    ################# Medication Duration Cleaning ####################    
    # Step1. This Code wil join any medications which are re-continued.
    # re-continued is defined as starting the same medication, with same dosage
    # on the day of, or the day after the previous prescriptoin finished. 
    # If a new Dosage is given, they are not joined together. 
    MedMat.sort_values("ORDER_START_TIME",inplace=True)
    print(len(MedMat))
    prevlen=0
    MedMat['group']=MedMat.groupby(["Key_Patient","ATC",'Dose'],group_keys=False).apply(lambda x:(x.ORDER_START_TIME > (1+x.ORDER_END_TIME.shift())).cumsum())
    result=MedMat.groupby(["Key_Patient","ATC","group",'Dose']).agg({'ORDER_START_TIME': 'min',"ORDER_END_TIME": 'max'}).reset_index()
    print(len(result))
    while len(result)!=prevlen:
       prevlen=len(result)
       CG=result[["Key_Patient","ATC",'Dose','ORDER_START_TIME','ORDER_END_TIME']]
       CG.sort_values("ORDER_START_TIME",inplace=True)
       CG['group']=CG.groupby(["Key_Patient","ATC",'Dose'],group_keys=False).apply(lambda x:(x.ORDER_START_TIME > (1+x.ORDER_END_TIME.shift())).cumsum())
       result=CG.groupby(["Key_Patient","ATC","group",'Dose']).agg({'ORDER_START_TIME': 'min',"ORDER_END_TIME": 'max'}).reset_index()
       print(len(result))
        
    MedMat=result.copy()   
    # Step 2: Removing Medications that Start After the End Date
    MedMat=MedMat[MedMat['ORDER_END_TIME']>MedMat['ORDER_START_TIME']]
    
    # Calculating Durations
    MedMat['Durations']=(MedMat['ORDER_END_TIME']-MedMat['ORDER_START_TIME'])+1 # since same day are counted as 1 day duration


    # MedMat['Durations']=(MedMat['ORDER_END_TIME']-MedMat['ORDER_START_TIME'])+1 # since same day are counted as 1 day duration
    # MedMat=MedMat[MedMat['Durations']>=0] # Keeping only positive durations
    # MedMat['ORDER_END_TIME'][MedMat['Durations']==0]=MedMat['ORDER_START_TIME'][MedMat['Durations']==0]+1 #setting one time medicaitons to a duration of one day
    # MedMat['Durations']=(MedMat['ORDER_END_TIME']-MedMat['ORDER_START_TIME'])+1# re calculating Durations
    
          
     # ADDING ATC LEVELS
    MedMat['ATC_lev1']=MedMat['ATC'].astype(str).str[0]
    MedMat['ATC_lev2']=MedMat['ATC'].astype(str).str[0:3]
    MedMat['ATC_lev4']=MedMat['ATC'].astype(str).str[0:4]
    MedMat['ATC_lev5']=MedMat['ATC'].astype(str).str[0:5]
    MedMat['ATC_lev7']=MedMat['ATC'].copy()
    MedMat=MedMat.drop(['ATC'],axis=1).reset_index(drop=True)

    # Cumulative Dosage
    MedMat['CumDosage']=(MedMat['Durations'])*(MedMat['Dose'])
        


 
    return MedMat


# %% LOAD DATA

basepath = "/home/projects/reg_00012/data/Ldrive_mirror/Data/EHR/SP_files_15102020/"


OrdMed = pd.read_feather(basepath + "clean_tables/"+"OrdineretMedicin.feather")


ATCTrans = pd.read_csv("ATCFSecNames.csv")
ATCNames=ATCTrans['ATCLevel2Name']
ATCNamesAbv=ATCTrans['ATCLevel2']
#
## ################# ORDERED MEDICINES ###############################
## Cleaning
#OrdMed = pd.read_feather(basepath + "clean_tables/"+"OrdineretMedicin.feather")
#dti_Ord=OrdMed['ORDER_END_TIME'].max() # CHANGE
#OrdMed['Dose']=OrdMed['value_min'].copy()
#OrdMed_Clean=MedCleaner(OrdMed.copy(),dti_Ord)
#OrdMed_Clean.to_feather('/home/projects/reg_00012/people/rudagi/OrdMed_Clean.feather')
#
## For Debugging Purposes############################
## PKey='0007647C1AF277A857DC62918AF3E6827D4C5FD303CEBB5FC51C8DED12160FC8'
## OrdMed_Clean_Test=OrdMed_Clean[OrdMed_Clean.Key_Patient==PKey]
##################################################
#
#
#
## ################# ADM MEDICINES ###############################
## Cleaning
#AdmMed = pd.read_feather(basepath + "clean_tables/" + "AdministreretMedicin.feather")
#AdmMed=AdmMed.copy()
#AdmMed['ORDER_START_TIME']=AdmMed['TAKEN_TIME'].copy()
#AdmMed['ORDER_END_TIME']=AdmMed['TAKEN_TIME']+datetime.timedelta(hours=24)
#dti_Adm=AdmMed['ORDER_END_TIME'].max() # CHANGE
#AdmMed['Dose']=AdmMed['SIG'].copy()
#
## PKey='0007647C1AF277A857DC62918AF3E6827D4C5FD303CEBB5FC51C8DED12160FC8'
## AdmMed=AdmMed[AdmMed.Key_Patient==PKey]
#
#AdmMed_Clean=MedCleaner(AdmMed.copy(),dti_Adm)
#AdmMed_Clean.to_feather('/home/projects/reg_00012/people/rudagi/AdmMed_Clean.feather')
#
#


# ################# Diagnosis ###############################
Diag_full = pd.read_feather(basepath + "clean_tables/" + "Diagnoses_combined.feather")
Diag_full['ICD10']=Diag_full['SKSKode'].copy()


 
Diag_T=DiagICDGroup(Diag_full.copy())


ICD10_GroupDescription=Diag_T.ICD10_GroupDescription.copy()
ICD10_Group= Diag_T.ICD10_Group.copy()
Diag_T.ICD10_GroupDescription=ICD10_GroupDescription.copy()
Diag_T.ICD10_Group=ICD10_Group.copy()
Diag_T['ORDER_START_TIME']=Diag_T['StartTid'].copy()
Diag_T['ORDER_END_TIME']=Diag_T['SlutTid'].copy()
   
dti_Diag=Diag_T['ORDER_END_TIME'].max() # CHANGE
# Setting Time Relative to dti (reference date)
Diag_T['ORDER_START_TIME']=(Diag_T['ORDER_START_TIME']-dti_Diag).astype('timedelta64[D]').astype(float)
Diag_T['ORDER_END_TIME']=(Diag_T['ORDER_END_TIME']-dti_Diag).astype('timedelta64[D]').astype(float)  
    
# Calculating Durations
Diag_T['Durations']=(Diag_T['ORDER_END_TIME']-Diag_T['ORDER_START_TIME'])+1 # since same day are counted as 1 day duration


Diag_T.to_feather('/home/projects/reg_00012/people/rudagi/Diag_Clean.feather')
