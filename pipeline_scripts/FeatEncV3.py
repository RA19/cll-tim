# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 20:13:50 2020

@author: RAGI0001
"""


import numpy as np
# import feather
import pandas as pd
from scipy import stats
import sklearn
import shap
import sklearn
from catboost import CatBoostClassifier, Pool, cv
import scipy.stats
import os
import datetime
from functools import reduce

from sklearn.metrics import auc,precision_recall_curve

def EvalMetricsMC(YProb,Ytest,YPred):

    EvaResy=pd.DataFrame()
    TP=np.sum(np.logical_and(Ytest==1,YPred>0)).astype(float)
    FN=np.sum(np.logical_and(Ytest==1,YPred==0)).astype(float)
    FP=np.sum(np.logical_and(Ytest==0,YPred>0)).astype(float)
    TN=np.sum(np.logical_and(Ytest==0,YPred==0)).astype(float)
    EvaResy['TP']=[TP]
    EvaResy['FP']=[FP]
    EvaResy['TN']=[TN]
    EvaResy['FN']=[FN]
    EvaResy['MCC']=[(TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))] #MCC,
    EvaResy['PrecPPV']=[TP/(TP+FP) ]# PRECISION PPV,
    EvaResy['SensRecall']=[TP/(TP+FN) ]# RECALL / Sensitivity / True Pos Rate,
    EvaResy['SpecTNR']=[TN/(TN+FP) ]# Specicitiy , True Neg Rate
    EvaResy['NPR']=[TN/(TN+FN)] # NPR

    prec,rec,thol=precision_recall_curve(Ytest,YProb)

    EvaResy['PRAUC']=[auc(rec,prec)]

    if np.logical_or(sum(Ytest==1)==len(Ytest),sum(Ytest==0)==len(Ytest)):
         EvaResy['ROC']=0
    else:
        EvaResy['ROC']=[sklearn.metrics.roc_auc_score(Ytest,YProb)]
    return EvaResy.round(3)

def shannonFunc(y):
    unu,inverse=np.unique(y,return_inverse=True)
    y=np.bincount(inverse)

    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    t = t[t!=0]
    H = -np.sum( t*np.log(t) )
    return H




def simpson(y):
    unu,inverse=np.unique(y,return_inverse=True)
    y=np.bincount(inverse)

    notabs = ~np.isnan(y)
    t = y[notabs] / np.sum(y[notabs])
    D = 1-np.sum( t**2 )
    return D




def PrescOverlaps(x,St,Et,atc2):
    x[St]=-x[St].copy()
    x[Et]=-x[Et].copy()
    atc2=atc2.values
   
    Len_atc2=len(atc2)
    NotNan=x[St].isnull()==0
    NanET=x[Et].isnull()
    x[Et][NanET]=x[St][NanET]
    #print(x[St])
    #print(x[St].min())
    #s1=abs(x[St].min().astype(int))
    s1=abs(np.int(x[St].min()))

    Maty=np.zeros((Len_atc2,abs(np.nanmin(x[St]).astype(int))+abs(np.nanmax(x[Et]).astype(int))))

    for atc in atc2:
        w=atc2==atc

        sties=(x[St][x.ATC_lev2==atc]+s1)[NotNan].values.astype(int)
        eties=(x[Et][x.ATC_lev2==atc]+s1)[NotNan].values.astype(int)
        for ncnt in range(len(eties)):

            Maty[w,sties[ncnt]:eties[ncnt]]=Maty[w,sties[ncnt]:eties[ncnt]]+1


    if np.sum(Maty)!=0:  
        d={}  
        MaxOverLapDist=Maty[:,np.sum(Maty>0,axis=0).argmax()]
        # print(MaxOverLapDist)
        for nst in range(Len_atc2):
            d[atc2[nst]+' medications at peak medication intake']=MaxOverLapDist[nst]
            

        
        d['Number of (ATC Level 2) prescriptions at peak medication intake']=np.sum(MaxOverLapDist) 
        d['Time Since Peak Prescriptions (ATC Level 2)']=np.argmax(np.sum(Maty>0,axis=0))+np.nanmin(x[St]).astype(int)
        d['Peak Number of Jointly Active Unique Prescriptions (ATC Level 2)']=np.sum(Maty>0,axis=0).max()
        d['Peak Number of Jointly Active Non-Unique Prescriptions (ATC Level 2)']=np.sum(Maty,axis=0).max()
                
                
        ActiveDays=np.sum(Maty>0,axis=1).astype(int)
        for nst in range(Len_atc2):
            d['Active Days on '+atc2[nst]+' Medication']=ActiveDays[nst]
    
        tt=np.bincount(np.sum(Maty>0,axis=0))
        ActivOver=np.append(tt,np.zeros((1,Len_atc2+1-len(tt)))[0]).astype(int)
        # print(ActivOver)
        for nst in range(len(ActivOver)):
            d['Active Days with '+str(nst)+' or more Unique Prescriptions (ATC level 2)']=np.nansum(ActivOver[nst:])



        hey=''
        for s in atc2:
            hey=np.append(hey,s+' medications at peak medication intake')
        
        hey=np.append(hey,'Number of (ATC Level 2) prescriptions at peak medication intake')
        hey=np.append(hey,'Time Since Peak Prescriptions (ATC Level 2)')
        hey=np.append(hey,'Peak Number of Jointly Active Unique Prescriptions (ATC Level 2)')
        hey=np.append(hey,'Peak Number of Jointly Active Non-Unique Prescriptions (ATC Level 2)')
        
        for s in atc2:
            hey=np.append(hey,'Active Days on '+s+' Medication')

        for nst in range(len(ActivOver)):

            hey=np.append(hey,'Active Days with '+str(nst)+' or more Unique Prescriptions (ATC level 2)')
        
        hey=hey[1:]     
                                      
        return pd.Series(d,index=hey)






def slopes(x,y):

    slp,intercept,r,p,stder=scipy.stats.linregress(x,y)
    return slp

def slopesIntercept(x,y):

    slp,intercept,r,p,stder=scipy.stats.linregress(x,y)
 
    return preddy

def slopesPred(x,y,T):

    slp,intercept,r,p,stder=scipy.stats.linregress(x,y)
    preddy=slp*(np.max(x)+T)+intercept
    return preddy


def f_baseStats_Values(x,OnWhat,St):
    d={}
    d['Number of Prescriptions/Tests for']=len(x[OnWhat])
    d['Average Dose/Value for']=np.nanmean(x[OnWhat])
    d['Median Dose/Valu for']=np.nanmedian(x[OnWhat])
    d['Highest Dose/Value for']=np.nanmax(x[OnWhat])
    d['Lowest Dose/Value for']=np.nanmin(x[OnWhat])
    d['Variability (stdev) in Dose/Value for']=np.nanstd(x[OnWhat])
    xs=x.copy()
    xs.sort_values(St,inplace=True)
    d['Last Dose/Value for']=xs[OnWhat].values[0]
    return pd.Series(d,index=['Number of Prescriptions/Tests for', 'Average Dose/Value forr','Median Dose/Valu for','Highest Dose/Value for',
                              'Lowest Dose/Value for','Variability (stdev) in Dose/Value for','Last Dose/Value for'])

def f_Durations(x,OnWhat,St):
    d={}
    d['Number of Events for']=len(x[OnWhat])
    d['Cumulative Duration for']=np.nansum(x[OnWhat])
    d['Average Duration for']=np.nanmean(x[OnWhat])
    d['Median Duration for']=np.nanmedian(x[OnWhat])
    d['Longest Duration for']=np.nanmax(x[OnWhat])
    d['Shortest Duration for']=np.nanmin(x[OnWhat])
    d['Variability (stdev) in Duration for']=np.nanstd(x[OnWhat])
    xs=x.copy()
    xs.sort_values(St,inplace=True)
    d['Latest Duration for']=xs[OnWhat].values[0]
    return pd.Series(d,index=['Number of Events for','Cumulative Duration for', 'Average Duration for','Median Duration for','Longest Duration for',
                              'Shortest Duration for','Variability (stdev) in Duration for','Latest Duration for'])



def f_Time_Medicine(x,ST,ET):
    d={}
    d['Number of Times Medication Restarted or Dose Changed:']=len(x[ST])
    d['Previously Took Medication:']=np.int64(len(x[ST])>0)
    d['Days Since a Restart or Dose Change of Prescription for:']=(x[ST]).min()
    d['Days Since End of Prescription for:']=(x[ET]).min()
    d['Days Since Start of  Prescription for:']=(x[ST]).max()
    return pd.Series(d,index=['Number of Times Medication Restarted or Dose Changed:',
                              'Previously Took Medication:',
                              'Days Since a Restart or Dose Change of Prescription for:',
                              'Days Since End of Prescription for:',
                              'Days Since Start of  Prescription for:'])

def f_Time_Basic(x,ST,ET):
    d={}
    d['Number of Tests or Events For:']=len(x[ST])
    d['Days Since Last Test or Event For:']=(x[ST]).min()
    d['Days Since First Test or Event For:']=(x[ST]).max()
    return pd.Series(d,index=['Number of Tests or Events For:',
                              'Days Since Last Test or Event For:',
                              'Days Since First Test or Event For:'])
                    
def f_Diversity_Values(x,OnWhat):
    d={}
    d['Shannon Diversity of:']=shannonFunc(x[OnWhat].values[x[OnWhat].isnull()==0])
    d['Simpson Diversity of:']=simpson(x[OnWhat].values[x[OnWhat].isnull()==0])

    return pd.Series(d,index=['Shannon Diversity of:','Simpson Diversity of:'])


def f_Diversity_Values_Global(x,OnWhat):
    d={}
    d['Shannon Diversity of: '+OnWhat]=shannonFunc(x[OnWhat].values[x[OnWhat].isnull()==0])
    d['Simpson Diversity of: '+OnWhat]=simpson(x[OnWhat].values[x[OnWhat].isnull()==0])
    
    d['Total Number of Events/Tests for: '+OnWhat]=len(x[OnWhat].values[x[OnWhat].isnull()==0])
    d['Total Number of Unique Events/Tests for: '+OnWhat]=len(x[OnWhat].unique())
    return pd.Series(d,index=['Shannon Diversity of: '+OnWhat,'Simpson Diversity of: '+OnWhat
                              ,'Total Number of Events/Tests for: '+OnWhat,'Total Number of Unique Events/Tests for: '+OnWhat])


    
def f_Cum_Dosage(x,St,Et,Cd,Du):
    xs=x.copy()
    xs.sort_values(St,inplace=True)
    d={}
    d['Rate of Change in Cumulative Dosage for:']=-slopes(xs[St],xs[Cd]) # negative since days, are days ago
    d['Active Number of Days on:']=xs[Du].sum()
    d['Number OFF days from Prescription of:']=abs(xs[Et].values[0]-xs[St].values[-1])-np.nansum(xs[Du]) # since first and last presciption
    d['Average Number of OFF Days Between Prescription of:']=np.nanmean((xs[Et].values-np.insert(xs[St].values,0,0)[0:-1])[1:])
    d['Current Cumulative Dosage for:']=np.nanmax(xs[Cd])

    return pd.Series(d,index=['Rate of Change in Cumulative Dosage for:',
                                  'Active Number Days on:',
                                  'Number of Days Not Taking:',
                                  'Mean Number of Days of Not Taking:',
                                  'Current Cumulative Dosage for:'])    

def f_RateOfChange(x,St,Et,V):
    xs=x.copy()
    xs.sort_values(St,inplace=True)
    d={}

    d['Predicted Value in 1 Week for:']=slopesPred(xs[St],xs[V],-7)
    d['Todays Predicted Value for:']=slopesPred(xs[St],xs[V],0)
    d['Rate of Increase of:']=slopes(xs[St],xs[V])
    d['Last Known Value for:']=np.nanmax(xs[V])

    return pd.Series(d,index=['Predicted Value in 1 Week for:',
                                  'Todays Predicted Value for:',
                                  'Rate of Increase of:',
                                  'Last Known Value for:'])    
            
  
def NFeatCalc(Curr,Categ,OnWhat,ST,ET,DUR,WhichFeat):
    Flag_IN=0
     
    if np.sum(WhichFeat=='Basic_Values')>0:
        print('Basic Values on '+Categ)
       
        YO=Curr.groupby(['Key_Patient',Categ]).apply(f_baseStats_Values,OnWhat,ST)
        YO=YO.pivot_table(index='Key_Patient',
                               columns=Categ,
                               values=['Number of Prescriptions/Tests for', 'Average Dose/Value forr','Median Dose/Valu for','Highest Dose/Value for',
                              'Lowest Dose/Value for','Variability (stdev) in Dose/Value for','Last Dose/Value for'],
                               aggfunc='first'
                               )
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)   
        

        
    if np.sum(WhichFeat=='Time_Medicine')>0:
        print('Time_Medicine on '+Categ)
   
        YO=Curr.groupby(['Key_Patient',Categ]).apply(f_Time_Medicine,ST,ET)
        YO=YO.pivot_table(index='Key_Patient',
                               columns=Categ,
                               values=['Number of Times Medication Restarted or Dose Changed:',
                                       'Previously Took Medication:',
                                       'Days Since a Restart or Dose Change of Prescription for:',
                                       'Days Since End of Prescription for:',
                                       'Days Since Start of  Prescription for:'],
                               aggfunc='first'
                               )
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)   
            


    if np.sum(WhichFeat=='Cumulative_Dosage')>0:
        print('Cumulative Dosage, Active Days, and Gap Time on '+Categ)
        YO=Curr.groupby(['Key_Patient',Categ]).apply(f_Cum_Dosage,ST,ET,OnWhat,DUR)
        YO=YO.pivot_table(index='Key_Patient',
                               columns=Categ,
                               values=['Rate of Change in Cumulative Dosage for:',
                                  'Active Number Days on:',
                                  'Number of Days Not Taking:',
                                  'Mean Number of Days of Not Taking:',
                                  'Current Cumulative Dosage for:']
                               )
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)   
     
 
            
     
    if np.sum(WhichFeat=='RateOfChange')>0:
        print('Rate of Change on '+Categ)
        YO=Curr.groupby(['Key_Patient',Categ]).apply(f_RateOfChange,ST,ET,OnWhat)
        YO=YO.pivot_table(index='Key_Patient',
                               columns=Categ,
                               values=['Predicted Value in 1 Week for:',
                                  'Todays Predicted Value for:',
                                  'Rate of Increase of:',
                                  'Last Known Value for:']
                               )
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)   

    if np.sum(WhichFeat=='ATC2_Active_Overlaps')>0:
        ATCTrans = pd.read_csv("ATCFSecNames.csv")
        ATCNames=ATCTrans['ATCLevel2Name']
        ATCNamesAbv=ATCTrans['ATCLevel2']
      
        print('Overlap Of Multiple Medications at ATC Level 2')
        YO=Curr.groupby(['Key_Patient']).apply(PrescOverlaps,'ORDER_START_TIME','ORDER_END_TIME',ATCNamesAbv)
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)   


    if np.sum(WhichFeat=='Diversity_Global')>0:
    
        print('Diversity on '+OnWhat+' type medications')
        YO=Curr.groupby(['Key_Patient']).apply(f_Diversity_Values_Global,OnWhat)

        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)  

    if np.sum(WhichFeat=='Diversity_Local')>0:

        print('Diversity on '+OnWhat+'type medications within each ATC1 code')

        YO=Curr.groupby(['Key_Patient',Categ]).apply(f_Diversity_Values,OnWhat)
        YO=YO.pivot_table(index='Key_Patient',
                               columns=Categ,
                               values=['Shannon Diversity of:','Simpson Diversity of:'],
                               aggfunc='first'
                               )
    
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)  
            
            

    if np.sum(WhichFeat=='Val_Changes')>0:
        print('Val_Changes')
        YO=Curr.groupby(['Key_Patient',Categ]).apply(DoseChanges,OnWhat)
        YO=YO.pivot_table(index='Key_Patient',
                               columns=Categ,
                               values=[OnWhat+'DoseChanges'],
                               aggfunc='first'
                               )
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()            
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)        

    if np.sum(WhichFeat=='Val_Diversity')>0:        
        print('Val_Diversity')
        YO=Curr.groupby(['Key_Patient',Categ]).apply(f_Diversity_Values,OnWhat)
        YO=YO.pivot_table(index='Key_Patient',
                               columns=Categ,
                               values=[OnWhat+'shannonFunc',OnWhat+'richness',OnWhat+'giniFunc',OnWhat+'simpson',OnWhat+'dom',OnWhat+'evenFunc'],
                               aggfunc='first'
                               )
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()    
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)        

    if np.sum(WhichFeat=='Val_Multi')>0:    
        print('Val_Multi')
        YO=Curr.groupby(['Key_Patient',Categ]).apply(DoseTimesAndValues,OnWhat,ST,ET,DUR)
        YO=YO.pivot_table(index='Key_Patient',
                               columns=Categ,
                               values=['Dose'+'FutPredValue','Dose'+'RateofChange','Prescp'+'24hrContinuations','Prescp'+'1weekContinuations','Prescp'+'1monthContinuations'],
                               aggfunc='first'
                               )
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()    
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)      
 
            

    if np.sum(WhichFeat=='Time_Basic')>0:
        print('Time_Basicon '+Categ)       
  
        YO=Curr.groupby(['Key_Patient',Categ]).apply(f_Time_Basic,ST,ET)
        YO=YO.pivot_table(index='Key_Patient',
                               columns=Categ,
                               values=['Number of Tests or Events For:',
                              'Days Since Last Test or Event For:',
                              'Days Since First Test or Event For:'],
                               aggfunc='first'
                               )
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()    
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)   




     
    if np.sum(WhichFeat=='Durations_Basic')>0:
        print('Durations_Basic '+Categ)
       
        YO=Curr.groupby(['Key_Patient',Categ]).apply(f_Durations,OnWhat,ST)
        YO=YO.pivot_table(index='Key_Patient',
                               columns=Categ,
                               values=['Number of Events for','Cumulative Duration for', 'Average Duration for','Median Duration for','Longest Duration for',
                              'Shortest Duration for','Variability (stdev) in Duration for','Latest Duration for'],
                               aggfunc='first'
                               )
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)   
        


         
    if np.sum(WhichFeat=='Val_Multi_Lab')>0:    
        print('Val_Multi_Lab')
        YO=Curr.groupby(['Key_Patient',Categ]).apply(LabTimesAndValues,OnWhat,ST)
        YO=YO.pivot_table(index='Key_Patient',
                               columns=Categ,
                               values=['Lab'+'FutPredValue','Lab'+'RateofChange','Lab'+'24hrContinuations','Lab'+'1weekContinuations','Lab'+'1monthContinuations'],
                               aggfunc='first'
                               )
        f1=pd.DataFrame(YO.to_records())
        f1=f1.rename(columns={'index': 'Key_Patient'})    
        if Flag_IN==0:
            Flag_IN=1
            df=f1.copy()    
        else:
            df=[df,f1]
            df=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df)  


    return df





def Enc_Med(Curr,NameSuffix):
    
    # FEATURE GENERATION
    # Time_Medicine on ATC_lev1
    dfy11=NFeatCalc(Curr,'ATC_lev2','','ORDER_START_TIME','ORDER_END_TIME','','Time_Medicine')
    dfy12=NFeatCalc(Curr,'ATC_lev7','','ORDER_START_TIME','ORDER_END_TIME','','Time_Medicine')
    # Basic On Dosage
    dfy2=NFeatCalc(Curr,'ATC_lev7','Dose','ORDER_START_TIME','ORDER_END_TIME','','Basic_Values')
      # Cumulative Dosage, Active Days, and Gap Time on ATC_lev7
    dfy3=NFeatCalc(Curr,'ATC_lev7','CumDosage','ORDER_START_TIME','ORDER_END_TIME','Durations','Cumulative_Dosage')
    # Overlap Of Multiple Medications at ATC Level 1 ATC_lev7
    dfy4=NFeatCalc(Curr,'','','ORDER_START_TIME','ORDER_END_TIME','Durations','ATC2_Active_Overlaps')
    # Diversity of prescriptoins at various ATC Levels
    dfy51=NFeatCalc(Curr,'','ATC_lev2','ORDER_START_TIME','ORDER_END_TIME','Durations','Diversity_Global')

    dfy55=NFeatCalc(Curr,'','ATC_lev7','ORDER_START_TIME','ORDER_END_TIME','Durations','Diversity_Global')
    # Diversity of ATC7 within each ATC2
    dfy6=NFeatCalc(Curr,'ATC_lev2','ATC_lev7','ORDER_START_TIME','ORDER_END_TIME','Durations','Diversity_Local')
    
    
    df_OUT=[dfy11,
    dfy12,
    dfy2,
    dfy3,
    dfy4,
    dfy51,
    dfy55,
    dfy6]
    
    # Joining All Feature Matrices
    df_OUT=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df_OUT)  
    df_OUT=df_OUT.add_suffix(NameSuffix)
    df_OUT=df_OUT.rename(columns={'Key_Patient'+NameSuffix:'Key_Patient'})
    
    return df_OUT

def Enc_Diag(Curr,NameSuffix):
    
    # FEATURE GENERATION
    # Time Features on Diagnosis
    dfy1=NFeatCalc(Curr,'ICD10','','ORDER_START_TIME','ORDER_END_TIME','','Time_Basic')
    dfy1a=NFeatCalc(Curr,'ICD10_GroupDescription','','ORDER_START_TIME','ORDER_END_TIME','','Time_Basic')

    # Diversity of Diagnosis 
    dfy2=NFeatCalc(Curr,'','ICD10','ORDER_START_TIME','ORDER_END_TIME','','Diversity_Global')
    dfy2a=NFeatCalc(Curr,'ICD10_GroupDescription','ICD10','ORDER_START_TIME','ORDER_END_TIME','Durations','Diversity_Local')

    # Durations
    dfy3=NFeatCalc(Curr,'ICD10','Durations','ORDER_START_TIME','ORDER_END_TIME','','Durations_Basic')
    dfy3a=NFeatCalc(Curr,'ICD10_GroupDescription','Durations','ORDER_START_TIME','ORDER_END_TIME','','Durations_Basic')
   
    
    df_OUT=[dfy1,dfy1a,dfy2,dfy2a,dfy3,dfy3a]
    
    
    # Joining All Feature Matrices
    df_OUT=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df_OUT)  
    df_OUT=df_OUT.add_suffix(NameSuffix)
    df_OUT=df_OUT.rename(columns={'Key_Patient'+NameSuffix:'Key_Patient'})
    
    return df_OUT




def Enc_VitalsC(Curr,NameSuffix):
    
    # FEATURE GENERATION
    # Time_Medicine on Diagnosis
    dfy1=NFeatCalc(Curr,'CategVal','','ORDER_START_TIME','ORDER_END_TIME','','Time_Basic')


    # Diversity of prescriptoins at various ATC Levels
    dfy2=NFeatCalc(Curr,'','CategVal','ORDER_START_TIME','ORDER_END_TIME','','Diversity_Global')

    
    df_OUT=[dfy1,dfy2]
    
    # Joining All Feature Matrices
    df_OUT=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df_OUT)  
    df_OUT=df_OUT.add_suffix(NameSuffix)
    df_OUT=df_OUT.rename(columns={'Key_Patient'+NameSuffix:'Key_Patient'})
    
    return df_OUT





def Enc_Lab(Curr,NameSuffix):
    
    # FEATURE GENERATION
    # Time_Medicine on Name of Labtest
    dfy1=NFeatCalc(Curr,'EXTERNAL_NAME','','ORDER_START_TIME','ORDER_START_TIME','','Time_Basic')

    # Basic On Results of Labtest
    dfy2=NFeatCalc(Curr,'EXTERNAL_NAME','ORD_VALUE_num','ORDER_START_TIME','ORDER_START_TIME','','Basic_Values')

    dfy3=NFeatCalc(Curr,'EXTERNAL_NAME','ORD_VALUE_num','ORDER_START_TIME','ORDER_START_TIME','','RateOfChange')
    # Diversity of of test taken
    dfy4=NFeatCalc(Curr,'','EXTERNAL_NAME','ORDER_START_TIME','ORDER_START_TIME','','Diversity_Global')

    
    
    df_OUT=[dfy1,
    dfy2,
    dfy3,
    dfy4]
    
    # Joining All Feature Matrices
    df_OUT=reduce(lambda left, right: pd.merge(left,right,on='Key_Patient',how='outer'),df_OUT)  
    df_OUT=df_OUT.add_suffix(NameSuffix)
    df_OUT=df_OUT.rename(columns={'Key_Patient'+NameSuffix:'Key_Patient'})
    
    return df_OUT



def EvalMetrics(YProb,Ytest,THold):
    YPred=(YProb>THold).astype(int)
    EvaResy=pd.DataFrame()
    TP=np.sum(np.logical_and(Ytest==1,YPred>0)).astype(float)
    FN=np.sum(np.logical_and(Ytest==1,YPred==0)).astype(float)
    FP=np.sum(np.logical_and(Ytest==0,YPred>0)).astype(float)
    TN=np.sum(np.logical_and(Ytest==0,YPred==0)).astype(float)
    EvaResy['TP']=[TP]
    EvaResy['FN']=[FN]
    EvaResy['FP']=[FP]
    EvaResy['TN']=[TN]
    EvaResy['MCC']=[(TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))] #MCC,
    EvaResy['PrecPPV']=[TP/(TP+FP) ]# PRECISION PPV,
    EvaResy['SensRecall']=[TP/(TP+FN) ]# RECALL / Sensitivity / True Pos Rate,
    EvaResy['SpecTNR']=[TN/(TN+FP) ]# Specicitiy , True Neg Rate
    EvaResy['NPR']=[TN/(TN+FN)] # NPR
    prec,rec,thol=precision_recall_curve(Ytest,YProb)

    EvaResy['PRAUC']=[auc(rec,prec)]
    if np.logical_or(sum(Ytest==1)==len(Ytest),sum(Ytest==0)==len(Ytest)):
       EvaResy['ROC']=0
    else:
        EvaResy['ROC']=[sklearn.metrics.roc_auc_score(Ytest,YProb)]
    
    
    return EvaResy.round(3)

