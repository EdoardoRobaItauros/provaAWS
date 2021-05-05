
import pandas as pd
import numpy as np
import boto3
# import seaborn as sns
# import matplotlib.pyplot as plt
import statsmodels.api as sm
# from scipy import stats
from datetime import datetime
from calendar import monthrange
import calendar as c




def lettura_temp_norm(file_t_norm,anno_inizio,anno_fine):
    df = pd.read_csv(file_t_norm)#,encoding='utf-16')

    df.rename(columns={'TIME - Date':'date','Codice Provincia':'provincia','Temperature Min':'temp_min','Temperature Max':'temp_max'},inplace=True)
    
    df['date'] = pd.to_datetime(df['date'],format='%d/%m/%Y')
    df = df[(df['date'].dt.year>=anno_inizio) & (df['date'].dt.year<=anno_fine)]
    df['T_mean'] = df[['temp_min', 'temp_max']].mean(axis=1)
    df.drop('Desc_Provincia',axis=1,inplace=True)

    df = df.assign(
#         dayofyear=lambda x: (x.date).dt.dayofyear, # + pd.DateOffset(days=92)
        year=lambda x: (x.date).dt.year, # + pd.DateOffset(days=92)
    )

    df['monthyear'] = (df.date.dt.month).astype('str').str.pad(2, side='left', fillchar='0')+(df.year).astype('str')
    return df

def lettura_osservatorio(file_oss,anno_inizio,anno_fine):
    df_oss = pd.read_csv(file_oss)

    df_oss.rename(columns={'Cd Oss':'codice_oss','T Oss':'T_oss','Data':'date'},inplace=True)

    df_oss['T_oss'] = df_oss.apply(lambda x: 18 if x['T_oss']==0 else 18-x['T_oss'],axis=1)
    
    df_oss['date'] = df_oss['date'].astype('str')
    df_oss['date'] = pd.to_datetime(df_oss['date'],format='%Y-%m-%d', exact=False)
    df_oss = df_oss[(df_oss['date'].dt.year>=anno_inizio) & (df_oss['date'].dt.year<=anno_fine)]
    df_oss = df_oss.assign(
#         dayofyear=lambda x: (x.date).dt.dayofyear, # + pd.DateOffset(days=92)
        year=lambda x: (x.date).dt.year, # + pd.DateOffset(days=92)
    )
    df_oss.drop('Oss',axis=1,inplace=True)
    df_oss['monthyear'] = (df_oss.date.dt.month).astype('str').str.pad(2, side='left', fillchar='0')+(df_oss.year).astype('str')
    return df_oss

def temp_best_col_in_periodo(df,mese_inizio,mese_fine,col):
    
    if mese_inizio <= mese_fine:
        date_value = pd.to_datetime('2020-'+str(mese_fine)+'-01',format='%Y-%m-%d')
        date_value = date_value.replace(day = monthrange(date_value.year, date_value.month)[1])
        count_days = ((date_value - pd.to_datetime('2020-'+str(mese_inizio)+'-01',format='%Y-%m-%d')).days)*df.provincia.nunique()
        tmp = df[(df['date'].dt.month>=mese_inizio) & (df['date'].dt.month<=mese_fine)]
#         print(count_days)
        prov = tmp[['year','T_'+col]].groupby('year').count().reset_index()
        prov = prov[prov['T_'+col]>=count_days]
        years_ok = prov.year.unique()
        tmp = tmp[tmp['year'].isin(years_ok)]
        tmp = tmp.loc[~((tmp['date'].dt.month == 2) & (tmp['date'].dt.day == 29))]
        
        print("Individuazione curve di temperatura minima o massima")
        province = tmp.provincia.unique()
        final = pd.DataFrame()
        for prov in province:
            tmp_p = tmp[tmp['provincia']==prov]
            sums = tmp_p[['year','T_'+col]].groupby('year').sum().reset_index()
            if col=='max':
                year = sums[sums['T_max']==max(sums['T_max'])].year.unique()[0]
                to_drop = 'T_min'
            else:
                year = sums[sums['T_min']==min(sums['T_min'])].year.unique()[0]
                to_drop = 'T_max'
            temp_best_in_period = tmp_p[tmp_p.date.dt.year == year]
            final = final.append(temp_best_in_period)
    else:
        date_value = pd.to_datetime('2021-'+str(mese_fine)+'-01',format='%Y-%m-%d')
        date_value = date_value.replace(day = monthrange(date_value.year, date_value.month)[1])
        count_days = ((date_value - pd.to_datetime('2020-'+str(mese_inizio)+'-01',format='%Y-%m-%d')).days)*df.provincia.nunique()
        df['to_groupby'] = 0

        df['to_groupby'] = df.apply(lambda x: x.date.year+1 if x.date.month>=mese_inizio else x.date.year,axis=1)
        tmp = df[((df['date'].dt.month>=mese_inizio) | (df['date'].dt.month<=mese_fine))]

        prov = tmp[['to_groupby','T_'+col]].groupby('to_groupby').count().reset_index()
        prov = prov[prov['T_'+col]>=count_days]
        years_ok = prov.to_groupby.unique()
        tmp = tmp[tmp['to_groupby'].isin(years_ok)]
        tmp = tmp.loc[~((tmp['date'].dt.month == 2) & (tmp['date'].dt.day == 29))]
        
        print("Individuazione curve di temperatura minima o massima")
        tmp['monthyear'] = pd.to_datetime(tmp['monthyear'],format='%m%Y')
        province = tmp.provincia.unique()
        final = pd.DataFrame()
        
        for prov in province:
            tmp_p = tmp[tmp['provincia']==prov]
            sums = tmp_p[['to_groupby','T_'+col]].groupby('to_groupby').sum().reset_index()
            if col=='max':
                year = sums[sums['T_max']==max(sums['T_max'])].to_groupby.unique()[0]
                to_drop = 'T_min'
            else:
                year = sums[sums['T_min']==min(sums['T_min'])].to_groupby.unique()[0]
                to_drop = 'T_max'
            temp_best_in_period = tmp_p[tmp_p.to_groupby == year]
            final = final.append(temp_best_in_period)
    final.drop(to_drop,axis=1,inplace=True)
    return final

def temp_best_col_in_periodo_oss(df,mese_inizio,mese_fine,col):
    if mese_inizio < mese_fine:
        date_value = pd.to_datetime('2020-'+str(mese_fine)+'-01',format='%Y-%m-%d')
        date_value = date_value.replace(day = monthrange(date_value.year, date_value.month)[1])
        count_days = ((date_value- pd.to_datetime('2020-'+str(mese_inizio)+'-01',format='%Y-%m-%d')).days)*df.codice_oss.nunique()
        tmp = df[(df['date'].dt.month>=mese_inizio) & (df['date'].dt.month<=mese_fine)]
        
        prov = tmp[['year','T_oss']].groupby('year').count().reset_index()
        prov = prov[prov['T_oss']>=count_days]
        years_ok = prov.year.unique()
        tmp = tmp[tmp['year'].isin(years_ok)]
        tmp = tmp.loc[~((tmp['date'].dt.month == 2) & (tmp['date'].dt.day == 29))]
        
        print("Individuazione curve di temperatura minima o massima")
        osservatori = tmp.codice_oss.unique()
        final = pd.DataFrame()
        for oss in osservatori:
            tmp_o = tmp[tmp['codice_oss']==oss]
            sums = tmp_o[['year','T_oss']].groupby('year').sum().reset_index()
            if col=='max':
                year = sums[sums['T_oss']==max(sums['T_oss'])].year.unique()[0]
            else:
                year = sums[sums['T_oss']==min(sums['T_oss'])].year.unique()[0]
            temp_best_in_period = tmp_o[tmp_o.date.dt.year == year]
            final = final.append(temp_best_in_period)
    else:
        date_value = pd.to_datetime('2021-'+str(mese_fine)+'-01',format='%Y-%m-%d')
        date_value = date_value.replace(day = monthrange(date_value.year, date_value.month)[1])
        count_days = ((date_value - pd.to_datetime('2020-'+str(mese_inizio)+'-01',format='%Y-%m-%d')).days)*df.codice_oss.nunique()
        df['to_groupby'] = 0
        df['to_groupby'] = df.apply(lambda x: x.date.year+1 if x.date.month>=mese_inizio else x.date.year,axis=1)
        tmp = df[((df['date'].dt.month>=mese_inizio) | (df['date'].dt.month<=mese_fine))]
        tmp['monthyear'] = pd.to_datetime(tmp['monthyear'],format='%m%Y')
        tmp = tmp.loc[~((tmp['date'].dt.month == 2) & (tmp['date'].dt.day == 29))]

        prov = tmp[['to_groupby','T_oss']].groupby('to_groupby').count().reset_index()
        prov = prov[prov['T_oss']>=count_days]
        years_ok = prov.to_groupby.unique()
        tmp = tmp[tmp['to_groupby'].isin(years_ok)]

        print("Individuazione curve di temperatura minima o massima")
        osservatori = tmp.codice_oss.unique()
        final = pd.DataFrame()
        for oss in osservatori:
            tmp_o = tmp[tmp['codice_oss']==oss]
            sums = tmp_o[['to_groupby','T_oss']].groupby('to_groupby').sum().reset_index()
            if col=='max':
                year = sums[sums['T_oss']==max(sums['T_oss'])].to_groupby.unique()[0]
            else:
                year = sums[sums['T_oss']==min(sums['T_oss'])].to_groupby.unique()[0]
            temp_best_in_period = tmp_o[tmp_o.to_groupby == year]
            final = final.append(temp_best_in_period)
        
    return final

def individuazione_min_max_periodo_norm(file_t_norm,mese_inizio,mese_fine,anno_inizio,anno_fine,col,path_to_output,my_bucket,idrun_f):

    print("Lettura file temperature per provincia.")
    df = lettura_temp_norm(file_t_norm,anno_inizio,anno_fine)

    print("Numero province presenti: ", df['provincia'].nunique())

    df.drop(['T_mean'],axis=1,inplace=True)
    df.rename(columns={'temp_max':'T_max','temp_min':'T_min'},inplace=True)

    nome_output = str(mese_inizio) + '_' + str(mese_fine) + '_' + str(anno_inizio) + '_' + str(anno_fine)

    temp_best_period = temp_best_col_in_periodo(df,mese_inizio,mese_fine,col)
    temp_best_period.drop(['year','monthyear'],inplace=True,axis=1)
    
    temp_best_period['mese'] = temp_best_period['date'].dt.month
    temp_best_period['giorno'] = temp_best_period['date'].dt.day
    
    temp_best_period = temp_best_period.assign(
        dayofyear=lambda x: (x.date).dt.dayofyear # + pd.DateOffset(days=92)
        #year=lambda x: (x.date).dt.year, # + pd.DateOffset(days=92)
    )
    temp_best_period['dayofyear'] = temp_best_period.apply(lambda x: x['dayofyear']+1 if (c.isleap(x['year'])==False and x['date'].month>=3 and x['date'].month<=12) else x['dayofyear'],axis=1)
    temp_best_period.drop('to_groupby',axis=1,inplace=True)
    temp_best_period.to_csv(path_to_output+col+'_periodo_prov/'+idrun_f+'/'+col+'_periodo.csv',index=False)
    
    idrun = path_to_output+col+'_periodo_prov/'+idrun_f+'/'+col+'_periodo.csv'
    metadatati = pd.DataFrame(data={'MODELLO':[col+'_PERIODO_PROV']*7,'ID_RUN' : [idrun]*7,'NOME_PARAMETRO':['FILE_INPUT','MESE_INIZIO','MESE_FINE','ANNO_INIZIO','ANNO_FINE','COLONNA','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[file_t_norm,mese_inizio,mese_fine,anno_inizio,anno_fine,col,path_to_output+col+'_periodo_prov/best/'+col+'_periodo_'+nome_output+'.csv']})

    metadatati.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/'+col+'_periodo_prov/'+idrun_f+'/metadati.csv',index=False)

def individuazione_min_max_periodo_oss(file_oss,mese_inizio_oss,mese_fine_oss,anno_inizio,anno_fine,col_oss,path_to_output,my_bucket,idrun_f):

    print("Lettura file temperature per osservatorio.")
    df_oss = lettura_osservatorio(file_oss,anno_inizio,anno_fine)
    
    print("Numero osservatori presenti: ", df_oss['codice_oss'].nunique())
    nome_output = str(mese_inizio_oss) + '_' + str(mese_fine_oss) + '_' + str(anno_inizio) + '_' + str(anno_fine)
    
    temp_best_period_oss = temp_best_col_in_periodo_oss(df_oss,mese_inizio_oss,mese_fine_oss,col_oss)

    temp_best_period_oss.drop(['year','monthyear'],inplace=True,axis=1)

    temp_best_period_oss['mese'] = temp_best_period_oss['date'].dt.month
    temp_best_period_oss['giorno'] = temp_best_period_oss['date'].dt.day
    
    temp_best_period_oss = temp_best_period_oss.assign(
        dayofyear=lambda x: (x.date).dt.dayofyear # + pd.DateOffset(days=92)
        #year=lambda x: (x.date).dt.year, # + pd.DateOffset(days=92)
    )
    temp_best_period_oss['dayofyear'] = temp_best_period_oss.apply(lambda x: x['dayofyear']+1 if (c.isleap(x['year'])==False and x['date'].month>=3 and x['date'].month<=12) else x['dayofyear'],axis=1)
    temp_best_period_oss.drop('to_groupby',axis=1,inplace=True)


    temp_best_period_oss.to_csv(path_to_output+col_oss+'_periodo_oss/'+idrun_f+'/'+col_oss+'_periodo.csv',index=False)
    idrun = path_to_output+col_oss+'_periodo_oss/'+idrun_f+'/'+col_oss+'_periodo.csv'
    metadatati = pd.DataFrame(data={'MODELLO':[col_oss+'_PERIODO_OSS']*7,'ID_RUN':[idrun]*7,'NOME_PARAMETRO':['FILE_INPUT','MESE_INIZIO','MESE_FINE','ANNO_INIZIO','ANNO_FINE','COLONNA','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[file_oss,mese_inizio_oss,mese_fine_oss,anno_inizio,anno_fine,col_oss,path_to_output+col_oss+'_periodo_prov/best/'+col_oss+'_periodo_'+nome_output+'.csv']})

    metadatati.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/'+col_oss+'_periodo_oss/'+idrun_f+'/metadati.csv',index=False)

def main_in_periodo(file_input,mese_inizio,mese_fine,anno_inizio,anno_fine,col,path_output):
    print('Funzione dedicata all\'individuazione della temperatura massima e minima nel periodo definito.')
    print('\n')
    
    bucket = 'zus-prod-s3'
    
    file_input = 's3://' +bucket + '/' + file_input
    path_output = 's3://' +bucket + '/'  + path_output
    
    idrun = str(anno_inizio)+str(mese_inizio) + "_" + str(anno_fine)+str(mese_fine) + str(datetime.now())[0:-7].replace('-','').replace(' ','').replace(':','')
    
    anno_inizio = int(anno_inizio)
    anno_fine = int(anno_fine)
    
    df = pd.read_csv(file_input)#,encoding='utf-16')
    if 'Codice Provincia' in df.columns:
        print('Individuazione curva con temperatura normale min/max nel periodo definito')
        individuazione_min_max_periodo_norm(file_input,mese_inizio,mese_fine,anno_inizio,anno_fine,col,path_output,bucket,idrun)
        
        print('\n')
    else:
        print('Individuazione curva con temperatura osservatorio min/max nel periodo definito')
        individuazione_min_max_periodo_oss(file_input,mese_inizio,mese_fine,anno_inizio,anno_fine,col,path_output,bucket,idrun)
        
        print('\n')
    print("Procedura terminata.")

