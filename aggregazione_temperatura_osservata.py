
import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm, gumbel_r, gumbel_l
from scipy import stats
import math
from statsmodels.distributions.empirical_distribution import ECDF
import boto3
import calendar as c

def isdir_s3(bucket, key):
    objs = list(bucket.objects.filter(Prefix=key))
#     print('I am the master_dir')
    return len(objs)

def costruzione_file_da_mensile(source, inizio, fine, tipo):
    bucketname = 'zus-prod-s3'
    s3 = boto3.resource('s3')
#     source = 'preprocessato/sistema/temperatura/epson/temperatura'
    my_bucket = s3.Bucket(bucketname)
    annomesi = pd.date_range(inizio, fine,freq='MS').strftime('%Y%m').tolist()
    df = pd.DataFrame()
    for am in annomesi:
        if isdir_s3(my_bucket,source + '/' +str(am))>0:
            df = df.append(pd.read_csv('s3://'+bucketname+'/'+source+'/'+str(am)+'/epson_best.csv'))

    df.drop('DATA',axis=1,inplace=True)
    df.rename(columns={'DATA_VAL':'date',tipo.upper():tipo.lower(),'TEMPERATURA_MIN':'temp_min','TEMPERATURA_MAX':'temp_max'},inplace=True)
    df['date'] = pd.to_datetime(df['date'],format="%Y-%m-%d")
    df['temp_mean_gc'] = (x['temp_max'] + x['temp_min'])/2
    df['temp_mean_gg'] = df.apply(lambda x: (max(0,18-x['temp_max']) + max(0,18-x['temp_min']))/2,axis=1)
    df['dayofyear'] = df.date.dt.dayofyear
#     df = df.assign(temp_mean_gc = lambda x: (x['temp_max'] + x['temp_min'])/2, temp_mean_gg = lambda x: (max(0,18-x['temp_max']) + max(0,18-x['temp_min']))/2, dayofyear = lambda x: x.date.dt.dayofyear)
    df['dayofyear'] = df.apply(lambda x: x['dayofyear']+1 if (c.isleap(x['date'].year)==False and x['date'].month>=3 and x['date'].month<=12) else x['dayofyear'],axis=1)
    
    return df

def lettura_pesi(path_pesi):
    
#     DA CONTROLLARE CON FILE PESI NUOVO PER EPSON
    
    my_bucket = 'zus-prod-s3'

#     AGGIUNGERE QUESTO POI NELLA REALTA'
#     df_pesi = pd.read_csv('s3://'+ my_bucket +'/'+path_pesi)
    df_pesi = pd.read_csv(path_pesi)
    df_pesi.rename(columns={'consumo_termico':'peso'},inplace=True)
    
    if 'consumo_termico' not in df_pesi.columns:
        df_pesi['date'] = pd.to_datetime(df_pesi['date'],format='%d/%m/%Y') # %Y-%m-%d
        df_pesi = df_pesi.assign(dayofyear = lambda x: x.date.dt.dayofyear)
        df_pesi['dayofyear'] = df_pesi.apply(lambda x: x['dayofyear']+1 if (c.isleap(x['date'].year)==False and x['date'].month>=3 and x['date'].month<=12) else x['dayofyear'],axis=1)
    else:
        df_pesi.rename(columns={'consumo_termico':'peso'},inplace=True)
        prv = pd.DataFrame()
        prv['dayofyear'] = range(1,367)
        prv['date'] = pd.date_range('2020-01-01','2020-12-31').astype('str')#.str.slice(start=5)
        prv['date'] = pd.to_datetime(prv['date'],format='%Y-%m-%d')
        df_pesi = df_pesi.merge(prv,on='dayofyear',how='left')
        
    df_pesi['sum_pesi_gg'] = df_pesi.groupby(['date'])['peso'].transform('sum')
    
    return df_pesi
    
def aggregazione_storico_epson(df,df_pesi,col_pesi_merge,tipo_colonna):
#     DA RICONTROLLARE COL FILE GIUSTO.
    plurale = {'provincia':'province','osservatorio':'osservatori'}
    if tipo_colonna=='provincia':
        tipo = 'prov'
    else:
        tipo = 'oss'
        
    print("Numero di " +plurale[tipo_colonna]+ " :",df[tipo_colonna].nunique())

    df = df[[tipo_colonna,'temp_mean']+col_pesi_merge]
    df_pesi = df.merge(df_pesi,on=col_pesi_merge+[tipo_colonna],how='left')
    
    df_pesi['temp_pesato'] = df_pesi['temp_mean']*df_pesi['peso']
    df_pesi_mean_gg = df_pesi[col_pesi_merge+['temp_pesato']].groupby(col_pesi_merge).sum().reset_index()

    df_pesi_mean = df_pesi_mean_gg.copy()
    df_pesi_mean = df_pesi_mean.merge(df_pesi[col_pesi_merge+['sum_pesi_gg']],on=col_pesi_merge,how='left')
    df_pesi_mean['temp_pesato'] = df_pesi_mean['temp_pesato']/df_pesi_mean['sum_pesi_gg']
    df_pesi_mean = df_pesi_mean[col_pesi_merge+['temp_pesato','sum_pesi_gg']]
    df_pesi_mean.drop_duplicates(subset=col_pesi_merge,inplace=True)
    
    return df_pesi_mean

def calcolo_media_mobile(df,finestra_media_mobile):
    
    df.sort_values(by=['dayofyear'],ascending=True,inplace=True)
    df.to_csv('prima_media_mobile.csv',index=False)
    df['media_mobile'] = 0
    df['media_mobile'] = df[['temp_pesato']].rolling(window=finestra_media_mobile, min_periods=1, center=True, closed='right').mean() #.stack().groupby('dayofyear')[['T_min_sum', 'T_max_sum','T_mean_sum']].agg(lambda x: np.mean(x))  
    df.to_csv('dopo_media_mobile.csv',index=False)
    
    return df

def main(inizio, fine, tipo_colonna, path_pesi, path_output):
    
    my_bucket = 'zus-prod-s3'
#     source_epson = 'preprocessato/sistema/temperatura/epson/temperatura'
    source_epson = 'm.piras/temperatura/epson/mm_3gg/'
    col_pesi_merge = ['dayofyear']
    tipo_colonna = tipo_colonna.lower()
    tipo_per_file = {'provincia':'prov','osservatorio':'oss'}
    timestamp = str(datetime.now())[0:-7].replace('-','').replace(' ','').replace(':','')
    idrun = inizio.replace('-','').replace('/','')+'_'+fine.replace('-','').replace('/','')+'_'+timestamp
    
    print("Costruzione epson da mensile.\n")
    epson = costruzione_file_da_mensile(source_epson, inizio, fine, tipo_colonna)
    epson_gg = epson.drop('temp_mean_gc').rename(columns={'temp_mean_gg':'temp_mean'}).copy()
    epson_gc = epson.drop('temp_mean_gg').rename(columns={'temp_mean_gc':'temp_mean'}).copy()

    print("Lettura file pesi.\n")
    df_pesi = lettura_pesi(path_pesi)

    print("Inizio procedura di aggregazione file epson per {}.\n".format(tipo_colonna))
    df_aggregato_gg = aggregazione_storico_epson(epson_gg,df_pesi,col_pesi_merge,tipo_colonna)
    df_aggregato_gc = aggregazione_storico_epson(epson_gc,df_pesi,col_pesi_merge,tipo_colonna)
  
#     print("Inizio procedura calcolo media mobile.\n")
#     output_gg = calcolo_media_mobile(df_aggregato_gg,3)
#     output_gc = calcolo_media_mobile(df_aggregato_gc,3)
    
    print("Scrittura output.\n")
    common_path_output = 's3://'+my_bucket+'/'+path_output+'agg_ita_norm_'+tipo_per_file[tipo_colonna]+'/'+idrun+'/'
    path_output_gg = common_path_output+'agg_ita_media3gg_gg_'+tipo_per_file[tipo_colonna]+'.csv'
    df_aggregato_gg.to_csv(pth_output_gg,index=False)
    path_output_gc = common_path_output+'agg_ita_media3gg_gc_'+tipo_per_file[tipo_colonna]+'.csv'
    df_aggregato_gc.to_csv(pth_output_gc,index=False)
    
    print("Scrittura metadati.\n")
    metadatati = pd.DataFrame(data={'CATEGORIA':['TEMPERATURA_NORM']*5,'FLUSSO':['AGG_ITA_NORM_'+tipo_colonna.upper()]*5,'TIMESTAMP':[timestamp]*5,'ID':['AGG_ITA_NORM_'+ tipo_colonna.upper() + '_' +now]*5,'PATH':[idrun]*5,'NOME_PARAMETRO':['DATA_INIZIO','DATA_FINE','TIPO_OUTPUT','PATH_PESI','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[inizio,fine,tipo_colonna,path_pesi,common_path_output]})
    metadatati.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/agg_ita_norm_'+tipo_dic[tipo]+'/'+idrun+'/metadati.csv',index=False)
    
    print("Procedura terminata.")
    
    return output

