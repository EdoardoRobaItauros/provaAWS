
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import statsmodels.api as sm
# from scipy import stats
from datetime import datetime
import calendar as c

def lettura_temp_norm(file_t_norm):
    df = pd.read_csv(file_t_norm)#,encoding='utf-16')
    
    df.rename(columns={'TIME - Date':'date','Codice Provincia':'provincia','Temperature Min':'temp_min','Temperature Max':'temp_max'},inplace=True)
#     df = df[df['provincia']==54]
    
    df['date'] = pd.to_datetime(df['date'],format='%d/%m/%Y')
    df['daymonth'] = (df.date.dt.day).astype('str').str.pad(2, side='left', fillchar='0') + (df.date.dt.month).astype('str').str.pad(2, side='left', fillchar='0')
    df['T_mean'] = df[['temp_min', 'temp_max']].mean(axis=1)
    df.drop('Desc_Provincia',axis=1,inplace=True)
#     df.info()

    df = df.assign(
        dayofyear=lambda x: (x.date).dt.dayofyear, # + pd.DateOffset(days=92)
        year=lambda x: (x.date).dt.year, #  + pd.DateOffset(days=92)
    )
#     print(df)
    df['dayofyear'] = df.apply(lambda x: x['dayofyear']+1 if (c.isleap(x['year'])==False and x['date'].month>=3 and x['date'].month<=12) else x['dayofyear'],axis=1)
    df['monthyear'] = (df.date.dt.month).astype('str').str.pad(2, side='left', fillchar='0')+(df.year).astype('str')
    return df

def lettura_temp_norm_curve_min_max(nome_file):
    df = pd.read_csv(nome_file)#,encoding='utf-16')
#     print(df)
#     df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')

#GIUSTO
    df['monthday'] = (df.mese).astype('str').str.pad(2, side='left', fillchar='0') +'-'+ (df.giorno).astype('str').str.pad(2, side='left', fillchar='0')
    df['date'] = "2020-" + df['monthday']
    df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')

#DA TOGLIERE
#     df['mese'] = df.date.dt.month
#     df['giorno'] = df.date.dt.day
    
    col = [x for x in df.columns if 'T_' in x][0]
    if 'provincia' in df.columns:
        codice = 'provincia'
    else:
        codice = 'codice_oss'
    df = df.assign(
      dayofyear=lambda x: (x.date).dt.dayofyear, # + pd.DateOffset(days=92)
      year=lambda x: (x.date).dt.year, # + pd.DateOffset(days=92)
    )
    df['dayofyear'] = df.apply(lambda x: x['dayofyear']+1 if (c.isleap(x['year'])==False and x['date'].month>=3 and x['date'].month<=12) else x['dayofyear'],axis=1)
    df.drop('date',axis=1,inplace=True)

    return df, col, codice

def lettura_osservatorio(file_oss):
    df_oss = pd.read_csv(file_oss)
    df_oss.rename(columns={'Cd Oss':'codice_oss','T Oss':'T_oss','Data':'date'},inplace=True)
#     df_oss = df_oss[df_oss['codice_oss']==11]
    df_oss['T_oss'] = df_oss.apply(lambda x: 18 if x['T_oss']==0 else 18-x['T_oss'],axis=1)
    df_oss['date'] = df_oss['date'].astype('str')
    df_oss['date'] = pd.to_datetime(df_oss['date'],format='%d/%m/%Y', exact=False)
    df_oss['daymonth'] = (df_oss.date.dt.day).astype('str').str.pad(2, side='left', fillchar='0') + (df_oss.date.dt.month).astype('str').str.pad(2, side='left', fillchar='0')
    df_oss = df_oss.assign(
        dayofyear=lambda x: (x.date).dt.dayofyear, # + pd.DateOffset(days=92)
        year=lambda x: (x.date).dt.year, # + pd.DateOffset(days=92)
    )
    df_oss['dayofyear'] = df_oss.apply(lambda x: x['dayofyear']+1 if (c.isleap(x['year'])==False and x['date'].month>=3 and x['date'].month<=12) else x['dayofyear'],axis=1)
    df_oss.drop('Oss',axis=1,inplace=True)
    df_oss['monthyear'] = (df_oss.date.dt.month).astype('str').str.pad(2, side='left', fillchar='0')+(df_oss.year).astype('str')
    return df_oss

def compute_yhat_prov(file_t_norm):

#     Calcolo temperature normali giornaliere    
    df = lettura_temp_norm(file_t_norm)
    df.rename(columns={'temp_max':'T_max_mean','temp_min':'T_min_mean','T_mean':'T_mean_m'},inplace=True)
    
    temp_norm = df.groupby(['dayofyear', 'provincia'])[['T_min_mean', 'T_max_mean','T_mean_m']].mean().unstack().reindex(range(367)).interpolate(method='linear').stack() #,'T_mean'
    province = temp_norm.reset_index(['provincia']).provincia.unique()
    finale = pd.DataFrame()

#     Fit armoniche di sesto grado per curve T_min, T_max e T_mean
    for p in province:
        temp_p = temp_norm.query('provincia == {}'.format(p)).reset_index(['provincia'])#.assign(std_mean=lambda x: x['std'], iv_l=lambda x: (x['mean'] - 2*x['std'].rolling(window=15, center=True).mean()), iv_h=lambda x: (x['mean'] + 2*x['std'].rolling(window=15, center=True).mean()))

        y_min = temp_p['T_min_mean'].values.ravel()
        y_max = temp_p['T_max_mean'].values.ravel()
        y_mean = temp_p['T_mean_m'].values.ravel()

        x = temp_p.index
        
        yhat_min = fit_harmonics(y_min, 6)
        yhat_max = fit_harmonics(y_max, 6)
        yhat_mean = fit_harmonics(y_mean, 6)

        temp_p['yhat_min'] = yhat_min
        temp_p['yhat_max'] = yhat_max
        temp_p['yhat_mean'] = yhat_mean

        finale = finale.append(temp_p)

    return finale.reset_index()

def fit_harmonics(y, n_harmonics):
    n = y.shape[0]
    X = np.matrix([*[[np.sin(i*k/n*2*np.pi) for i in range(n)] for k in range(n_harmonics)], *[[np.cos(i*k/n*2*np.pi) for i in range(n)] for k in range(n_harmonics)]]).T
    X = sm.add_constant(X)

    ols_model = sm.OLS(y, X).fit()
    yhat = ols_model.predict(X)
    
    return yhat

def aggregazione_serie_storiche_curve_min_max(nome_file,df_pesi,col_pesi_merge):
#     DA RICONTROLLARE COL FILE GIUSTO.
    
    df, col, codice = lettura_temp_norm_curve_min_max(nome_file)
    #df.drop('date',axis=1,inplace=True)
    print("Lettura file di temperature "+col+" per "+codice+".")
    
    print("Numero di "+codice+" presenti: "+str(df[codice].nunique()))
    
#     if col_pesi_merge==['mese','giorno']:
#         df_pesi.drop('date',axis=1,inplace=True)
    
    print("Procedura di ripartizione per peso avviata.")
    df_pesi = df.merge(df_pesi,on=col_pesi_merge+[codice],how='left')
    df_pesi['sum_pesi_gg'] = df_pesi.groupby(col_pesi_merge)['peso'].transform('sum')
    df_pesi[col+'_pesato'] = df_pesi[col]*df_pesi['peso']

    df_pesi[col + '_pesato'] = df_pesi.groupby(col_pesi_merge)[col + '_pesato'].transform('sum')

    df_pesi[col+'_pesato'] = df_pesi[col+'_pesato']/df_pesi['sum_pesi_gg']
    df_pesi = df_pesi[['date'] + col_pesi_merge+[col+'_pesato','sum_pesi_gg']]
    return df_pesi, col

def aggregazione_serie_storiche_norm_yhat(df,df_pesi,col_pesi_merge):
#     DA RICONTROLLARE COL FILE GIUSTO.
    print("Lettura file curva normale giornaliera da aggregare per provincia.")
    df = pd.read_csv(df)
    
    print("Numero province presenti: ",df.provincia.nunique())

    df = df[['provincia','yhat_mean']+col_pesi_merge]
    # df è il file di temperature max e min
    df_pesi = df.merge(df_pesi,on=col_pesi_merge+['provincia'],how='left')

    df_pesi['yhat_pesato'] = df_pesi['yhat_mean']*df_pesi['peso']
    
    df_pesi_mean_gg = df_pesi[['date','yhat_pesato']].groupby('date').sum().reset_index()

    df_pesi_mean_gg['mese'] = (df_pesi_mean_gg.date.dt.month).astype('int')
    df_pesi_mean_gg['giorno'] = (df_pesi_mean_gg.date.dt.day).astype('int')

    df_pesi_mean = df_pesi_mean_gg.copy()#[['daymonth',col+'_pesato']].groupby('daymonth').mean().reset_index()
    df_pesi_mean = df_pesi_mean.merge(df_pesi[col_pesi_merge+['sum_pesi_gg']],on=col_pesi_merge,how='left')
    df_pesi_mean['yhat_pesato'] = df_pesi_mean['yhat_pesato']/df_pesi_mean['sum_pesi_gg']
    df_pesi_mean = df_pesi_mean[['date','yhat_pesato','sum_pesi_gg']]
    df_pesi_mean.drop_duplicates(subset=['date'],inplace=True)
    return df_pesi_mean


def aggregazione_serie_storiche_oss_yhat(df,df_pesi,col_pesi_merge):
#     DA RICONTROLLARE COL FILE GIUSTO.
    print("Lettura file curva normale giornaliera da aggregare per osservatorio.")
    df = pd.read_csv(df)
    print("Numero osservatori presenti: ",df.codice_oss.nunique())

#     df['date_prov'] = '2020-'+df['mese'].astype('str').str.pad(2,'left','0')+'-'+df['giorno'].astype('str').str.pad(2,'left','0')
#     df.drop(['mese','giorno'],axis=1,inplace=True)
#     df['date_prov'] = pd.to_datetime(df['date_prov'],format='%Y-%m-%d')
#     df = df.assign(
#       dayofyear=lambda x: (x.date_prov).dt.dayofyear,
#     )

    df = df[['codice_oss','yhat']+col_pesi_merge]
    to_merge = df[col_pesi_merge].copy()

    df_pesi = df.merge(df_pesi,on=col_pesi_merge+['codice_oss'],how='left')
    df_pesi['yhat_pesato'] = df_pesi['yhat']*df_pesi['peso']
    df_pesi_mean_gg = df_pesi[col_pesi_merge+['yhat_pesato','date']].groupby(col_pesi_merge).sum().reset_index()
    df_pesi_mean_gg = df_pesi_mean_gg.merge(to_merge,on=col_pesi_merge,how='left')
    #df_pesi_mean_gg['monthday'] = (df_pesi_mean_gg['date_prov'].dt.month).astype('str').str.pad(2, side='left', fillchar='0')+'-'+(df_pesi_mean_gg['date_prov'].dt.day).astype('str').str.pad(2, side='left', fillchar='0')

    df_pesi_mean = df_pesi_mean_gg.copy()#[['daymonth',col+'_pesato']].groupby('daymonth').mean().reset_index()
    df_pesi_mean = df_pesi_mean.merge(df_pesi[col_pesi_merge+['sum_pesi_gg','date']],on=col_pesi_merge,how='left')
    df_pesi_mean['yhat_pesato'] = df_pesi_mean['yhat_pesato']/df_pesi_mean['sum_pesi_gg']
    df_pesi_mean = df_pesi_mean[['date','mese','giorno','yhat_pesato','sum_pesi_gg']]
    df_pesi_mean.drop_duplicates(subset=['mese','giorno'],inplace=True)
#     df.drop('date_prov',axis=1,inplace=True)
    
    return df_pesi_mean
    

def main_aggregazioni(file_input,pesi,path_to_output):
    print('Inizio funzione aggregazione')
    print('\n')
    
    my_bucket = 'zus-prod-s3'
    
    #idrun = '_'.join(file_input.split('/')[-2].split('_')[0:2]) + '_' + str(datetime.now())[0:-7].replace('-','').replace(' ','').replace(':','')
    
    df_pesi = pd.read_csv('s3://'+ my_bucket +'/'+pesi)
    if 'consumo_termico' in df_pesi.columns:
        df_pesi.rename(columns={'consumo_termico':'peso','osservatorio':'codice_oss'},inplace=True)
        prv = pd.DataFrame()
        prv['dayofyear'] = range(1,367)
        prv['date'] = pd.date_range('2020-01-01','2020-12-31').astype('str')#.str.slice(start=5)
        prv['date'] = pd.to_datetime(prv['date'],format='%Y-%m-%d')
        df_pesi = df_pesi.merge(prv,on='dayofyear',how='left')
    else:
        df_pesi['date'] = pd.to_datetime(df_pesi['date'],format='%d/%m/%Y') #'%Y-%m-%d') '%d/%m/%Y'

    df_pesi['mese'] = (df_pesi.date.dt.month).astype('int')
    df_pesi['giorno'] = (df_pesi.date.dt.day).astype('int')
    
    inizio_pesi = min(df_pesi['date']).strftime('%Y-%m-%d')
    fine_pesi = max(df_pesi['date']).strftime('%Y-%m-%d')
    idrun = inizio_pesi.replace('-','') + '_' + fine_pesi.replace('-','') + '_' + str(datetime.now())[0:-7].replace('-','').replace(' ','').replace(':','')

    df = pd.read_csv('s3://'+ my_bucket +'/'+file_input)
    list_nome_file = file_input.split('/')

    if 'provincia' in df.columns:
        
        col_pesi_merge = ['mese','giorno']

        if 'yhat_mean' in df.columns:
            df_pesi['sum_pesi_gg'] = df_pesi.groupby(['date'])['peso'].transform('sum')
            
            print('Aggregazione serie storiche temperature normali fittate (yhat_norm * pesi)')
            
            agg_ita = aggregazione_serie_storiche_norm_yhat('s3://'+ my_bucket +'/'+file_input,df_pesi,col_pesi_merge)
            file_csv = file_input.split('/')[-1]
            if 'max' in file_csv: tipo_yhat='max'
            elif 'min' in file_csv: tipo_yhat='min'
            else: tipo_yhat='mean'
            nome_output = 'agg_ita_norm_prov/'+idrun+'/agg_ita_norm_'+tipo_yhat+'_prov.csv'
            
            metadatati = pd.DataFrame(data={'MODELLO':['AGG_ITA_NORM_PROV']*3,'ID_RUN':['s3://'+ my_bucket +'/'+path_to_output+nome_output]*3,'NOME_PARAMETRO':['FILE_INPUT','PESI','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[file_input,pesi,path_to_output+'agg_ita_norm_prov/best/agg_ita_norm_'+tipo_yhat+'_prov.csv']})
            metadatati.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/agg_ita_norm_prov/'+idrun+'/metadati.csv',index=False)

        else:
            print('Aggregazione curve min_max periodo (curve * pesi)')
            
            agg_ita, col = aggregazione_serie_storiche_curve_min_max('s3://'+ my_bucket +'/'+file_input,df_pesi,col_pesi_merge)
            agg_ita.drop_duplicates(subset=['mese','giorno',col + '_pesato'],inplace=True) 
#             name = file_input.split('/')[-1]
#             name = file_input.replace('.csv','')
#             name = name.split('_')[-4:]
#             name = '_'.join(name)
            
            if 'T_min' in df.columns:
                nome_output = 'agg_ita_min_periodo_prov/'+idrun+'/agg_ita_min_periodo_prov.csv'
                metadatati = pd.DataFrame(data={'MODELLO':['AGG_ITA_MIN_PERIODO_PROV']*3,'ID_RUN':['s3://'+ my_bucket +'/'+path_to_output+nome_output]*3,'NOME_PARAMETRO':['FILE_INPUT','PESI','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[file_input,pesi,path_to_output+nome_output]})
                metadatati.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/agg_ita_min_periodo_prov/'+idrun+'/metadati.csv',index=False)
            else:
                nome_output = 'agg_ita_max_periodo_prov/'+idrun+'/agg_ita_max_periodo_prov.csv'
                metadatati = pd.DataFrame(data={'MODELLO':['AGG_ITA_MAX_PERIODO_PROV']*3,'ID_RUN':['s3://'+ my_bucket +'/'+path_to_output+nome_output]*3,'NOME_PARAMETRO':['FILE_INPUT','PESI','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[file_input,pesi,path_to_output+nome_output]})
                metadatati.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/agg_ita_max_periodo_prov/'+idrun+'/metadati.csv',index=False)

#         agg_ita['mese'] = agg_ita['date'].dt.month
#         agg_ita['giorno'] = agg_ita['date'].dt.day
#         print("prov: ",agg_ita)

        agg_ita.to_csv('s3://'+ my_bucket +'/'+path_to_output+nome_output,index=False)
    
#         pesi_mean_norm.to_csv(path_output+'agg_ita/best/'+nome_output+'_T_mean_p.csv',index=False)
#         pesi_yhat_norm.to_csv(path_output+'agg_ita/best/'+nome_output+'_yhat_p.csv',index=False)
        #agg_ita.to_csv('prova_agg_prov.csv',index=False)
    
    else:
        
        col_pesi_merge = ['mese','giorno']#['dayofyear']
        
        if 'yhat' in df.columns:
            df_pesi['sum_pesi_gg'] = df_pesi.groupby(['dayofyear'])['peso'].transform('sum')
            df_pesi.drop('dayofyear',axis=1,inplace=True)

            print('Aggregazione serie storiche temperature normali fittate (yhat_norm * pesi)')
            agg_ita = aggregazione_serie_storiche_oss_yhat('s3://'+ my_bucket +'/'+file_input,df_pesi,col_pesi_merge)
            file_csv = file_input.split('/')[-1]
            if 'max' in file_csv: tipo_yhat='max'
            elif 'min' in file_csv: tipo_yhat='min'
            else: tipo_yhat='mean'
            nome_output = 'agg_ita_norm_oss/'+idrun+'/agg_ita_norm_'+tipo_yhat+'_oss.csv'
            
            metadatati = pd.DataFrame(data={'MODELLO':['AGG_ITA_NORM_OSS']*3,'ID_RUN':['s3://'+ my_bucket +'/'+path_to_output+nome_output]*3,'NOME_PARAMETRO':['FILE_INPUT','PESI','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[file_input,pesi,path_to_output+'agg_ita_norm_oss/best/agg_ita_norm_'+tipo_yhat+'_oss.csv']})

            metadatati.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/agg_ita_norm_oss/'+idrun+'/metadati.csv',index=False)
           
        else:
            print('Aggregazione curve min_max periodo (curve * pesi)')
            df_pesi.drop('dayofyear',axis=1,inplace=True)

            agg_ita, col = aggregazione_serie_storiche_curve_min_max('s3://'+ my_bucket +'/'+file_input,df_pesi,col_pesi_merge)
            agg_ita.drop_duplicates(subset=['mese','giorno',col + '_pesato'],inplace=True)
#             name = file_input.split('/')[-1]
#             print(name)
#             name = file_input.replace('.csv','')
#             name = name.split('_')[-4:]
#             name = '_'.join(name)
            if 'min' in list_nome_file[-1]:
                nome_output = 'agg_ita_min_periodo_oss/'+idrun+'/agg_ita_min_periodo_oss.csv'
                metadatati = pd.DataFrame(data={'MODELLO':['AGG_ITA_MIN_PERIODO_PROV']*3,'ID_RUN':['s3://'+ my_bucket +'/'+path_to_output+nome_output]*3,'NOME_PARAMETRO':['FILE_INPUT','PESI','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[file_input,pesi,path_to_output+nome_output]})
                metadatati.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/agg_ita_min_periodo_oss/'+idrun+'/metadati.csv',index=False)
            else:
                nome_output = 'agg_ita_max_periodo_oss/'+idrun+'/agg_ita_max_periodo_oss.csv'
                metadatati = pd.DataFrame(data={'MODELLO':['AGG_ITA_MIN_PERIODO_PROV']*3,'ID_RUN':['s3://'+ my_bucket +'/'+path_to_output+nome_output]*3,'NOME_PARAMETRO':['FILE_INPUT','PESI','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[file_input,pesi,path_to_output+nome_output]})
                metadatati.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/agg_ita_max_periodo_oss/'+idrun+'/metadati.csv',index=False)
        
#         agg_ita['mese'] = agg_ita['monthday'].str.slice(stop=2)
#         agg_ita['giorno'] = agg_ita['monthday'].str.slice(start=3)
#         agg_ita.drop(['monthday'],axis=1,inplace=True)
#         print("oss: ",agg_ita)
#         agg_ita = agg_ita[['mese','giorno','yhat_pesato','sum_pesi_gg']]

        agg_ita.to_csv('s3://'+ my_bucket +'/'+path_to_output+nome_output,index=False)
#         agg_ita.to_csv('prova_pesi_oss_yhat.csv',index=False)
    print("Procedura terminata.")

