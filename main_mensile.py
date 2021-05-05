
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import statsmodels.api as sm
# from scipy import stats
from datetime import datetime,timedelta
import calendar as c

def compute_std_var_month(df_in,finestra_media_mobile_m,col):
    df = df_in.copy()
    days = int(np.floor(finestra_media_mobile_m/2))
    mds = sorted(df.monthday.unique())
    min_md = df[df['date']==min(df['date'])]['monthday'].ravel()[0]
    max_md = df[df['date']==max(df['date'])]['monthday'].ravel()[0]
    df['isleap'] = 0
    df['isleap'] = df.apply(lambda x: 1 if c.isleap(x['date'].year) else 0,axis=1)
    to_append = pd.DataFrame()
    for md in mds:
        if md!='02-29':
            lista = pd.date_range(datetime.strptime(str(md), "%m-%d")-timedelta(days=days), datetime.strptime(str(md), "%m-%d")+timedelta(days=days)).strftime("%m-%d").tolist()
            var = df[df.monthday.isin(lista)][col].var(ddof=0)
            std = np.sqrt(var)
            to_append = to_append.append(pd.DataFrame({col+'_var':[var],col+'_std':[std]},index=[md]))
        else:
            lista = pd.date_range(datetime.strptime(str('02-22'), "%m-%d"), datetime.strptime(str('03-07'), "%m-%d")).strftime("%m-%d").tolist()
            lista = lista + ['02-29']
            var = df[(df.monthday.isin(lista)) & (df.isleap == 1)][col].var(ddof=0)
            std = np.sqrt(var)
            to_append = to_append.append(pd.DataFrame({col+'_var':[var],col+'_std':[std]},index=[md]))
    to_append.reset_index(inplace=True)
    to_append.rename(columns={'index':'monthday'},inplace=True)
    df = df.merge(to_append,on='monthday',how='left')
    return df

def compute_std(df,finestra_media_mobile_g):
    return df.set_index(['dayofyear', 'year']).unstack().rolling(window=finestra_media_mobile_g, min_periods=1, center=True).var().stack().groupby('dayofyear')[['T_min_mean', 'T_max_mean','T_mean_m']].agg(lambda x: np.sqrt(np.mean(x)))    

def compute_var(df,finestra_media_mobile_g):
    return df.set_index(['dayofyear', 'year']).unstack().rolling(window=finestra_media_mobile_g, min_periods=1, center=True).var().stack().groupby('dayofyear')[['T_min_mean', 'T_max_mean','T_mean_m']].agg(lambda x: np.mean(x))

# def compute_std_month(df,finestra_media_mobile_m):
# #     print(df)
#     df = df.set_index(['dayofyear', 'year']).unstack().rolling(window=finestra_media_mobile_m, min_periods=1, center=True).var().stack()#.groupby('dayofyear')[['T_min_sum', 'T_max_sum','T_mean_sum']].agg(lambda x: np.sqrt(np.mean(x)))  
# #     df = df.reset_index()
#     df.rename(columns={'T_min_sum':'std_t_min', 'T_max_sum':'std_t_max','T_mean_sum':'std_t_mean'},inplace=True)
#     return df
    
# def compute_var_month(df,finestra_media_mobile_m):
#     df = df.set_index(['dayofyear', 'year']).unstack().rolling(window=finestra_media_mobile_m, min_periods=1, center=True).var().stack()#.groupby('dayofyear')[['T_min_sum', 'T_max_sum','T_mean_sum']].agg(lambda x: np.mean(x))  
# #     df = df.reset_index()
#     df.rename(columns={'T_min_sum':'var_t_min', 'T_max_sum':'var_t_max','T_mean_sum':'var_t_mean'},inplace=True)
# #     print(df)
#     return df
    
def compute_std_oss(df,finestra_media_mobile_g):
    return df.set_index(['dayofyear', 'year'],append=True).unstack().rolling(window=finestra_media_mobile_g, min_periods=1, center=True).var().stack().groupby('dayofyear')[['T_oss']].agg(lambda x: np.sqrt(np.mean(x)))    

def compute_var_oss(df,finestra_media_mobile_g):
    return df.set_index(['dayofyear', 'year'],append=True).unstack().rolling(window=finestra_media_mobile_g, min_periods=1, center=True).var().stack().groupby('dayofyear')[['T_oss']].agg(lambda x: np.mean(x))

# def compute_std_month_oss(df,finestra_media_mobile_m):
#     df = df.set_index(['dayofyear', 'year'],append=True).unstack().rolling(window=finestra_media_mobile_m, min_periods=1, center=True).var().stack()#.groupby('dayofyear')[['T_oss']].agg(lambda x: np.sqrt(np.mean(x)))  
# #     df = df.reset_index()
# #     print(df.info())
#     df.rename(columns={'T_oss':'std_oss'},inplace=True)
#     return df
    
# def compute_var_month_oss(df,finestra_media_mobile_m):
#     df = df.set_index(['dayofyear', 'year'],append=True).unstack().rolling(window=finestra_media_mobile_m, min_periods=1, center=True).var().stack()#.groupby('dayofyear')[['T_oss']].agg(lambda x: np.mean(x))  
# #     df = df.reset_index()
#     df.rename(columns={'T_oss':'var_oss'},inplace=True)
#     return df

def fit_harmonics(y, n_harmonics):
    n = y.shape[0]
    X = np.matrix([*[[np.sin(i*k/n*2*np.pi) for i in range(n)] for k in range(n_harmonics)], *[[np.cos(i*k/n*2*np.pi) for i in range(n)] for k in range(n_harmonics)]]).T
    X = sm.add_constant(X)

    ols_model = sm.OLS(y, X).fit()
    yhat = ols_model.predict(X)
    
    return yhat

# Calcolo percentili
def calcolo_percentili(df,column):
    if column=='mean':
        df['percentile_'+column+'_05_T_norm'] = df['T_'+column+'_m'] - 1.645 * df['std']
        df['percentile_'+column+'_95_T_norm'] = df['T_'+column+'_m'] + 1.645 * df['std']
        df['percentile_'+column+'_10_T_norm'] = df['T_'+column+'_m'] - 1.282 * df['std']
        df['percentile_'+column+'_90_T_norm'] = df['T_'+column+'_m'] + 1.282 * df['std']
        df['percentile_'+column+'_25_T_norm'] = df['T_'+column+'_m'] - 0.675 * df['std']
        df['percentile_'+column+'_75_T_norm'] = df['T_'+column+'_m'] + 0.675 * df['std']
    else:
        df['percentile_'+column+'_05_T_norm'] = df['T_'+column+'_mean'] - 1.645 * df['std']
        df['percentile_'+column+'_95_T_norm'] = df['T_'+column+'_mean'] + 1.645 * df['std']
        df['percentile_'+column+'_10_T_norm'] = df['T_'+column+'_mean'] - 1.282 * df['std']
        df['percentile_'+column+'_90_T_norm'] = df['T_'+column+'_mean'] + 1.282 * df['std']
        df['percentile_'+column+'_25_T_norm'] = df['T_'+column+'_mean'] - 0.675 * df['std']
        df['percentile_'+column+'_75_T_norm'] = df['T_'+column+'_mean'] + 0.675 * df['std']
    return df

def calcolo_percentili_yhat(df,column):
    df['percentile_'+column+'_05_yhat'] = df['yhat_'+column] - 1.645 * df['std']
    df['percentile_'+column+'_95_yhat'] = df['yhat_'+column] + 1.645 * df['std']
    df['percentile_'+column+'_10_yhat'] = df['yhat_'+column] - 1.282 * df['std']
    df['percentile_'+column+'_90_yhat'] = df['yhat_'+column] + 1.282 * df['std']
    df['percentile_'+column+'_25_yhat'] = df['yhat_'+column] - 0.675 * df['std']
    df['percentile_'+column+'_75_yhat'] = df['yhat_'+column] + 0.675 * df['std']
    return df

def calcolo_percentili_oss(df):
    df['percentile_05_yhat'] = df['yhat'] - 1.645 * df['std']
    df['percentile_95_yhat'] = df['yhat'] + 1.645 * df['std']
    df['percentile_10_yhat'] = df['yhat'] - 1.282 * df['std']
    df['percentile_90_yhat'] = df['yhat'] + 1.282 * df['std']
    df['percentile_25_yhat'] = df['yhat'] - 0.675 * df['std']
    df['percentile_75_yhat'] = df['yhat'] + 0.675 * df['std']
    return df

def calcolo_percentili_oss_T_oss(df):
    df['percentile_05_T_oss'] = df['T_oss'] - 1.645 * df['std']
    df['percentile_95_T_oss'] = df['T_oss'] + 1.645 * df['std']
    df['percentile_10_T_oss'] = df['T_oss'] - 1.282 * df['std']
    df['percentile_90_T_oss'] = df['T_oss'] + 1.282 * df['std']
    df['percentile_25_T_oss'] = df['T_oss'] - 0.675 * df['std']
    df['percentile_75_T_oss'] = df['T_oss'] + 0.675 * df['std']
    return df

# T_max_mean yhat_max
def intervalli_confidenza(df,column,n):
    df['c_i_95_'+column+'_l'] = df[column] - 1.96 * df['std']/np.sqrt(n)
    df['c_i_95_'+column+'_h'] = df[column] + 1.96 * df['std']/np.sqrt(n)
#     df['c_i_90'] = df[column] + 1.645 * df['std']/np.sqrt(n)
    return df

def lettura_temp_norm(file_t_norm,inizio,fine):
    df = pd.read_csv(file_t_norm)#,encoding='utf-16')
    
    df.rename(columns={'TIME - Date':'date','Codice Provincia':'provincia','Temperature Min':'temp_min','Temperature Max':'temp_max'},inplace=True)
#     df = df[df['provincia']==54]
    
    df['date'] = pd.to_datetime(df['date'],format='%d/%m/%Y')
    df = df[(df['date']>=inizio) & (df['date']<=fine)]
    df['T_mean'] = df[['temp_min', 'temp_max']].mean(axis=1)
    df.drop('Desc_Provincia',axis=1,inplace=True)
#     df.info()

    df = df.assign(
        dayofyear=lambda x: (x.date + pd.DateOffset(days=92)).dt.dayofyear,
        year=lambda x: (x.date + pd.DateOffset(days=92)).dt.year,
    )
#     print(df)
    df['monthyear'] = (df.date.dt.month).astype('str').str.pad(2, side='left', fillchar='0')+(df.year).astype('str')
    df['monthday'] = df['date'].dt.month.astype('str').str.pad(2, side='left', fillchar='0') + '-' + df['date'].dt.day.astype('str').str.pad(2, side='left', fillchar='0')
    return df

def lettura_osservatorio(file_oss,inizio,fine):
    df_oss = pd.read_csv(file_oss)
#     RINOMINARE A T_OSS_MIN, T_OSS_MAX E POI CREARE T_OSS_MEAN
    df_oss.rename(columns={'Cd Oss':'codice_oss','T Oss':'T_oss','Data':'date'},inplace=True)
#     df_oss = df_oss[df_oss['codice_oss']==11]
#     df_oss['T_oss'] = df_oss.apply(lambda x: 18 if x['T_oss']==0 else 18-x['T_oss'],axis=1)
    
    df_oss['date'] = df_oss['date'].astype('str')
    df_oss['date'] = pd.to_datetime(df_oss['date'],format='%Y-%m-%d', exact=False)
    df_oss = df_oss[(df_oss['date']>=inizio) & (df_oss['date']<=fine)]
    df_oss = df_oss.assign(
        dayofyear=lambda x: (x.date + pd.DateOffset(days=92)).dt.dayofyear,
        year=lambda x: (x.date + pd.DateOffset(days=92)).dt.year,
    )
    df_oss.drop('Oss',axis=1,inplace=True)
    df_oss['monthyear'] = (df_oss.date.dt.month).astype('str').str.pad(2, side='left', fillchar='0')+(df_oss.year).astype('str')
    df_oss['monthday'] = df_oss['date'].dt.month.astype('str').str.pad(2, side='left', fillchar='0') + '-' + df_oss['date'].dt.day.astype('str').str.pad(2, side='left', fillchar='0')
    return df_oss

# def calcolo_temp_norm_gg(file_t_norm,n_osservazioni):

# #     Calcolo temperature normali giornaliere    
#     df = lettura_temp_norm(file_t_norm,inizio,fine)
#     df.rename(columns={'temp_max':'T_max_mean','temp_min':'T_min_mean','T_mean':'T_mean_m'},inplace=True)
    
#     temp_norm = df.groupby(['dayofyear', 'provincia'])[['T_min_mean', 'T_max_mean','T_mean_m']].mean().unstack().reindex(range(367)).interpolate(method='linear').stack() #,'T_mean'
#     province = temp_norm.reset_index(['provincia']).provincia.unique()
#     finale = pd.DataFrame()

# #     Fit armoniche di sesto grado per curve T_min, T_max e T_mean
#     for p in province:
#         temp_p = temp_norm.query('provincia == {}'.format(p)).reset_index(['provincia'])#.assign(std_mean=lambda x: x['std'], iv_l=lambda x: (x['mean'] - 2*x['std'].rolling(window=15, center=True).mean()), iv_h=lambda x: (x['mean'] + 2*x['std'].rolling(window=15, center=True).mean()))

#         y_min = temp_p['T_min_mean'].values.ravel()
#         y_max = temp_p['T_max_mean'].values.ravel()
#         y_mean = temp_p['T_mean_m'].values.ravel()

#         x = temp_p.index
        
#         yhat_min = fit_harmonics(y_min, 6)
#         yhat_max = fit_harmonics(y_max, 6)
#         yhat_mean = fit_harmonics(y_mean, 6)

#         temp_p['yhat_min'] = yhat_min
#         temp_p['yhat_max'] = yhat_max
#         temp_p['yhat_mean'] = yhat_mean

#         finale = finale.append(temp_p)
        
# #     Calcolo gradi giorno
#     finale['gradi_giorno_yhat_mean'] = 0
#     finale['gradi_giorno_yhat_mean'] = finale.apply(lambda x: max(0,18-x['yhat_mean']),axis=1)
#     finale.reset_index(inplace=True)
    
#     finale['gradi_giorno_T_mean'] = 0
#     finale['gradi_giorno_T_mean'] = finale.apply(lambda x: max(0,18-x['T_mean_m']),axis=1)
#     finale.reset_index(inplace=True)
#     # finale.to_csv('temp_giorno_normali_primi_punti_'+file_name+'.csv',index=False)    
    
#     temp_norm = temp_norm.join(df.groupby('provincia').apply(compute_std, finestra_media_mobile_g=finestra_media_mobile_g), rsuffix='_std')
#     temp_norm = temp_norm.join(df.groupby('provincia').apply(compute_var, finestra_media_mobile_g=finestra_media_mobile_g), rsuffix='_var')
# #     temp.info()

#     temp_max = temp_norm[['T_max_mean', 'T_max_mean_std', 'T_max_mean_var']].rename(columns={'T_max_mean_std': 'std', 'T_max_mean_var': 'var'}).reset_index()
#     temp_min = temp_norm[['T_min_mean', 'T_min_mean_std', 'T_min_mean_var']].rename(columns={'T_min_mean_std': 'std', 'T_min_mean_var': 'var'}).reset_index()
#     temp_mean = temp_norm[['T_mean_m', 'T_mean_m_std', 'T_mean_m_var']].rename(columns={'T_mean_m_std': 'std', 'T_mean_m_var': 'var'}).reset_index()
    
#     temp_max = temp_max.merge(finale[['dayofyear','provincia','yhat_max']],on=['dayofyear','provincia'],how='left')
#     temp_min = temp_min.merge(finale[['dayofyear','provincia','yhat_min']],on=['dayofyear','provincia'],how='left')
#     temp_mean = temp_mean.merge(finale[['dayofyear','provincia','yhat_mean','gradi_giorno_yhat_mean','gradi_giorno_T_mean']],on=['dayofyear','provincia'],how='left')
    
#     temp_max.drop_duplicates(inplace=True)
#     temp_min.drop_duplicates(inplace=True)
#     temp_mean.drop_duplicates(inplace=True)

#     temp_max = calcolo_percentili(temp_max,'max')
#     temp_min = calcolo_percentili(temp_min,'min')
#     temp_mean = calcolo_percentili(temp_mean,'mean')

#     temp_max = calcolo_percentili_yhat(temp_max,'max')
#     temp_min = calcolo_percentili_yhat(temp_min,'min')
#     temp_mean = calcolo_percentili_yhat(temp_mean,'mean')
    
#     temp_max = intervalli_confidenza(temp_max,'T_max_mean',n_osservazioni)
#     temp_min = intervalli_confidenza(temp_min,'T_min_mean',n_osservazioni)
#     temp_mean = intervalli_confidenza(temp_mean,'T_mean_m',n_osservazioni)

#     temp_max = intervalli_confidenza(temp_max,'yhat_max',n_osservazioni)
#     temp_min = intervalli_confidenza(temp_min,'yhat_min',n_osservazioni)
#     temp_mean = intervalli_confidenza(temp_mean,'yhat_mean',n_osservazioni)
    
#     prv = pd.DataFrame()
#     prv['dayofyear'] = range(1,367)
#     prv['daymonth'] = pd.date_range('2019-10-01','2020-09-30').astype('str').str.slice(start=5)
    
#     temp_max = temp_max.merge(prv,on='dayofyear',how='left')
#     temp_min = temp_min.merge(prv,on='dayofyear',how='left')
#     temp_mean = temp_mean.merge(prv,on='dayofyear',how='left')

#     temp_max = temp_max[['dayofyear','provincia','daymonth','T_max_mean','std','var','yhat_max','percentile_max_10_T_norm','percentile_max_90_T_norm','percentile_max_10_yhat','percentile_max_90_yhat','c_i_95_T_max_mean_l','c_i_95_T_max_mean_h','c_i_95_yhat_max_l','c_i_95_yhat_max_h']]
#     temp_min = temp_min[['dayofyear','provincia','daymonth','T_min_mean','std','var','yhat_min','percentile_min_10_T_norm','percentile_min_90_T_norm','percentile_min_10_yhat','percentile_min_90_yhat','c_i_95_T_min_mean_l','c_i_95_T_min_mean_h','c_i_95_yhat_min_l','c_i_95_yhat_min_h']]
#     temp_mean = temp_mean[['dayofyear','provincia','daymonth','T_mean_m','gradi_giorno_yhat_mean','gradi_giorno_T_mean','std','var','yhat_mean','percentile_mean_10_T_norm','percentile_mean_90_T_norm','percentile_mean_10_yhat','percentile_mean_90_yhat','c_i_95_T_mean_m_l','c_i_95_T_mean_m_h','c_i_95_yhat_mean_l','c_i_95_yhat_mean_h']]
    
#     return temp_max,temp_min,temp_mean

def find_min_max_in_month(df):
    df['T_min_min'] = df['T_min'].min()
    df['T_min_max'] = df['T_min'].max()
    
    df['T_max_min'] = df['T_max'].min()
    df['T_max_max'] = df['T_max'].max()
    
    df['T_mean_min'] = df['T_mean'].min()
    df['T_mean_max'] = df['T_mean'].max()
    return df

def calcolo_temp_norm_mensile(file_t_norm,inizio,fine,finestra_media_mobile_m):

    print("Lettura file temperature per provincia.")
#     Calcolo temperature normali giornaliere    
    df = lettura_temp_norm(file_t_norm,inizio,fine)
#     print(df)
    print("Numero province presenti: ", df_oss['provincia'].nunique())
    
    print("Individuazione temperature massima e minima per mese e per provincia.")
    df['month'] = df.date.dt.month
    df.rename(columns={'temp_max':'T_max','temp_min':'T_min'},inplace=True)
    min_max_in_month = df[['month','T_min','T_max','T_mean','provincia']].groupby(['month','provincia']).apply(find_min_max_in_month)
    min_max_in_month.drop(['T_min','T_max','T_mean'],axis=1,inplace=True)    
    min_max_in_month.drop_duplicates(inplace=True)
    df.drop('month',axis=1,inplace=True)
    
    df.rename(columns={'T_max':'T_max_sum','T_min':'T_min_sum','T_mean':'T_mean_sum'},inplace=True)
    
    print("Calcolo delle medie mensili e delle varianze centrate a metà del mese.")
    prova1 = df[['provincia','T_max_sum','date','monthday','dayofyear','year']].groupby('provincia').apply(compute_std_var_month, finestra_media_mobile_m=finestra_media_mobile_m, col='T_max_sum')
    prova1.drop(['provincia','monthday'],axis=1,inplace=True)
    prova1 = prova1.reset_index()
    
    prova2 = df[['provincia','T_min_sum','date','monthday','dayofyear','year']].groupby('provincia').apply(compute_std_var_month, finestra_media_mobile_m=finestra_media_mobile_m, col='T_min_sum')
    prova2.drop(['provincia','monthday'],axis=1,inplace=True)
    prova2 = prova2.reset_index()
    
    prova3 = df[['provincia','T_mean_sum','date','monthday','dayofyear','year']].groupby('provincia').apply(compute_std_var_month, finestra_media_mobile_m=finestra_media_mobile_m, col='T_mean_sum')
    prova3.drop(['provincia','monthday'],axis=1,inplace=True)
    prova3 = prova3.reset_index()
    
    prova = prova1.merge(prova2,on=['provincia','dayofyear','year','date'],how='left')
    prova = prova.merge(prova3,on=['provincia','dayofyear','year','date'],how='left')
    
    prova.rename(columns={'T_max_sum_std':'std_t_max','T_min_sum_std':'std_t_min','T_mean_sum_std':'std_t_mean','T_max_sum_var':'var_t_max','T_min_sum_var':'var_t_min','T_mean_sum_var':'var_t_mean'},inplace=True)
    totale = df.merge(prova,on=['provincia','dayofyear','year','date'],how='left')

    n_prov = df.provincia.nunique()
    tmp = (df[['monthyear','date']].groupby('monthyear').count()/n_prov)
    tmp.rename(columns={'date':'n_giorni'},inplace=True)
    tmp['n_giorni'] = tmp['n_giorni'].astype('int')
    tmp['central'] = (tmp['n_giorni']/2).apply(np.ceil)
    tmp['central'] = tmp['central'].astype('int')
    tmp = tmp.reset_index()
    tmp['monthyear_minus'] = tmp['monthyear'].str.slice(stop=2) + '-' + tmp['monthyear'].str.slice(start=2)
    tmp['central'] = tmp['central'].mask(tmp['monthyear_minus'].str.contains('02-'),15)
    tmp.drop('monthyear_minus',axis=1,inplace=True)
    tmp = tmp.reset_index()
    # tmp
    totale = totale.merge(tmp,on='monthyear',how='left')

    provvi = totale[totale['date'].dt.day==totale['central']]

    provvi['month'] = provvi.date.dt.month
    provvi = provvi.groupby('month').apply(most_frequent_days)
    
    provvi['std_t_min'] = provvi['std_t_min']*provvi['n_giorni']
    provvi['std_t_max'] = provvi['std_t_max']*provvi['n_giorni']
    provvi['std_t_mean'] = provvi['std_t_mean']*provvi['n_giorni']
    provvi['var_t_min'] = provvi['var_t_min']*provvi['n_giorni']
    provvi['var_t_max'] = provvi['var_t_max']*provvi['n_giorni']
    provvi['var_t_mean'] = provvi['var_t_mean']*provvi['n_giorni']
    provvi['month'] = provvi['month'].astype('str').str.pad(2, side='left', fillchar='0')
    
    temp = df.groupby(['monthyear', 'provincia'])[['T_min_sum', 'T_max_sum','T_mean_sum']].sum().unstack().interpolate(method='linear').stack().reset_index() #,'T_mean'
    temp.rename(columns={'monthyear':'month'},inplace=True)
    temp['month'] = temp['month'].str.slice(start=0,stop=2)
    #temp.drop_duplicates(subset=['provincia','month'],inplace=True)
    temp = temp.groupby(['month', 'provincia'])[['T_min_sum', 'T_max_sum','T_mean_sum']].mean().unstack().interpolate(method='linear').stack() #.reset_index() #,'T_mean'

    temp = temp.reset_index()
    
    temp = temp.merge(provvi,on=['provincia','month'],how='left')
    temp['month'] = temp['month'].astype('int')
#     print(temp)
    
    temp = temp.merge(min_max_in_month,on=['month','provincia'],how='left')
#     print(provvi.info())
    temp = temp[['month','provincia','T_min_sum','T_min_min','T_min_max','T_max_sum','T_max_min','T_max_max','T_mean_sum','T_mean_min','T_mean_max','std_t_min','std_t_max','std_t_mean','var_t_min','var_t_max','var_t_mean']]
    #temp.rename(columns={'T_min_sum':'t_min_sum','T_min_min':'t_min_min','T_min_max':'t_min_max','T_max_sum':'t_max_sum','T_max_min':'t_max_min','T_max_max':'t_max_max','T_mean_sum':'t_mean_sum','T_mean_min':'t_mean_min','T_mean_max':'t_mean_max'},inplace=True)
    return temp

def most_frequent_days(df):
    df['n_giorni'] = df.n_giorni.mode()[0]
    return df

def find_min_max_in_month_oss(df):
    df['T_oss_min'] = df['T_oss'].min()
    df['T_oss_max'] = df['T_oss'].max()
    return df

def calcolo_oss_mensile(file_oss,inizio,fine,finestra_media_mobile_m):
    
    print("Lettura file temperature per osservatorio.")
    df_oss = lettura_osservatorio(file_oss,inizio,fine)
    
    print("Numero osservatori presenti: ", df_oss['codice_oss'].nunique())

    print("Individuazione temperature massima e minima per mese e per osservatorio.")
    df_oss['month'] = df_oss.date.dt.month
    min_max_in_month = df_oss[['month','T_oss','codice_oss']].groupby(['month','codice_oss']).apply(find_min_max_in_month_oss)
    min_max_in_month.drop('T_oss',axis=1,inplace=True)    
    min_max_in_month.drop_duplicates(inplace=True)
    df_oss.drop('month',axis=1,inplace=True)
#     print(df_oss[df_oss['monthyear']=='102006'])

    prova_oss = df_oss[['codice_oss','T_oss','date','monthday','dayofyear','year']].groupby('codice_oss').apply(compute_std_var_month, finestra_media_mobile_m=finestra_media_mobile_m, col='T_oss')
    prova_oss.drop(['codice_oss','monthday','T_oss'],axis=1,inplace=True)
    prova_oss = prova_oss.reset_index()
    
    totale_oss = df_oss.merge(prova_oss,on=['codice_oss','dayofyear','year','date'],how='left')
    
    print("Calcolo delle medie mensili e delle varianze centrate a metà del mese.")
    n_oss = df_oss.codice_oss.nunique()
    tmp_oss = (df_oss[['monthyear','date']].groupby('monthyear').count()/n_oss)
    tmp_oss.rename(columns={'date':'n_giorni'},inplace=True)
    tmp_oss['n_giorni'] = tmp_oss['n_giorni'].astype('int')

    tmp_oss['central'] = (tmp_oss['n_giorni']/2).apply(np.ceil)
    tmp_oss['central'] = tmp_oss['central'].astype('int')
    tmp_oss = tmp_oss.reset_index()
    tmp_oss['monthyear_minus'] = tmp_oss['monthyear'].str.slice(stop=2) + '-' + tmp_oss['monthyear'].str.slice(start=2)
    tmp_oss['central'] = tmp_oss['central'].mask(tmp_oss['monthyear_minus'].str.contains('02-'),15)
    tmp_oss.drop('monthyear_minus',axis=1,inplace=True)
    tmp_oss = tmp_oss.reset_index()

    totale_oss = totale_oss.merge(tmp_oss,on='monthyear',how='left')
    
    provvi_oss = totale_oss[totale_oss['date'].dt.day==totale_oss['central']]
    provvi_oss['month'] = provvi_oss.date.dt.month
    provvi_oss = provvi_oss.groupby('month').apply(most_frequent_days)

#     n_giorni è diverso!!!!!!!!! why? per 29 febbraio ok, ma marzo, aprile...?
#     print(provvi_oss[provvi_oss['month']==2].n_gior.unique())
    provvi_oss['T_oss_std'] = provvi_oss['T_oss_std']*provvi_oss['n_giorni']
    provvi_oss['T_oss_var'] = provvi_oss['T_oss_var']*provvi_oss['n_giorni']
    provvi_oss = provvi_oss[['month','T_oss','codice_oss','T_oss_std','T_oss_var','n_giorni']]
    provvi_oss['month'] = provvi_oss['month'].astype('str').str.pad(2, side='left', fillchar='0')
#     print(provvi_oss[provvi_oss['month']=='02']['n_giorni'].unique())
#     (provvi_oss[(provvi_oss['month']==3) & (provvi_oss['codice_oss']==11)].to_csv('std_appena_calcolata_oss_prov.csv',index=False))
    
    temp = df_oss.groupby(['monthyear', 'codice_oss'])[['T_oss']].sum().unstack().interpolate(method='linear').stack().reset_index() #,'T_mean'
    temp.rename(columns={'monthyear':'month'},inplace=True)
    temp['month'] = temp['month'].str.slice(start=0,stop=2)
#     print(temp[(temp['codice_oss']==11) & (temp['month']=='01')])
    #temp.drop_duplicates(subset=['provincia','month'],inplace=True)
    temp = temp.groupby(['month', 'codice_oss'])[['T_oss']].mean().unstack().interpolate(method='linear').stack()#.reset_index() #,'T_mean'
#     temp.to_csv('std_appena_calcolata_oss.csv',index=False)
    
    temp = temp.reset_index()
    temp['month'] = temp['month'].astype(int)
    provvi_oss['month'] = provvi_oss['month'].astype(int)
    #print(temp.info())
    provvi_oss.drop(['T_oss'],axis=1,inplace=True)
    temp = temp.merge(provvi_oss,on=['codice_oss','month'],how='left')
    
    temp = temp.merge(min_max_in_month[['month','T_oss_min','T_oss_max','codice_oss']],on=['month','codice_oss'],how='left')
    temp = temp[['month','codice_oss','T_oss','T_oss_min','T_oss_max','T_oss_std','T_oss_var']]
#     print(temp.info())
    return temp


def main_mensile(file_input,inizio,fine,finestra_media_mobile_m,path_to_output):
    
    print('Funzione dedicata al calcolo delle temperature mensili.')
    print('\n')
    
    my_bucket = 'zus-prod-s3'
#     my_bucket=''
#     file_input = my_bucket + file_input
#     path_to_output = my_bucket + path_to_output

    finestra_media_mobile_m = int(finestra_media_mobile_m)
    
    inizio_dt = pd.to_datetime(inizio,format="%Y-%m-%d")
    fine_dt = pd.to_datetime(fine,format="%Y-%m-%d")
    
#     years = fine_dt.year - inizio_dt.year
#     if years==0:
#         n_osservazioni = finestra_media_mobile_m
#     else:
#         n_osservazioni = finestra_media_mobile_m*years
    
    # IDRUN:
    file_name_out = inizio.replace('-','') + '_' + fine.replace('-','') + str(datetime.now())[0:-7].replace('-','').replace(' ','').replace(':','')

    df = pd.read_csv('s3://'+my_bucket+'/'+file_input)#,encoding='utf-16')
    #df = file_input.copy()
    if 'Codice Provincia' in df.columns:
        print('Calcolo temperatura normale mensile')
        temp_norm_mensile = calcolo_temp_norm_mensile('s3://'+my_bucket+'/'+file_input,inizio_dt,fine_dt,finestra_media_mobile_m)
#         print(temp_norm_mensile.info())
#         temp_norm_mensile['T_min_sum']=temp_norm_mensile.groupby(['provincia','month'])['T_min_sum'].transform('mean')
#         temp_norm_mensile['T_max_sum']=temp_norm_mensile.groupby(['provincia','month'])['T_max_sum'].transform('mean')
#         temp_norm_mensile['T_mean_sum']=temp_norm_mensile.groupby(['provincia','month'])['T_mean_sum'].transform('mean')
#         temp_norm_mensile[temp_norm_mensile['month']==6].to_csv('debug.csv',index=False)



        temp_norm_mensile.drop_duplicates(inplace=True)
        temp_norm_mensile.to_csv('s3://'+my_bucket+'/'+path_to_output+'m_prov/'+file_name_output+'/m_prov.csv',index=False)
        idrun = 's3://'+my_bucket+'/'+path_to_output+'m_prov/'+file_name_output+'/m_prov.csv'
        metadatati = pd.DataFrame(data={'MODELLO':['m_prov']*5,'ID_RUN':[idrun]*5,'NOME_PARAMETRO':['FILE_INPUT','INIZIO','FINE','FINESTRA_MEDIA_MOBILE_M','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[file_input,inizio,fine,finestra_media_mobile_m,path_to_output+'m_prov/best/m_prov_'+file_name_out+'.csv']})
        metadatati.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/m_prov/'+file_name_out+'/metadati.csv',index=False)
#         print(temp_norm_mensile[(temp_norm_mensile['provincia']==1) & (temp_norm_mensile['month']==1)])
#         temp_norm_mensile.to_csv('prova_mensile_prov.csv',index=False)
    #     print(temp_mensile)
        print('\n')
    else:
        print('Calcolo osservatorio mensile')
        temp_oss_mensile = calcolo_oss_mensile('s3://'+my_bucket+'/'+file_input,inizio_dt,fine_dt,finestra_media_mobile_m)
#         print(temp_oss_mensile.info())
#         temp_oss_mensile[(temp_oss_mensile['month']==2) & (temp_oss_mensile['codice_oss']==11)].to_csv('prova_prima_mean_loc.csv',index=False)
#         temp_oss_mensile['T_oss']=temp_oss_mensile.groupby(['codice_oss','month'])['T_oss'].transform('mean')
#         temp_oss_mensile.to_csv('prova_dopo_mean.csv',index=False)

        temp_oss_mensile.drop_duplicates(inplace=True)
        temp_oss_mensile.rename(columns={'T_oss':'T_oss_sum'},inplace=True)
        temp_oss_mensile.to_csv('s3://'+my_bucket+'/'+path_to_output+'m_oss/'+file_name_output+'/m_oss.csv',index=False)   
        idrun = 's3://'+my_bucket+'/'+path_to_output+'m_oss/'+file_name_output+'/m_oss.csv'
        metadatati = pd.DataFrame(data={'MODELLO':['m_oss']*5,'ID_RUN':[idrun]*5,'NOME_PARAMETRO':['FILE_INPUT','INIZIO','FINE','FINESTRA_MEDIA_MOBILE_M','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[file_input,inizio,fine,finestra_media_mobile_m,path_to_output+'m_oss/best/m_oss_'+file_name_out+'.csv']})
        metadatati.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/m_oss/'+file_name_out+'/metadati.csv',index=False)
#         temp_oss_mensile.to_csv('prova_mensile_oss.csv',index=False)
    #     print(temp_max)
        print('\n')
    print("Procedura terminata")
    
#     return temp_norm_mensile, temp_oss_mensile

# file_name = str(datetime.now())[0:-7].replace('-','_').replace(' ','_').replace(':','_')
# temp_max_norm, temp_min_norm, temp_mean_norm, temp_norm_mensile, temp_oss_gg, temp_oss_mensile = main_gg_mensile('Storico_Temp_2005-2015.csv','T_x_OSS.csv',150)
