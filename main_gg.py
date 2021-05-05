
import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
import statsmodels.api as sm
# from scipy import stats
from datetime import datetime,timedelta
import calendar as c

def compute_std_var(df_in,finestra_media_mobile_g,col):
    df = df_in.copy()
#     print(df_in)#.drop('provincia',axis=1).reset_index())
    days = int(np.floor(finestra_media_mobile_g/2))
    mds = sorted(df.monthday.unique())
    min_md = df[df['date']==min(df['date'])]['monthday'].ravel()[0]
    max_md = df[df['date']==max(df['date'])]['monthday'].ravel()[0]
    df['isleap'] = 0
    df['isleap'] = df.apply(lambda x: 1 if c.isleap(x['date'].year) else 0,axis=1)
    to_append = pd.DataFrame()
    for md in mds:
#         print(md)
        if md!='02-29':
            lista = pd.date_range(datetime.strptime(str(md), "%m-%d")-timedelta(days=days), datetime.strptime(str(md), "%m-%d")+timedelta(days=days)).strftime("%m-%d").tolist()
            var = df[df.monthday.isin(lista)][col].var(ddof=0)
#             print(df[df.monthday.isin(lista)])
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

# def compute_std(df,finestra_media_mobile_g):
# #     df['monthday'] = df['date'].dt.month.astype('str').str.pad(2, side='left', fillchar='0') + '-' + df['date'].dt.day.astype('str').str.pad(2, side='left', fillchar='0')
#     return df.set_index(['dayofyear', 'year'], append=True).unstack().rolling(window=finestra_media_mobile_g, min_periods=1, center=True).var().stack().groupby('dayofyear')[['T_min_mean', 'T_max_mean','T_mean_m']].agg(lambda x: np.sqrt(np.mean(x)))    

# def compute_var(df,finestra_media_mobile_g):
#     return df.set_index(['dayofyear', 'year'], append=True).unstack().rolling(window=finestra_media_mobile_g, min_periods=1, center=True).var().stack().groupby('dayofyear')[['T_min_mean', 'T_max_mean','T_mean_m']].agg(lambda x: np.mean(x))

def compute_std_month(df,finestra_media_mobile_m):
#     print(df)
    df = df.set_index(['dayofyear', 'year'], append=True).unstack().rolling(window=finestra_media_mobile_m, min_periods=1, center=True).var().stack().groupby('dayofyear')[['T_min_sum', 'T_max_sum','T_mean_sum']].agg(lambda x: np.sqrt(np.mean(x)))  
    df = df.reset_index()
    df.rename(columns={'T_min_sum':'std_t_min', 'T_max_sum':'std_t_max','T_mean_sum':'std_t_mean'},inplace=True)
    return df
    
def compute_var_month(df,finestra_media_mobile_m):
    df = df.set_index(['dayofyear', 'year'], append=True).unstack().rolling(window=finestra_media_mobile_m, min_periods=1, center=True).var().stack().groupby('dayofyear')[['T_min_sum', 'T_max_sum','T_mean_sum']].agg(lambda x: np.mean(x))  
    df = df.reset_index()
    df.rename(columns={'T_min_sum':'var_t_min', 'T_max_sum':'var_t_max','T_mean_sum':'var_t_mean'},inplace=True)
    return df

def compute_std_month_oss(df,finestra_media_mobile_m):
    df = df.set_index(['dayofyear', 'year'],append=True).unstack().rolling(window=finestra_media_mobile_m, min_periods=1, center=True).var().stack().groupby('dayofyear')[['T_oss']].agg(lambda x: np.sqrt(np.mean(x)))  
    df = df.reset_index()
    df.rename(columns={'T_oss':'std_oss'},inplace=True)
    return df
    
def compute_var_month_oss(df,finestra_media_mobile_m):
    df = df.set_index(['dayofyear', 'year'],append=True).unstack().rolling(window=finestra_media_mobile_m, min_periods=1, center=True).var().stack().groupby('dayofyear')[['T_oss']].agg(lambda x: np.mean(x))  
    df = df.reset_index()
    df.rename(columns={'T_oss':'var_oss'},inplace=True)
    return df

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

def calcolo_percentili_oss_yhat(df):
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

def intervalli_confidenza(df,column):
    df['c_i_95_'+column+'_l'] = df[column] - 1.96 * df['std']/np.sqrt(df['n_osservazioni'])
    df['c_i_95_'+column+'_h'] = df[column] + 1.96 * df['std']/np.sqrt(df['n_osservazioni'])
    return df

def lettura_temp_norm(file_t_norm,inizio,fine,finestra_media_mobile_g):
    df = pd.read_csv(file_t_norm)#,encoding='utf-16')
    
    df.rename(columns={'TIME - Date':'date','Codice Provincia':'provincia','Temperature Min':'temp_min','Temperature Max':'temp_max'},inplace=True)
    
    df['date'] = pd.to_datetime(df['date'],format='%d/%m/%Y')
    df = df[(df['date']>=inizio) & (df['date']<=fine)]
    df['T_mean'] = df[['temp_min', 'temp_max']].mean(axis=1)
    df.drop('Desc_Provincia',axis=1,inplace=True)

    df = df.assign(
        dayofyear=lambda x: (x.date).dt.dayofyear, # + pd.DateOffset(days=92)
        year=lambda x: (x.date).dt.year, # + pd.DateOffset(days=92)
    )
#     df_con = df.assign(
#         dayofyear=lambda x: (x.date + pd.DateOffset(days=92)).dt.dayofyear, # + pd.DateOffset(days=92)
#         year=lambda x: (x.date + pd.DateOffset(days=92)).dt.year, # + pd.DateOffset(days=92)
#     )

#     QUA SOTTO METTERE <=9 PER OFFSET=92, SENZA OFFSET METTERE <=12
    df['dayofyear'] = df.apply(lambda x: x['dayofyear']+1 if (c.isleap(x['year'])==False and x['date'].month>=3 and x['date'].month<=12) else x['dayofyear'],axis=1)
    
    df['monthyear'] = (df.date.dt.month).astype('str').str.pad(2, side='left', fillchar='0')+(df.year).astype('str')
#     df_con['monthyear'] = (df_con.date.dt.month).astype('str').str.pad(2, side='left', fillchar='0')+(df_con.year).astype('str')
    
#     df.to_csv('df_senza_92.csv',index=False)
#     df_con.to_csv('df_con_92.csv',index=False)
#     ciao
#     Raggruppo per provincia e monthyear
    tmp = (df[['monthyear','date','provincia']].groupby(['monthyear','provincia']).count()).reset_index()
    tmp['monthyear'] = tmp['monthyear'].astype('str')
    tmp['monthyear'] = tmp['monthyear'].str.slice(stop=2)
    tmp.rename(columns={'date':'n_anni','monthyear':'month'},inplace=True)
    tmp['n_anni'] = 1
    
    tmp = tmp.groupby(['month','provincia']).count().reset_index()
    
    df = df.merge(tmp,on=['provincia'],how='left')
    
    df['n_osservazioni'] = df['n_anni']*finestra_media_mobile_g

    df.drop(['n_anni','month'],axis=1,inplace=True)
    df.drop_duplicates(inplace=True)
    df['monthday'] = df['date'].dt.month.astype('str').str.pad(2, side='left', fillchar='0') + '-' + df['date'].dt.day.astype('str').str.pad(2, side='left', fillchar='0')   
    return df

def lettura_osservatorio(file_oss,inizio,fine,finestra_media_mobile_g):
    df_oss = pd.read_csv(file_oss)
    df_oss.rename(columns={'Cd Oss':'codice_oss','T Oss':'T_oss','Data':'date'},inplace=True)
    
    df_oss['date'] = df_oss['date'].astype('str')
    df_oss['date'] = pd.to_datetime(df_oss['date'],format='%Y-%m-%d', exact=False)
    df_oss = df_oss[(df_oss['date']>=inizio) & (df_oss['date']<=fine)]
    df_oss = df_oss.assign(
        dayofyear=lambda x: (x.date).dt.dayofyear, # + pd.DateOffset(days=92)
        year=lambda x: (x.date).dt.year, # + pd.DateOffset(days=92)
    )
    
    df_oss['dayofyear'] = df_oss.apply(lambda x: x['dayofyear']+1 if (c.isleap(x['year'])==False and x['date'].month>=3 and x['date'].month<=12) else x['dayofyear'],axis=1)
    
    df_oss.drop('Oss',axis=1,inplace=True)
    df_oss['monthyear'] = (df_oss.date.dt.month).astype('str').str.pad(2, side='left', fillchar='0')+(df_oss.date.dt.year).astype('str')
    tmp = (df_oss[['monthyear','date','codice_oss']].groupby(['monthyear','codice_oss']).count()).reset_index()
    tmp.rename(columns={'date':'n_anni'},inplace=True) #,'monthyear':'month'
    tmp['n_anni'] = 1
    tmp['monthyear'] = tmp['monthyear'].astype('str')
    tmp['month'] = tmp['monthyear'].str.slice(stop=2)

    tmp = tmp.groupby(['month','codice_oss']).count().reset_index()
    tmp.drop('monthyear',axis=1,inplace=True)
    df_oss['month'] = df_oss['monthyear'].str.slice(stop=2)
    df_oss = df_oss.merge(tmp,on=['codice_oss','month'],how='left')
    
    df_oss['n_osservazioni'] = df_oss['n_anni']*finestra_media_mobile_g
    
    df_oss.drop(['n_anni'],axis=1,inplace=True)
    df_oss['monthday'] = df_oss['date'].dt.month.astype('str').str.pad(2, side='left', fillchar='0') + '-' + df_oss['date'].dt.day.astype('str').str.pad(2, side='left', fillchar='0')
    return df_oss

def calcolo_temp_norm_gg(file_t_norm,inizio,fine,finestra_media_mobile_g):

#     Calcolo temperature normali giornaliere
    print("Lettura file temperature per provincia.")
    df = lettura_temp_norm(file_t_norm,inizio,fine,finestra_media_mobile_g)
    
    print("Numero province presenti: ",df['provincia'].nunique())
    
    print("Calcolo deviazione standard.")
    df_max = df.groupby('provincia').apply(compute_std_var, finestra_media_mobile_g=finestra_media_mobile_g, col='temp_max')[['provincia','date','temp_max_var','temp_max_std']]
    df_min = df.groupby('provincia').apply(compute_std_var, finestra_media_mobile_g=finestra_media_mobile_g, col='temp_min')[['provincia','date','temp_min_var','temp_min_std']]
    df_mean = df.groupby('provincia').apply(compute_std_var, finestra_media_mobile_g=finestra_media_mobile_g, col='T_mean')[['provincia','date','T_mean_var','T_mean_std']]

    df_max.drop('provincia',axis=1,inplace=True)
    df_min.drop('provincia',axis=1,inplace=True)
    df_mean.drop('provincia',axis=1,inplace=True)
    
    df_max.reset_index(inplace=True)
    df_min.reset_index(inplace=True)
    df_mean.reset_index(inplace=True)
    
    df = df.merge(df_max, on=['provincia','date'], how='left')
    df = df.merge(df_min, on=['provincia','date'], how='left')
    df = df.merge(df_mean, on=['provincia','date'], how='left')
    
    df.rename(columns={'temp_max':'T_max_mean','temp_min':'T_min_mean','T_mean':'T_mean_m'},inplace=True)
    
#     df.to_csv("df_senza_92.csv",index=False)
    
    max_min_gg = df.groupby(['dayofyear','provincia']).agg({'T_min_mean':['min'],'T_max_mean':['max']})
    max_min_gg.columns = max_min_gg.columns.droplevel(0)
    max_min_gg = max_min_gg.reset_index()
    max_min_gg.rename(columns={'min':'T_min_abs','max':'T_max_abs'},inplace=True)
    
    print("Calcolo media giornaliera.")
    temp_norm = df.groupby(['dayofyear', 'provincia'])[['T_min_mean', 'T_max_mean','T_mean_m', 'temp_max_std', 'temp_max_var', 'temp_min_std', 'temp_min_var', 'T_mean_std', 'T_mean_var']].mean().unstack().reindex(range(367)).interpolate(method='linear').stack()#.reset_index() #,'T_mean'
    
    province = temp_norm.reset_index(['provincia']).provincia.unique()
    finale = pd.DataFrame()

    print("Inizio procedura di fit con armoniche di sesto grado.")
#     Fit armoniche di sesto grado per curve T_min, T_max e T_mean
    for p in province:
        temp_p = temp_norm.query('provincia == {}'.format(p)).reset_index(['provincia','dayofyear'])#.assign(std_mean=lambda x: x['std'], iv_l=lambda x: (x['mean'] - 2*x['std'].rolling(window=15, center=True).mean()), iv_h=lambda x: (x['mean'] + 2*x['std'].rolling(window=15, center=True).mean()))
        ###########################
#         temp_p = temp_p.merge(tmp,on='')
        ###########################
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
        
#         if p==1:
# #             temp_p.to_csv('temp_p_con_92.csv',index=False)
#             print(y_mean)

        finale = finale.append(temp_p)
        
#     Calcolo gradi giorno

#################
#     ciao
#################
    print("Conversione in gradi giorno.")
    finale['gradi_giorno_yhat_mean'] = 0
    finale['gradi_giorno_yhat_mean'] = finale.apply(lambda x: max(0,18-x['yhat_mean']),axis=1)
    finale.reset_index(inplace=True)
    
    finale['gradi_giorno_T_mean'] = 0
    finale['gradi_giorno_T_mean'] = finale.apply(lambda x: max(0,18-x['T_mean_m']),axis=1)
    finale.reset_index(inplace=True)
    
    selezione = df[['provincia','n_osservazioni']]
    selezione = selezione.groupby('provincia').apply(highest_number)
    selezione.drop_duplicates(inplace=True)
    finale = finale.merge(selezione,on=['provincia'],how='left')

    temp_max = temp_norm[['T_max_mean', 'temp_max_std', 'temp_max_var']].rename(columns={'temp_max_std': 'std', 'temp_max_var': 'var'}).reset_index()
    temp_min = temp_norm[['T_min_mean', 'temp_min_std', 'temp_min_var']].rename(columns={'temp_min_std': 'std', 'temp_min_var': 'var'}).reset_index()
    temp_mean = temp_norm[['T_mean_m', 'T_mean_std', 'T_mean_var']].rename(columns={'T_mean_std': 'std', 'T_mean_var': 'var'}).reset_index()

    temp_max = temp_max.merge(finale[['dayofyear','provincia','yhat_max','n_osservazioni']],on=['dayofyear','provincia'],how='left')
    temp_min = temp_min.merge(finale[['dayofyear','provincia','yhat_min','n_osservazioni']],on=['dayofyear','provincia'],how='left')
    temp_mean = temp_mean.merge(finale[['dayofyear','provincia','yhat_mean','gradi_giorno_yhat_mean','gradi_giorno_T_mean','n_osservazioni']],on=['dayofyear','provincia'],how='left')

    print("Calcolo percentili per curve di minimo, massimo e media.")
    temp_max = calcolo_percentili(temp_max,'max')
    temp_min = calcolo_percentili(temp_min,'min')
    temp_mean = calcolo_percentili(temp_mean,'mean')

    temp_max = calcolo_percentili_yhat(temp_max,'max')
    temp_min = calcolo_percentili_yhat(temp_min,'min')
    temp_mean = calcolo_percentili_yhat(temp_mean,'mean')

    print("Calcolo intervalli di confidenza per curve di minimo, massimo e media.")
    temp_max = intervalli_confidenza(temp_max,'T_max_mean')
    temp_min = intervalli_confidenza(temp_min,'T_min_mean')
    temp_mean = intervalli_confidenza(temp_mean,'T_mean_m')
    
    temp_max = intervalli_confidenza(temp_max,'yhat_max')
    temp_min = intervalli_confidenza(temp_min,'yhat_min')
    temp_mean = intervalli_confidenza(temp_mean,'yhat_mean')
    
    prv = pd.DataFrame()
    prv['dayofyear'] = range(1,367)
    prv['monthday'] = pd.date_range('2020-01-01','2020-12-31').astype('str').str.slice(start=5) #'2019-10-01','2020-09-30' for offset=92!
    
    temp_max = temp_max.merge(prv,on='dayofyear',how='left')
    temp_min = temp_min.merge(prv,on='dayofyear',how='left')
    temp_mean = temp_mean.merge(prv,on='dayofyear',how='left')

    temp_max['mese'] = temp_max['monthday'].str.slice(stop=2).astype('int')
    temp_max['giorno'] = temp_max['monthday'].str.slice(start=-2).astype('int')
    
    temp_min['mese'] = temp_min['monthday'].str.slice(stop=2).astype('int')
    temp_min['giorno'] = temp_min['monthday'].str.slice(start=-2).astype('int')
    
    temp_mean['mese'] = temp_mean['monthday'].str.slice(stop=2).astype('int')
    temp_mean['giorno'] = temp_mean['monthday'].str.slice(start=-2).astype('int')

    temp_max = temp_max.merge(max_min_gg,on=['dayofyear','provincia'],how='left')
    temp_min = temp_min.merge(max_min_gg,on=['dayofyear','provincia'],how='left')
    temp_mean = temp_mean.merge(max_min_gg,on=['dayofyear','provincia'],how='left')

    temp_max = temp_max[['dayofyear','mese','giorno','provincia','T_max_mean','T_min_abs','T_max_abs','std','var','yhat_max','percentile_max_10_T_norm','percentile_max_90_T_norm','percentile_max_10_yhat','percentile_max_90_yhat','c_i_95_T_max_mean_l','c_i_95_T_max_mean_h','c_i_95_yhat_max_l','c_i_95_yhat_max_h']]
    temp_min = temp_min[['dayofyear','mese','giorno','provincia','T_min_mean','T_min_abs','T_max_abs','std','var','yhat_min','percentile_min_10_T_norm','percentile_min_90_T_norm','percentile_min_10_yhat','percentile_min_90_yhat','c_i_95_T_min_mean_l','c_i_95_T_min_mean_h','c_i_95_yhat_min_l','c_i_95_yhat_min_h']]
    temp_mean = temp_mean[['dayofyear','mese','giorno','provincia','T_mean_m','T_min_abs','T_max_abs','gradi_giorno_yhat_mean','gradi_giorno_T_mean','std','var','yhat_mean','percentile_mean_10_T_norm','percentile_mean_90_T_norm','percentile_mean_10_yhat','percentile_mean_90_yhat','c_i_95_T_mean_m_l','c_i_95_T_mean_m_h','c_i_95_yhat_mean_l','c_i_95_yhat_mean_h']]
    
#     temp_max.rename(columns={'T_min_abs':'t_min_abs','T_max_abs':'t_max_abs','T_max_mean':'t_mean','yhat_max':'yhat','percentile_max_10_T_norm':'percentile_10_t_mean','percentile_max_90_T_norm':'percentile_90_t_mean','percentile_max_10_yhat':'percentile_10_yhat','percentile_max_90_yhat':'percentile_90_yhat','c_i_95_T_max_mean_l':'c_i_95_t_mean_l','c_i_95_T_max_mean_h':'c_i_95_t_mean_u','c_i_95_yhat_max_l':'c_i_95_yhat_l','c_i_95_yhat_max_h':'c_i_95_yhat_u'},inplace=True)
#     temp_min.rename(columns={'T_min_abs':'t_min_abs','T_max_abs':'t_max_abs','T_min_mean':'t_mean','yhat_min':'yhat','percentile_min_10_T_norm':'percentile_10_t_mean','percentile_min_90_T_norm':'percentile_90_t_mean','percentile_min_10_yhat':'percentile_10_yhat','percentile_min_90_yhat':'percentile_90_yhat','c_i_95_T_min_mean_l':'c_i_95_t_mean_l','c_i_95_T_min_mean_h':'c_i_95_t_mean_u','c_i_95_yhat_min_l':'c_i_95_yhat_l','c_i_95_yhat_min_h':'c_i_95_yhat_u'},inplace=True)
#     temp_mean.rename(columns={'T_min_abs':'t_min_abs','T_max_abs':'t_max_abs','T_mean_m':'t_mean','yhat_mean':'yhat','percentile_mean_10_T_norm':'percentile_10_t_mean','percentile_mean_90_T_norm':'percentile_90_t_mean','percentile_mean_10_yhat':'percentile_10_yhat','percentile_mean_90_yhat':'percentile_90_yhat','c_i_95_T_mean_m_l':'c_i_95_t_mean_l','c_i_95_T_mean_m_h':'c_i_95_t_mean_u','c_i_95_yhat_mean_l':'c_i_95_yhat_l','c_i_95_yhat_mean_h':'c_i_95_yhat_u'},inplace=True)

    return temp_max,temp_min,temp_mean

def highest_number(df):
    df['n_osservazioni'] = df.n_osservazioni.max()
    return df

def calcolo_oss_gg(file_oss,inizio,fine,finestra_media_mobile_g):

    print("Lettura file temperature per osservatorio.")
    df_oss = lettura_osservatorio(file_oss,inizio,fine,finestra_media_mobile_g)
    
    print("Numero osservatori presenti: ",df_oss['codice_oss'].nunique())

    df_oss = df_oss.assign(
        dayofyear=lambda x: (x.date).dt.dayofyear, # + pd.DateOffset(days=92)
        year=lambda x: (x.date).dt.year, # + pd.DateOffset(days=92)
    )
    print("Calcolo deviazione standard.")
    df_oss_std_var = df_oss.groupby('codice_oss').apply(compute_std_var, finestra_media_mobile_g=finestra_media_mobile_g, col='T_oss')[['codice_oss','date','T_oss_var','T_oss_std']]
    
    df_oss_std_var.drop('codice_oss',axis=1,inplace=True)

    df_oss_std_var.reset_index(inplace=True)
    df_oss = df_oss.merge(df_oss_std_var,on=['codice_oss','date'],how='left')

    print("Calcolo media giornaliera.")
    temp_oss = df_oss.groupby(['dayofyear', 'codice_oss'])[['T_oss','T_oss_var','T_oss_std']].mean().unstack().reindex(range(367)).interpolate(method='linear').stack() #,'T_mean'

    osservatori = temp_oss.reset_index(['codice_oss']).codice_oss.unique()
    finale_oss = pd.DataFrame()

    print("Inizio procedura di fit con armoniche di sesto grado.")
    for o in osservatori:
        temp_o = temp_oss.query('codice_oss == {}'.format(o)).reset_index(['codice_oss'])#.assign(std_mean=lambda x: x['std'], iv_l=lambda x: (x['mean'] - 2*x['std'].rolling(window=15, center=True).mean()), iv_h=lambda x: (x['mean'] + 2*x['std'].rolling(window=15, center=True).mean()))

        y = temp_o['T_oss'].values.ravel()
        x = temp_o.index
        
        yhat = fit_harmonics(y, 6)

        temp_o['yhat'] = yhat

        finale_oss = finale_oss.append(temp_o)
        
    print("Conversione in gradi giorno.")
    finale_oss['gradi_giorno_yhat'] = 0
    finale_oss['gradi_giorno_yhat'] = finale_oss.apply(lambda x: max(0,18-x['yhat']),axis=1)
    finale_oss.reset_index(inplace=True)#[finale['T_mean_m']>=18]

    finale_oss['gradi_giorno_T_oss'] = 0
    finale_oss['gradi_giorno_T_oss'] = finale_oss.apply(lambda x: max(0,18-x['T_oss']),axis=1)
    finale_oss.reset_index(inplace=True)#[finale['T_mean_m']>=18]
    
    selezione = df_oss[['codice_oss','n_osservazioni']]
    selezione = selezione.groupby('codice_oss').apply(highest_number)
    selezione.drop_duplicates(inplace=True)
        
    finale_oss = finale_oss.merge(selezione,on=['codice_oss'],how='left')

    temp_oss = temp_oss.reset_index()
    temp_oss.drop_duplicates(inplace=True)
    temp_oss = temp_oss.set_index(['dayofyear','codice_oss'])

    temp_oss_sv = temp_oss[['T_oss', 'T_oss_std', 'T_oss_var']].rename(columns={'T_oss_std': 'std', 'T_oss_var': 'var'}).reset_index()
    temp_oss_sv = temp_oss_sv.merge(finale_oss[['dayofyear','codice_oss','yhat','gradi_giorno_T_oss','gradi_giorno_yhat','n_osservazioni']],on=['dayofyear','codice_oss'],how='left')

    temp_oss_sv.drop_duplicates(inplace=True)

    print("Calcolo percentili per curve di minimo, massimo e media.")
    temp_oss_sv = calcolo_percentili_oss_yhat(temp_oss_sv)
    temp_oss_sv = calcolo_percentili_oss_T_oss(temp_oss_sv)
    
    print("Calcolo intervalli di confidenza per curve di minimo, massimo e media.")
    temp_oss_sv_T_oss = intervalli_confidenza(temp_oss_sv,'T_oss')
    temp_oss_sv_yhat = intervalli_confidenza(temp_oss_sv,'yhat')
    
    prv = pd.DataFrame()
    prv['dayofyear'] = range(1,367)
    prv['monthday'] = pd.date_range('2020-01-01','2020-12-31').astype('str').str.slice(start=5)

    temp_oss_sv_yhat = temp_oss_sv_yhat.merge(prv,on='dayofyear',how='left')
    
    temp_oss_sv_yhat['mese'] = temp_oss_sv_yhat['monthday'].str.slice(stop=2).astype('int')
    temp_oss_sv_yhat['giorno'] = temp_oss_sv_yhat['monthday'].str.slice(start=-2).astype('int')
    
    temp_oss_sv_yhat = temp_oss_sv_yhat[['dayofyear','mese','giorno','codice_oss','gradi_giorno_yhat','gradi_giorno_T_oss','T_oss','std','var','yhat','percentile_10_yhat','percentile_90_yhat','percentile_10_T_oss','percentile_90_T_oss','c_i_95_T_oss_l','c_i_95_T_oss_h','c_i_95_yhat_l','c_i_95_yhat_h']]
    
    return temp_oss_sv_yhat

def main_gg(file_input,inizio,fine,finestra_media_mobile_g,path_to_output):
    
    print('Funzione dedicata al calcolo delle temperature giornaliere e mensili.')
    print('\n')
    
    my_bucket = 'zus-prod-s3'
    
    finestra_media_mobile_g = int(finestra_media_mobile_g)
    
    inizio_dt = pd.to_datetime(inizio,format="%Y-%m-%d")
    fine_dt = pd.to_datetime(fine,format="%Y-%m-%d")
    
    file_name_out = inizio.replace('-','') + '_' + fine.replace('-','') + '_' + str(datetime.now())[0:-7].replace('-','').replace(' ','').replace(':','')
        
    df = pd.read_csv('s3://'+ my_bucket +'/'+file_input)#,encoding='utf-16')
    if 'Codice Provincia' in df.columns:
        print('Calcolo temperatura normale giornaliera con funzioni armoniche di 6^ grado')
        temp_max_norm, temp_min_norm, temp_mean_norm = calcolo_temp_norm_gg('s3://'+ my_bucket +'/'+file_input,inizio_dt,fine_dt,finestra_media_mobile_g)
#         temp_max_norm.drop('dayofyear',inplace=True,axis=1)
#         temp_min_norm.drop('dayofyear',inplace=True,axis=1)
#         temp_mean_norm.drop('dayofyear',inplace=True,axis=1)
        
        temp_max_norm.to_csv('s3://'+ my_bucket +'/'+path_to_output+'max_d_prov/'+file_name_out+'/max_d_prov.csv',index=False)
        idrun_max = 's3://'+ my_bucket +'/'+path_to_output+'max_d_prov/'+file_name_out+'/max_d_prov.csv'
        metadatati_max = pd.DataFrame(data={'MODELLO':['MAX_D_PROV']*4,'ID_RUN':[idrun_max]*4,'NOME_PARAMETRO':['INIZIO','FINE','FINESTRA_MEDIA_MOBILE_G','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[inizio,fine,finestra_media_mobile_g,path_to_output]})
        metadatati_max.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/max_d_prov/'+file_name_out+'/metadati.csv',index=False)
        
        temp_min_norm.to_csv('s3://'+ my_bucket +'/'+path_to_output+'min_d_prov/'+file_name_out+'/min_d_prov.csv',index=False)
        idrun_min = 's3://'+ my_bucket +'/'+path_to_output+'min_d_prov/'+file_name_out+'/min_d_prov.csv'
        metadatati_min = pd.DataFrame(data={'MODELLO':['MIN_D_PROV']*4,'ID_RUN':[idrun_min]*4,'NOME_PARAMETRO':['INIZIO','FINE','FINESTRA_MEDIA_MOBILE_G','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[inizio,fine,finestra_media_mobile_g,path_to_output]})
        metadatati_min.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/min_d_prov/'+file_name_out+'/metadati.csv',index=False)
        
        temp_mean_norm.to_csv('s3://'+ my_bucket +'/'+path_to_output+'mean_d_prov/'+file_name_out+'/mean_d_prov.csv',index=False)
        idrun_mean = 's3://'+ my_bucket +'/'+path_to_output+'mean_d_prov/'+file_name_out+'/mean_d_prov.csv'
        metadatati_mean = pd.DataFrame(data={'MODELLO':['MEAN_D_PROV']*4,'ID_RUN':[idrun_mean]*4,'NOME_PARAMETRO':['INIZIO','FINE','FINESTRA_MEDIA_MOBILE_G','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[inizio,fine,finestra_media_mobile_g,path_to_output]})
        metadatati_mean.to_csv('s3://'+ my_bucket +'/metadati/sistema/temperatura_norm/zeus/metadati/mean_d_prov/'+file_name_out+'/metadati.csv',index=False)
#         temp_max_norm.to_csv('prova_gg_max.csv',index=False)
#         temp_min_norm.to_csv('prova_gg_min.csv',index=False)
#         temp_mean_norm.to_csv('prova_gg_mean.csv',index=False)
        print('\n')
    else:
        print('Calcolo osservatorio giornaliero con funzioni armoniche di 6^ grado')
        temp_oss_sv = calcolo_oss_gg('s3://'+ my_bucket +'/'+file_input,inizio_dt,fine_dt,finestra_media_mobile_g)
        #temp_oss_sv.drop(columns='dayofyear',inplace=True,axis=1)
        
        temp_oss_sv.to_csv('s3://'+ my_bucket +'/'+path_to_output+'mean_d_oss/'+file_name_out+'/mean_d_oss.csv',index=False)
        idrun = 's3://'+ my_bucket +'/'+path_to_output+'mean_d_oss/'+file_name_out+'/mean_d_oss.csv'
        metadati = pd.DataFrame(data={'MODELLO':['MEAN_D_OSS']*4,'ID_RUN':[idrun]*4,'NOME_PARAMETRO':['INIZIO','FINE','FINESTRA_MEDIA_MOBILE_G','PATH_TO_OUTPUT'],'VALORE_PARAMETRO':[inizio,fine,finestra_media_mobile_g,path_to_output]})
        metadati.to_csv('s3://'+ my_bucket +'/metadati/sistema/mean_d_v/zeus/temperatura_norm/zeus/metadati/mean_d_oss/'+file_name_out+'/metadati.csv',index=False)
#         temp_oss_sv.to_csv('prova_gg_oss.csv',index=False)
        print('\n')
    print("Procedura terminata.")
    
#     return temp_max_norm, temp_min_norm, temp_mean_norm, temp_oss_sv
