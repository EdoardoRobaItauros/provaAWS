# import delle temperature storiche per provincia da 2010 a 2020

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm, gumbel_r, gumbel_l
from scipy import stats
import math
from statsmodels.distributions.empirical_distribution import ECDF
import boto3

def isdir_s3(bucket, key):
    objs = list(bucket.objects.filter(Prefix=key))
#     print('I am the master_dir')
    return len(objs)

def costruzione_file_da_mensile(source, inizio, fine):
    bucketname = 'zus-prod-s3'
    s3 = boto3.resource('s3')
#     source = 'preprocessato/sistema/temperatura/epson/temperatura'
    my_bucket = s3.Bucket(bucketname)
    annomesi = pd.date_range(inizio, fine,freq='MS').strftime('%Y%m').tolist()
    df = pd.DataFrame()
    for am in annomesi:
        if isdir_s3(my_bucket,source + '/' +str(am))>0:
            df = df.append(pd.read_csv('s3://'+bucketname+'/'+source+'/'+str(am)+'/epson_best.csv'))
    return df

# def importa(path_temperature_storiche):
def importa(df,tipo):
    
    #df = pd.read_csv(path_temperature_storiche, delimiter = ',', decimal = '.', parse_dates = ['TIME - Date'], dayfirst = False)
#     print(df)
    df.rename(columns={'DATA':'date',tipo.upper():tipo.lower(),'TEMPERATURA_MIN':'temp_min','TEMPERATURA_MAX':'temp_max'},inplace=True)
#     df.rename(columns={'TIME - Date': 'date','Codice Provincia':'provincia','Temperature Min':'temp_min','Temperature Max':'temp_max'},inplace=True)
    df = df[['date', tipo, 'temp_max', 'temp_min']]
#     print(df)
    
    df['date'] = pd.to_datetime(df['date'],format="%Y-%m-%d")
    
    df = df.assign(mese = lambda x: x.date.dt.month, giorno = lambda x: x.date.dt.day, anno = lambda x: x.date.dt.year, giorno_anno = lambda x: x.date.dt.dayofyear)

    df = df.assign(temp = lambda x: (x['temp_max'] + x['temp_min'])/2)
                   
    df['temp_norm'] = df[['mese', 'giorno', tipo, 'temp']].groupby(['mese', 'giorno', tipo]).transform(lambda x: x.mean())
    
    df = df.drop(df[((df['mese'] == 2) & (df['giorno'] == 29))].index)
    
    df['giorno_anno'] = df.apply(lambda x: x['giorno_anno'] - 1 if((x['giorno_anno'] > 59) & (x['anno'] % 4 == 0)) else x['giorno_anno'], axis = 1)
    
    return df

# funzione che calcola il percentile nei dati passati in input come df e percentile parametro compreso tra 0 e 1

def compute_perc(dati, percentile):
    
    dati = dati[(~dati.isnull())]
    
    return dati.sort_values().iloc[max(0, math.ceil((len(dati) - 1)*percentile))]

# funzione per calcolare la coda estrema oltre il percentile passato in input come sopra

def compute_tail(dati, percentile, tail = None):
    
    dati = dati[(~dati.isnull())]
    
    #tail = 1 coda di sinistra
    
    if (tail == 1):
            
            return dati.sort_values().iloc[:max(0, math.ceil((len(dati) - 1)*percentile))]
    
    return dati.sort_values().iloc[max(0, math.ceil((len(dati) - 1)*percentile)):]

# funzione per calcolare gli estremi di finestra per campionare i delta su più giorni
# la finestra [a, b] è centrata in lag = 0
# se ampiezza dispari, allora -a = b
# se ampiezza pari, allora -a = b+1
# se ampiezza = 0, allora a=b=0

def compute_index(window):
    
    b = (window - 1)/2
    
    if (window % 2 == 0):
    
        a = - window/2
    
    else:
    
        a = - b
    
    return int(a), int(b)

# funzione per calcolare percentile di distribuzione di gumbel right

def perc_gumbel_right(dati, percentile, q):
    
    dati = dati[(~dati.isnull())]
    
    subset = compute_tail(dati, percentile)
    
    loc, scale = gumbel_r.fit(subset)
    
    return gumbel_r(loc, scale).ppf(q)

def perc_gumbel_left(dati, percentile, q):
    
    dati = dati[(~dati.isnull())]
    
    subset = compute_tail(dati, percentile, 1)
    
    loc, scale = gumbel_l.fit(subset)
    
    return gumbel_l(loc, scale).ppf(q)

def perc_empiric(dati, percentile, q, tail = None):
    
    dati = dati[(~dati.isnull())]
    
    subset = compute_tail(dati, percentile, tail = tail)
    
    return subset.sort_values().iloc[math.ceil((len(subset)-1)*q)]

def perc_normal(dati, percentile):
    
    dati = dati[(~dati.isnull())]
    
    m, sigma = norm.fit(dati)
    
    return norm.ppf(percentile, loc = m, scale = sigma)


def compute_code(df, col_media_oss,col_media_normale,tipo, valori_date, window_ma, percentile, window, q):
    
    #col_media_oss : colonna da usare come temp ==(T_max+T_min/2)
    #col_media_normale : colonna con media normale== temp_norm calcolata raggruppata su(giorno,mese,provincia)['temp'].mean()
    #tipo : provincia o osservatorio
    #valori_date : lista di colonne che caratterizzano le date presenti nel df -> giorno,mese,anno,dayofyear
    
    # window_ma : finestra giornaliera centrata per media mobile della temperatura effettiva
    # window : finestra giornaliera centrata per collezionare i delta: media_mobile - t.norm
    # percentile : percentile di inizio coda sulla distribuzione empirica dei dati
    # q : percentile estremo che si vuole individuare sulla coda fittata
    
    #df2 = df[(df['provincia'] == prov)][['anno','giorno_anno','temp_norm', 'temp']].copy()
    
    df.rename(columns={col_media_oss:'temp',col_media_normale:'temp_norm'},inplace=True)
    #print(df.info())
    if tipo=='None':
        df2 = df[valori_date +['temp_norm', 'temp']]
        df2 = df2.apply(pd.DataFrame.sort_values, valori_date)

        df2.reset_index(inplace=True)

        df2['media mobile'] = df2['temp'].transform(lambda x: x.rolling(window = window_ma, center = True, min_periods = 1).mean())

        df2 = df2.assign(delta = lambda x: x['media mobile'] - x['temp_norm'])

        df2 = df2[valori_date + ['delta']].copy()

        df3 = df2.copy()

        df3 = df3.assign(lag = lambda x: 0)
    #     ###################################### da togliere
    #     df2 = df2.loc[df2['codice_oss']==11]
    #     ######################################
        if(window != 0):

            a, b = compute_index(window)

            lags = np.arange(a, b + 1)

            lags = lags[lags != 0]

            for l in lags:

                df4 = roll(df2.copy(),valori_date, -l,None)

                df4 = df4.assign(lag = lambda x: l)

                df3 = pd.concat([df3, df4], axis = 0)
        #############da togliere
        df3.rename(columns={'giorno_anno':'dayofyear'},inplace=True)

        df3['tail_r'] = df3.groupby(by = ['dayofyear'])['delta'].transform(lambda x: compute_perc(x, percentile))

        df3['gumbel_r'] = df3.groupby(by = ['dayofyear'])['delta'].transform(lambda x: perc_gumbel_right(x, percentile, q))

        df3['empiric_r'] = df3.groupby(by = ['dayofyear'])['delta'].transform(lambda x: compute_perc(x, percentile + (1 - percentile)*q))

        df3['gauss_r'] = df3.groupby(by = ['dayofyear'])['delta'].transform(lambda x: perc_normal(x, percentile + (1 - percentile)*q))

        df3['tail_l'] = df3.groupby(by = ['dayofyear'])['delta'].transform(lambda x: compute_perc(x, (1 - percentile)))   

        df3['gumbel_l'] = df3.groupby(by = ['dayofyear'])['delta'].transform(lambda x: perc_gumbel_left(x, 1 - percentile, 1 - q))

        df3['empiric_l'] = df3.groupby(by = ['dayofyear'])['delta'].transform(lambda x: compute_perc(x, (1 - percentile)*(1 - q)))

        df3['gauss_l'] = df3.groupby(by = ['dayofyear'])['delta'].transform(lambda x: perc_normal(x, (1 - percentile)*(1 - q)))

        
    else:
        df2 = df[valori_date +['temp_norm', 'temp',tipo]]
        df2 = df2.groupby(tipo).apply(pd.DataFrame.sort_values, valori_date)

        df2.drop(tipo,axis=1,inplace=True)
        df2.reset_index(inplace=True)

        df2['media mobile'] = df2.groupby(tipo)['temp'].transform(lambda x: x.rolling(window = window_ma, center = True, min_periods = 1).mean())

        df2 = df2.assign(delta = lambda x: x['media mobile'] - x['temp_norm'])

        df2 = df2[valori_date + ['delta',tipo]].copy()

        df3 = df2.copy()

        df3 = df3.assign(lag = lambda x: 0)
    #     ###################################### da togliere
    #     df2 = df2.loc[df2['codice_oss']==11]
    #     ######################################
        if(window != 0):

            a, b = compute_index(window)

            lags = np.arange(a, b + 1)

            lags = lags[lags != 0]

            for l in lags:

                df4 = roll(df2.copy(),valori_date, -l,tipo)

                df4 = df4.assign(lag = lambda x: l)

                df3 = pd.concat([df3, df4], axis = 0)
        #############da togliere
        df3.rename(columns={'giorno_anno':'dayofyear'},inplace=True)

        df3['tail_r'] = df3.groupby(by = ['dayofyear',tipo])['delta'].transform(lambda x: compute_perc(x, percentile))

        df3['gumbel_r'] = df3.groupby(by = ['dayofyear',tipo])['delta'].transform(lambda x: perc_gumbel_right(x, percentile, q))

        df3['empiric_r'] = df3.groupby(by = ['dayofyear',tipo])['delta'].transform(lambda x: compute_perc(x, percentile + (1 - percentile)*q))

        df3['gauss_r'] = df3.groupby(by = ['dayofyear',tipo])['delta'].transform(lambda x: perc_normal(x, percentile + (1 - percentile)*q))

        df3['tail_l'] = df3.groupby(by = ['dayofyear',tipo])['delta'].transform(lambda x: compute_perc(x, (1 - percentile)))   

        df3['gumbel_l'] = df3.groupby(by = ['dayofyear',tipo])['delta'].transform(lambda x: perc_gumbel_left(x, 1 - percentile, 1 - q))

        df3['empiric_l'] = df3.groupby(by = ['dayofyear',tipo])['delta'].transform(lambda x: compute_perc(x, (1 - percentile)*(1 - q)))

        df3['gauss_l'] = df3.groupby(by = ['dayofyear',tipo])['delta'].transform(lambda x: perc_normal(x, (1 - percentile)*(1 - q)))
        
    df3.dropna(subset = ['delta',tipo],inplace=True)
    df = df.merge(df3,on=valori_date + [tipo],how='left')
    
    return df #df3.dropna(subset = ['delta',tipo])


# # popolamento del dataframe principale con i quantili stimati

def roll(df,valori_date, num,tipo):
    if tipo == None:
        if(num > 0):

            #temp = df.set_index(['anno','giorno_anno']).unstack().iloc[:,:(num)].values
            temp = df.set_index(valori_date).unstack().iloc[:,:(num)].values
            #print(temp)
            df = df.set_index(valori_date).unstack().shift(-num, axis = 1)
            #print(df)
            df.iloc[:, -num:] = temp

            df = df.stack().reset_index()
            #print(df)
        elif(num < 0):

            temp = df.set_index(valori_date).unstack().iloc[:, num:].values
            #print(temp)
            df = df.set_index(valori_date).unstack().shift(-num, axis = 1)
            #print(df)
            df.iloc[:,:(-num)] = temp

            df = df.stack().reset_index()
    else:
        if(num > 0):
            #temp = df.set_index(['anno','giorno_anno']).unstack().iloc[:,:(num)].values
            temp = df.set_index([tipo]+valori_date).unstack().iloc[:,:(num)].values
            #print(temp)
            df = df.set_index([tipo]+valori_date).unstack().shift(-num, axis = 1)
            #print(df)
            df.iloc[:, -num:] = temp

            df = df.stack().reset_index()
            #print(df)
        elif(num < 0):

            temp = df.set_index([tipo]+valori_date).unstack().iloc[:, num:].values
            #print(temp)
            df = df.set_index([tipo]+valori_date).unstack().shift(-num, axis = 1)
            #print(df)
            df.iloc[:,:(-num)] = temp

            df = df.stack().reset_index()
            #print(df)
    return df

######## Possibile PATH_OUTPUT: s3://zus-prod-s3/preprocessato/sistema/temperatura_norm/zeus/code_prov/best/code_prov_inizio_fine.csv

def main(source,inizio,fine,tipo,p,q_estremo,w,w_ma):
    
    df = costruzione_file_da_mensile(source, inizio, fine)
    df = df[~df[tipo.upper()].isnull()]
    df = importa(df,tipo)
    df.rename(columns={'anno':'year','giorno_anno':'dayofyear','mese':"month",'giorno':'day'},inplace=True)
    tipo = tipo.lower()

    q = (q_estremo - p)/(1 - p)
    a = compute_code(df, 'temp','temp_norm',tipo,['year','dayofyear'], window_ma = w_ma, percentile = p, window = w, q = q)
    return a
    
