from __future__ import division
import numpy as np
import scipy as sp
import scipy.stats as st
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, date, time
from dateutil.parser import parse
import csv
import json
import openpyxl
from openpyxl import load_workbook
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
#import statsmodels.api as sm
#import statsmodels.formula.api as smf
import pandas.tseries

pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.float_format = '{:,}'.format
pd.options.display.max_rows = 10

# DAX = web.DataReader(name='^GDAXI',
#                      data_source='yahoo',
#                      start='2000-1-1').dropna()
# 
# DAX['Return'] = np.log(DAX['Adj Close'] / DAX['Adj Close'].shift(1))
# 
# DAX['42d'] = DAX['Adj Close'].rolling(window=42).mean()
# DAX['252d'] = DAX['Adj Close'].rolling(window=252).mean()
# DAX['Mov_Vol'] = DAX['Return'].rolling(window=252).std() * np.sqrt(252)
# 
# print DAX.info()
# print DAX,'\n\n'
# 
# DAX[['Adj Close','42d','252d']].plot()
# plt.show()

#es_url = 'http://www.stoxx.com/download/historical_values/hbrbcpe.txt'
#vs_url = 'http://www.stoxx.com/download/historical_values/h_vstoxx.txt'
# es_file = r'C:\Users\Marlombrei\Documents\Python_For_Finance\es.txt'
# vs_file = r'C:\Users\Marlombrei\Documents\Python_For_Finance\vs.txt'
# #urlretrieve(es_url,es_file)
# #urlretrieve(vs_url,vs_file)
# 
# lines = open(es_file,'r').readlines()
# lines = [line.replace(' ','') for line in lines]
# 
# #cleaning the data
# es_file2 = r'C:\Users\Marlombrei\Documents\Python_For_Finance\es50.txt'
# new_es_file = open(es_file2,'w')
# new_es_file.writelines('date' + lines[3][:-1]
#                        +';DEL' + lines[3][-1])
# new_es_file.writelines(lines[4:])
# new_es_file.close()
# 
# new_lines = open(es_file2,'r').readlines()
# for i in new_lines[:5]:
#     print(i)
# 
# pd.read_csv
# es = pd.read_csv(es_file2,
#                  index_col=0,
#                  parse_dates=True,
#                  sep=';',
#                  dayfirst=True)
# 
# del es['DEL']
# print(es.tail())
# 
# vs = pd.read_csv(vs_file,
#                  skiprows=[0,1],
#                  index_col=0,
#                  parse_dates=True,
#                  dayfirst=True)
# 
# print(vs)
# 
# 
# mask_es = (es.index > datetime(1999,1,1)) & (es.index <= datetime(2015,12,1))
# data = pd.DataFrame({'EUROSTOXX':
#                      es['SX5E'][mask_es]})
# 
# mask_vs = (vs.index > datetime(1999,1,1)) & (vs.index <= datetime(2015,12,1))
# data = data.join(pd.DataFrame({'VSTOXX':
#                                vs['V2TX'][mask_vs]}))
# data.fillna(method='ffill', inplace=True)
# 
# print(data.info())
# 
# # data.plot(subplots=True,
# #           grid=True)
# # 
# # plt.show()
# 
# rets = np.log(data / data.shift(1))
# print(rets.round(4))
# 
# xdat = rets['EUROSTOXX'].dropna() #independent variable
# ydat = rets['VSTOXX'].dropna()    #Dependent variable
# 
# model = sm.OLS(ydat, xdat)
# res = model.fit()
# print(res.summary())
# 
# print(rets.corr())
# 
# cor = (rets['EUROSTOXX'].rolling(252,min_periods=50)
#                         .corr(rets['VSTOXX']))
# plt.plot(cor)
# plt.show()

file = r'C:\Users\Marlombrei\Documents\Python_For_Finance\fxcm_eur_usd_tick_data.csv'
eur_usd = pd.read_csv(file,
                      index_col=0,
                      parse_dates=True)

print(eur_usd)
print(eur_usd.info())

eur_usd['Bid'].plot()
plt.show()

eur_usd_resem = eur_usd.resample(rule='5min',
                                 how='mean')

eur_usd_resem.plot()
plt.show()










































































