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
from urllib import urlretrieve

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
es_file = r'C:\Users\Marlom Silva\Downloads\Python\PYTHON for Finance\Data\es.txt'
vs_file = r'C:\Users\Marlom Silva\Downloads\Python\PYTHON for Finance\Data\vs.txt'
#urlretrieve(es_url,es_file)
#urlretrieve(vs_url,vs_file)

lines = open(es_file,'r').readlines()
lines = [line.replace(' ','') for line in lines]

#cleaning the data
es_file2 = r'C:\Users\Marlom Silva\Downloads\Python\PYTHON for Finance\Data\es50.txt'
new_es_file = open(es_file2,'w')
new_es_file.writelines('date' + lines[3][:-1]
                       +';DEL' + lines[3][-1])
new_es_file.writelines(lines[4:])
new_es_file.close()

new_lines = open(es_file2,'r').readlines()
for i in new_lines[:5]:
    print i

pd.read_csv
es = pd.read_csv(es_file2,
                 parse_dates=True,
                 sep=';',
                 dayfirst=True)
print es

































































































