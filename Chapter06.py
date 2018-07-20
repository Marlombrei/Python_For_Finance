from __future__ import division
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
import datetime as dt
# from datetime import datetime
# from datetime import timedelta
from dateutil.parser import parse
import pytz
import json
from collections import defaultdict, Counter
import timeit
from pandas_datareader import data, wb



pd.options.display.width = 500
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:,.4f}'.format

# df = pd.DataFrame([10,20,30,40],
#                   columns=['numbers'],
#                   index=list('abcd'))
# 
# 
# 
# df['floats']= (1.5,2.5,3.5,4.5)
# 
# df['names'] = pd.DataFrame('Yves Guido Felix Francesc'.split(),
#                            index=list('dabc'))
# 
# df = df.append(pd.DataFrame({'numbers':100,
#                              'floats':5.75,
#                              'names':'Henry'},
#                              index=['z',]))
# 
# df = df.join(other=pd.DataFrame([1,4,9,16,25],
#                            index=list('abcdy'),
#                            columns=['squares']),
#                            how='outer')


# a = np.random.standard_normal((9,4))
# #print(a.round(6))
# 
# df = pd.DataFrame(a)
# df.columns = [['No1','No2','No3','No4']]
# df.index = pd.date_range(start='2017-Sep', periods=9, freq='M')
# 
# df['Quarter'] = 'Q1 Q1 Q1 Q2 Q2 Q2 Q3 Q3 Q3'.split()
# df['Odd_even'] = 'Odd Even Odd Even Odd Even Odd Even Odd'.split()
# print(df,'\n')
# 
# groups = df.groupby(by=['Quarter','Odd_even'])#, axis, level, as_index, sort, group_keys, squeeze)
# print(groups.mean())
# #print(np.array(df).round(4))

DAX = data.DataReader(name='^GDAXI',
                      data_source='yahoo',
                      start='2000-1-1').dropna()

DAX['Log Return'] = (np.log(DAX['Adj Close']/DAX['Adj Close'].shift(1))).round(6)
DAX['42d'] = DAX['Adj Close'].rolling(window=42).mean()
DAX['252d'] = DAX['Adj Close'].rolling(window=252).mean()
DAX['Mov_Vol'] = DAX['Log Return'].rolling(window=252).std() * np.sqrt(252)
print(DAX.info())
print(DAX.tail())


DAX[['Adj Close','Mov_Vol','Log Return']].plot(subplots=True)
plt.show()
















































