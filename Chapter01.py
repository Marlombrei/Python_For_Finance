from __future__ import division
import pandas as pd
import pandas_datareader as web
import datetime as dt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def MCoptionvaluation(S0, K, r, sigma, T=1.0, I=1000):
    '''Monte Carlo Valuation of European call Option'''
    from numpy import random
    z = random.standard_normal(I)
    ST = (S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z))
    hT = np.maximum(ST - K, 0)
    C0 = np.exp(-r * T) * np.sum(hT) / I
    print ('Value of the European Call %.3f' % C0)

    
def Volatility(StockName, start, end):
    '''Function adds the log Return and the Volatility of the returns
    It uses the new version of pandas.io.data (pandas_datareader)
    Also the pd.rolling_std has now been replaced by pd.Series.rolling(...,windown=252, center=False).std'''
    #
    #
    import pandas_datareader as web
    stock = web.DataReader(StockName,
                           data_source='yahoo',
                           start=start, end=end)
    stock['Log_Return'] = np.log(stock['Close'] / stock['Close'].shift(1))
    
    stock['Volatility'] = pd.Series.rolling(stock['Log_Return'],
                                            window=252,
                                            center=False).std() * np.sqrt(252)
    #stock[['Close', 'Volatility']].plot(subplots=True)
    #plt.show()
    print (stock.tail())


################################################################
start = dt.datetime(2014, 4, 14)
end = dt.datetime(2018, 6, 21)
goog = web.DataReader('WMT',
                      data_source='yahoo',
                      start=start,
                      end=end)

goog['Log_Ret'] = np.log(goog['Close'] / goog['Close'].shift(1))

goog['Volatility'] = pd.Series.rolling(goog['Log_Ret'],
                                       window=5,
                                       center=False).std() * np.sqrt(52)
 
#goog[['Close', 'Volatility']].plot(subplots=True)
#plt.show()

MCoptionvaluation(S0=100, K=105, r=0.05, sigma=0.2, T=1.0, I=1000)

# import numexpr as ne
# import timeit
# ne.set_num_threads(2)
# #a = np.arange(1,25000000)
# f = '3 * log(a) + cos(a) ** 2'
# print (timeit.timeit(ne.evaluate(f)))#, setup, timer, number, globals))

print(goog)













































































