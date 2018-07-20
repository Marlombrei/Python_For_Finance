from __future__ import division, print_function
import pandas.tseries
import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.stats as scs
import scipy.optimize as sco
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime, date, time
from dateutil.parser import parse
import csv
import json
import openpyxl
from openpyxl import load_workbook
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas.core.tools.datetimes as datetools
import statsmodels.api as sm
#import statsmodels.formula.api as smf
import numexpr as ne
from math import *
from sklearn.decomposition import KernelPCA



pd.set_option('display.width', 500)
pd.options.display.float_format = '{:,.4f}'.format
pd.options.display.max_rows = 10
np.set_printoptions(linewidth=200)

npr.seed(1000)

def gen_paths(S0, r, sigma, T, M, I):
        """Comments here"""

        dt = T / M # Dynamic mode
        paths = np.zeros((M+1, I), np.float64)
        paths[0] = S0
        for t in range(1,M+1,1):
            #Generating proper Normally Distributed Samples
            rand = np.random.randn(I)
            rand = (rand - rand.mean()) / rand.std()
            #Generating the paths
            paths[t] = paths[t-1] * np.exp((r - 0.5*sigma**2)*dt
                                           + sigma * np.sqrt(dt) * rand)
        return paths
#
#
# S0 = 100
# r = 0.05
# sigma = 0.20
# T = 1
# M = 50
# I = 250000
#
# paths = gen_paths(S0, r, sigma, T, M, I)

# Distribution of Log Returns
# log_returns = np.log(paths[1:] / paths[:-1])
# print(paths[:,0].round(4),'\n\n')
# print(log_returns[:,0].round(4))
#
def print_statistics(array):
    """Comments here"""

    sta = scs.describe(array)
    print(('%14s %15s') % ('Statistic', 'Value'))
    print(30*'-')
    print(('%14s %15.5f') % ('Size', sta[0]))
    print(('%14s %15.5f') % ('Min', sta[1][0]))
    print(('%14s %15.5f') % ('Max', sta[1][1]))
    print(('%14s %15.5f') % ('Mean', sta[2]))
    print(('%14s %15.5f') % ('Std', np.sqrt(sta[3])))
    print(('%14s %15.5f') % ('Skew', sta[4]))
    print(('%14s %15.5f') % ('Kurtosis', sta[5]))
#
# flat = log_returns.flatten()
# print_statistics(flat)

# plt.hist(log_returns.flatten(),
#          bins=50,
#          normed=True,
#          label='Frequency')
# x = np.linspace(plt.axis()[0], plt.axis()[1])
# plt.plot(x, scs.norm.pdf(x, loc=r / M,
#                          scale= sigma/np.sqrt(M)),
#                          'r', lw=2.0, label='pdf')
#
# sm.qqplot(log_returns.flatten()[::500], line='s')
# plt.show()

def normality_tests(arr):
    ''' Tests for normality distribution of given data set.

    Parameters
    ==========
    array: ndarray
        object to generate statistics on
    '''
    print("Skew of data set  %14.3f" % scs.skew(arr))
    print("Skew test p-value %14.3f" % scs.skewtest(arr)[1])
    print("Kurt of data set  %14.3f" % scs.kurtosis(arr))
    print("Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1])
    print("Norm test p-value %14.3f" % scs.normaltest(arr)[1])
#
#
# normality_tests(log_returns.flatten())

#=======================================================================================================================================================================================================
# Real World Data: DAX, S&P500, Yahoo, Microsoft
file = r'C:\Users\Marlombrei\Documents\Python_For_Finance\tr_eikon_eod_data.csv'
# C:\Users\Marlombrei\Documents\Python_For_Finance\tr_eikon_eod_data.csv
# C:\Users\Marlom Silva\Downloads\Python\PYTHON for Finance\Data\tr_eikon_eod_data.csv
raw = pd.read_csv(file,
                  index_col=0,
                  parse_dates=True)
# symbols = ['SPY', 'GLD', 'AAPL.O', 'MSFT.O']
# data = raw[symbols]
# data = data.dropna()
# print(data.head(),'\n\n')
#
# log_returns = np.log(data / data.shift(1))
# print(log_returns.head())

# log_returns.hist(bins=50)
# plt.show()

# for sym in symbols:
#     print('\n Results for symbol {}'.format(sym))
#     print(30*'-')
#     log_data = np.array(log_returns[sym].dropna())
#     print_statistics(log_data)
#=======================================================================================================================================================================================================

# sm.qqplot(log_returns['SPY'].dropna(), line='s')
# plt.show()



# for sym in symbols:
#     print('\n Results for symbol {}'.format(sym))
#     print(200*'#')
#     log_data = np.array(log_returns[sym].dropna())
#     normality_tests(log_data)


#=======================================================================================================================================================================================================
# MPT
symbols = ['AAPL.O', 'MSFT.O', 'AMZN.O', 'GDX', 'GLD']
noa = len(symbols)

data = raw[symbols]
# print(data.head(),'\n')

#log returns
rets = np.log(data / data.shift(1))
# print(rets.mean() * 252,'\n')
# print(rets.cov() * 252,'\n')
#=======================================================================================================================================================================================================

def normal_weights(I):
    weights = npr.random(I)
    weights /= np.sum(weights)
    return weights

weights = normal_weights(noa)
# print([weights],'\n')

#===============================================================================
# Portfolio Expected Return
expec_ret = np.sum(rets.mean() * weights) * 252
# print(expec_ret,'\n')
# print(np.sum(weights.T * rets.mean()) * 252,'\n')
#===============================================================================
#===============================================================================
# Expected Portfolio Variance
expec_vari = np.dot(weights.T,
                    np.dot(rets.cov() * 252,
                           weights))
# print(expec_vari,'\n')

# Expected Portfolio Std Dev / Volatility
expec_std_dev = np.sqrt(expec_vari)
# print(expec_std_dev,'\n')

#===============================================================================

#===============================================================================
# Monte Carlo Simulation for MPT
prets = [] # portfolio returns
pvols = [] # portfolio variance
for p in range(2500):
    weights = normal_weights(noa)
    prets.append(np.sum(rets.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T,
                                np.dot(rets.cov() * 252, weights))))

prets = np.array(prets)
pvols = np.array(pvols)

# plt.scatter(pvols, prets, c= prets / pvols, marker='o')
# plt.colorbar()
# plt.show()

#===============================================================================

def statistics(weights):
    """Portfolio Statistics"""

    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T,
                           np.dot(rets.cov()*252, weights)))
    return np.array([pret, pvol, pret/pvol])


def min_func_sharpe(weights):
    return -statistics(weights)[2]

#Constrain is: All weights add up to 1
cons = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})

#Parameters values (weights) to be within 0 and 1.
bnds = tuple((0,1) for x in range(noa))

opts = sco.minimize(fun=min_func_sharpe,
                    x0=noa * [1/noa],
                    method='SLSQP',
                    bounds=bnds,
                    constraints=cons)

# print(opts,'\n\n')
# print(opts['x'].round(4),'\n\n')
#
# print(statistics(opts['x'].round(4)))

def min_func_variance(weights):
    return statistics(weights)[1]**2

optv = sco.minimize(fun=min_func_variance,
                    x0=noa * [1/noa],
                    method='SLSQP',
                    bounds=bnds,
                    constraints=cons)

# print(optv,'\n')
# print(optv['x'].round(4),'\n')
# print(statistics(optv['x'].round(4)))

#EFFICIENT FRONTIER

cons = ({'type':'eq', 'fun':lambda x: statistics(x)[0] - tret}, # tret: Target Return
        {'type':'eq', 'fun':lambda x: np.sum(x) - 1})

bnds = tuple((0,1) for x in weights)

def min_func_port(weights):
    return statistics(weights)[1]

trets = np.linspace(0.0, 0.25,50)
tvols = []
for tret in trets:
    cons = ({'type':'eq', 'fun':lambda x: statistics(x)[0] - tret}, # tret: Target Return
            {'type':'eq', 'fun':lambda x: np.sum(x) - 1})

    res = sco.minimize(fun=min_func_port,
                       x0=noa*[1/noa],
                       method='SLSQP',
                       bounds=bnds,
                       constraints=cons)
    tvols.append(res['fun'])

tvols = np.array(tvols)

# plt.scatter(pvols, prets, c=prets/pvols, marker='o')
# plt.scatter(tvols, trets, c=trets/tvols, marker='x')
# plt.plot(statistics(opts['x'])[1],
#          statistics(opts['x'])[0], 'r*', markersize=15)
#
# plt.plot(statistics(optv['x'])[1],
#          statistics(optv['x'])[0], 'y*', markersize=15)
#
# plt.colorbar(label='Sharpe Ratio')
# plt.show()


#=======================================================================================================================================================================================================
# Applying PCA


symbols = ['ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE',
           'BMW.DE', 'CBK.DE', 'CON.DE', 'DAI.DE', 'DB1.DE',
           'DBK.DE', 'DPW.DE', 'DTE.DE', 'EOAN.DE', 'FME.DE',
           'FRE.DE', 'HEI.DE', 'HEN3.DE', 'IFX.DE', 'LHA.DE',
           'LIN.DE', 'LXS.DE', 'MRK.DE', 'MUV2.DE', 'RWE.DE',
           'SAP.DE', 'SDF.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE',
           '^GDAXI']


# data = pd.DataFrame()
# for sym in symbols:
#     # only retrieves data from Jan 2016 on
#     data[sym] = pdr.DataReader(sym,
#                                data_source='yahoo',
#                                start='2016-1-1')['Adj Close']
# data = data.dropna()
# print('')

file_path = r'C:\Users\Marlombrei\Documents\Python_For_Finance\DAX_stocks_price.csv'
# C:\Users\Marlom Silva\Downloads\Python\PYTHON for Finance\Data\DAX_stocks_price.csv
# C:\Users\Marlombrei\Documents\Python_For_Finance\DAX_stocks_price.csv
data = pd.read_csv(file_path,
                   parse_dates=['Date'],
                   index_col=0)
print(data.head())

dax = pd.DataFrame(data.pop('^GDAXI'))
print(dax.head())

print(data.head())

# Apply PCA
# PCA works with Normalized Data Sets
scale_function = lambda x: (x - x.mean()) / x.std()


pca = KernelPCA().fit(data.apply(scale_function))
print(pca.lambdas_[:10].round()) # Eigenvalues: the function give too many eigenvalues but here we are only showing the first 10 components


#Normalized Values: relative importance
get_we = lambda x: x / x.sum()
print(get_we(pca.lambdas_[:10]))
print(get_we(pca.lambdas_[:5]).sum())

# PCA Index with one component
# pca = KernelPCA(n_components=1).fit(data.apply(scale_function))
# dax['PCA_1'] = pca.transform(-data)
# dax.apply(scale_function).plot()
# plt.show()


# PCA Index with five components
pca = KernelPCA(n_components=5).fit(data.apply(scale_function))
pca_components = pca.transform(data)
weights = get_we(pca.lambdas_)
dax['PCA_5'] = np.dot(pca_components, weights)
# dax.apply(scale_function).plot()
# plt.show()

# Scatter Plot of the relationship between Dax and PCA Index

# Convert DataTimeIndex to a matplotlib compatible format
mpl_dates = mpl.dates.date2num(data.index.to_pydatetime())
#mpl_dates = data.index.to_string()
print(mpl_dates[:10])

plt.scatter(dax['PCA_5'], dax['^GDAXI'], c=mpl_dates)
lin_reg = np.polyval(np.polyfit(dax['PCA_5'],
                                dax['^GDAXI'], 1),
                                dax['PCA_5'])
plt.plot(dax['PCA_5'], lin_reg, 'r', lw=3)
plt.grid(True)
plt.xlabel('PCA_5')
plt.ylabel('^GDAXI')
plt.colorbar(ticks=mpl.dates.DayLocator(interval=250),
                format=mpl.dates.DateFormatter('%d %b %y'))
plt.show()


#=======================================================================================================================================================================================================






















































































































