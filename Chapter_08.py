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
import numexpr as ne
from math import *


pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.float_format = '{:,}'.format
pd.options.display.max_rows = 10


# def perf_comp_data(func_list, data_list, rep=3, number=1):
#     """ Function to compare the performance of different functions
#     
#     func_list: list - list with functions names as strings
#     data_list: list - list with data set names as strings
#     rep      : int  - number of repetitions of the whole comparison
#     number   : int  - number of executions for every function
#     """
#     
#     res_list = {}
#     for name in enumerate(func_list):
#         stmt = name[1] + '(' + data_list[name[0]] + ')'
#         
#         setup = 'from __main__ import ' +  name[1] + ',' + data_list[name[0]]
#         
#         results = repeat(stmt=stmt,
#                          setup=setup,
#                          repeat=rep,
#                          number=number)
#         
#         res_list[name[1]] = sum(results) / rep
#         
#     res_sort = sorted(res_list.iteritems(),
#                       key = lambda (k,v): (k,v))
#     
#     for item in res_sort:
#         rel = item[1] / res_sort[0][1]
#         print 

# 
# def complex_func(x):
#     return (np.sqrt(abs(np.cos(x)))
#             + np.sin(2 + 3*x))
# I = 500000
# 
# def f5(a, num_thr=1):
#     
#     ex = 'abs(cos(a))**0.5 + sin(2 + 3*a)'
#     ne.set_num_threads(num_thr)
#     return ne.evaluate(ex)
# 
# 
# def bsm_mcs_valuation(strike):
#     ''' Dynamic Black-Scholes-Merton Monte Carlo estimator
#     for European calls.
#     
#     Parameters
#     ==========
#     strike : float
#         strike price of the option
#     
#     Results
#     =======
#     value : float
#         estimate for present value of call option
#     '''
#     import numpy as np
#     S0 = 100.; T = 1.0; r = 0.05; vola = 0.2
#     M = 50; I = 20000
#     dt = T / M
#     rand = np.random.standard_normal((M + 1, I))
#     S = np.zeros((M + 1, I)); S[0] = S0
#     for t in range(1, M + 1):
#         S[t] = S[t-1] * np.exp((r - 0.5 * vola ** 2) * dt
#                                + vola * np.sqrt(dt) * rand[t])
#     value = (np.exp(-r * T)
#                      * np.sum(np.maximum(S[-1] - strike, 0)) / I)
#     return value
# 
# 
# def seq_value(n):
#     ''' Sequential option valuation.
#     
#     Parameters
#     ==========
#     n : int
#         number of option valuations/strikes
#     '''
#     strikes = np.linspace(80, 120, n)
#     option_values = []
#     for strike in strikes:
#         option_values.append(bsm_mcs_valuation(strike))
#     return strikes, option_values
# 
# n = 100 #number of options to be valued
# strikes, option_values_seq = seq_value(n)
# print(option_values_seq)
#   

#***Binomial option pricing with CRR model***
#model & options parameters
S0 = 100
T = 1
r = 0.05
vola = 0.20

#time parameters
M = 1000 #time steps
dt = T/M #length of time intervals
df = np.exp(-r * dt)

#binomial parameters
u = np.exp(vola * np.sqrt(dt))
d = 1/u
q = (np.exp(r * dt) - d) / (u - d) #martingale probability



 














































































































