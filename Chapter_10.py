from __future__ import division
import numpy as np
import numpy.random as npr
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
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
import pandas.tseries
import numexpr as ne
from math import *

pd.set_option('display.width', 500)
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.float_format = '{:,}'.format
pd.options.display.max_rows = 10
np.set_printoptions(linewidth=500)


def gen_sn(M, I, anti_paths=True, mo_match=True):
    
    if anti_paths is True:
        sn = npr.standard_normal((M + 1, int(I / 2)))
        sn = np.concatenate((sn, -sn), axis=1)
        
    else:
        sn = npr.standard_normal((M + 1, I))
    
    if mo_match is True:
        sn = (sn - sn.mean()) / sn.std()
    
    return sn


# Risk Neutral Monte Carlo estimator
# European Call on Index
S0 = 100.0
r = 0.05
sigma = 0.25
T = 1.0
I = 50000


def gbm_mcs_stat(K):
    # K: float (positive) strike price of the option
    # C0: float estimated present value of the European Call
    sn = gen_sn(1, I)
    # Simulate Index Level at Maturity
    ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T
                     +sigma * np.sqrt(T) * sn[1])
    
    # Calculate payoff at maturity
    hT = np.maximum(ST - K, 0)
    
    # Calculate MCS estimator
    C0 = np.exp(-r * T) * 1 / I * np.sum(hT)
    return C0

# print(gbm_mcs_stat(K=105))


#===============================================================================
# Dynamic Simulation : European Call and Put options
def gbm_mcs_dyna(K, option='call', M=50):
    
    dt = T / M  # in a Dynamic Simulation there is change of time
    
    # Simulation of Index Levels
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    for t in range(1, M + 1, 1):
        S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt
                               +sigma * np.sqrt(dt) * sn[t])
    
    # case-based calculation of payoff
    if option == 'call':
        hT = np.maximum(S[-1] - K, 0)
    else:
        hT = np.maximum(K - S[-1], 0)
    
    # Calculation of MCS simulator
    C0 = np.exp(-r * T) * 1 / I * np.sum(hT)
    return C0
#===============================================================================


print(gbm_mcs_dyna(K=110., option='call'), '\n\n')
print(gbm_mcs_dyna(K=110., option='put'), '\n\n')

