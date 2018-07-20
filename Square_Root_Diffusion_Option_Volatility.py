from __future__ import division, print_function
import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.stats as scs
import scipy.optimize as sco
from scipy.stats import ncx2
import pandas as pd
from pandas_datareader import data as pdr
from datetime import datetime, date, time
import openpyxl as oxl
from openpyxl import load_workbook
import xlrd, xlwt, xlsxwriter
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import pandas.core.tools.datetimes as datetools
# import statsmodels.api as sm
# #import statsmodels.formula.api as smf
# import numexpr as ne
# from math import *
# from sklearn.decomposition import KernelPCA

#===============================================================================
#
# Valuation of European volatility call options
# in Gruenbichler-Longstaff (1996) model
# square-root diffusion framework
# -- semianalytical formula
#
#===============================================================================

def calculate_option_value(V0, kappa, theta, sigma, zeta, T, r, K):
    ''' Calculation of European call option price in GL96 model.
    Parameters
    ==========
    V0 : float : current volatility level
    kappa : float : mean reversion factor
    theta : float : long-run mean of volatility
    sigma : float : volatility of volatility
    zeta : volatility risk premium
    T : float : time-to-maturity
    r : float : risk-free short rate
    K : float : strike price of the option
    Returns
    =======
    value : float : net present value of volatility call option
    '''

    # Discount Factor
    D = np.exp(-r * T)
    
    # Variables
    alpha = kappa * theta
    beta = kappa * zeta
    gamma = 4*beta / ((1-np.exp(-beta * T))*sigma**2)
    nu = 4*alpha / sigma**2
    lamb = gamma * V0 * np.exp(-beta * T)
    
    cx1 = 1 - ncx2.cdf(gamma * K,
                       nu + 4,
                       lamb)
    cx2 = 1 - ncx2.cdf(gamma * K,
                       nu + 2,
                       lamb)
    cx3 = 1 - ncx2.cdf(gamma * K,
                       nu,
                       lamb)
    
    #Formula for European Call Price
    value = (D * np.exp(-beta * T) * V0 * cx1
             + D * (alpha / beta) * (1 - np.exp(-beta * T)) * cx2
             - D * K * cx3)
    
    return value
    































































































































