'''
Created on 17 Aug 2017

@author: Marlombrei
'''
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
pd.options.display.width = 500
pd.options.display.max_rows = 20


def bsm_call_value(S0,K,T,r,sigma):
    '''Valuation of European call option in BSM model
    Parameters
    ====================
    S0 : float
        initial stock level
    K : float
        strike price
    T : float
        maturity (in year fractions)
    r : float
        constant risk-free short-term rate
    RETURNS
    ====================
    value : float
        return present value of call option
    '''
    from math import log, sqrt, exp
    from scipy import stats
    
    S0 = float(S0)
    d1 = (log(S0/K) + (r + 0.5*sigma**2)*(T)) / (sigma * sqrt(T))
    d2 = (log(S0/K) + (r - 0.5*sigma**2)*(T)) / (sigma * sqrt(T))
    value = (S0 * stats.norm.cdf(d1,0.0,1.0)
             - K * exp(-r*T) * stats.norm.cdf(d2,0.0,1.0))
    #stats.norm.cdf : cumulative distribution function for normal distribution
    return value

def bsm_vega(S0,K,T,r,sigma):
    '''Vega of European option in BSM model
    
    Returns
    ================
    vega : float
        partial derivative of BSM formula with respect to sigma
    '''
    from math import log, sqrt
    from scipy import stats
    S0 = float(S0)
    d1 = (log(S0/K) + (r + 0.5*sigma**2)*(T)) / (sigma * sqrt(T))
    vega = S0 * stats.norm.cdf(d1,0.0,1.0) * sqrt(T)
    return vega


def bsm_call_imp_vol(S0,K,T,r,C0,sigma_est, it=100):
    '''Implied volatility of European call option in the BSM model
    
    it : integer
        number of iterations
        
    Returns
    ================
    sigma_ext : float
        numerically estimated implied volatility
    '''
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0,K,T,r,sigma_est) - C0) #Calculated Call price with guessed sigma minus C0
                      / (bsm_vega(S0,K,T,r,sigma_est))) #Calculated Vega using guessed sigma
    return sigma_est



v0 = 17.6639
r = 0.01

path = r'C:\Users\Marlombrei\Documents\Python_For_Finance\vstoxx_data_31032014.h5'
h5 = pd.HDFStore(path, 'r')
futures = h5['futures_data'] #VSTOXX futures data
options = h5['options_data'] #VSTOXX options data
h5.close()

options['DATE'] = pd.to_datetime(options['DATE'])
futures['DATE'] = pd.to_datetime(futures['DATE'])
options['MATURITY'] = pd.to_datetime(options['MATURITY'])
futures['MATURITY'] = pd.to_datetime(futures['MATURITY'])
options['IMP_VOL'] = 0.0

#print(futures)
#print(options)

tol = 0.50 # tolerance level for moneyness (50% maximum deviation from future level) it was the authors choice
for option in options.index:
    forward = futures[futures['MATURITY'] == options.loc[option]['MATURITY']]['PRICE'].values[0]
    #picking the right value
    if (forward * (1 - tol)
        < options.loc[option]['STRIKE']
        < forward * (1 + tol)):
        
        #only options with moneyness within tolerance
        imp_vol = bsm_call_imp_vol(
            S0=17.6639, #v0 is VSTOXX Value
            K=options.loc[option]['STRIKE'],
            T=options.loc[option]['TTM'],
            r=0.01,
            C0=options.loc[option]['PRICE'],
            sigma_est=2.0,
            it=100)
        options['IMP_VOL'].loc[option] = imp_vol

plot_data = options[options['IMP_VOL'] > 0]

maturities = sorted(set(options['MATURITY']))

import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(maturities)











































