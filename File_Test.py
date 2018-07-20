import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import cm
from Chapter01 import MCoptionvaluation, Volatility
from Chapter03 import bsm_call_value, bsm_vega, bsm_call_imp_vol
import datetime as dt
import Dates_and_Times as DT
from bsm_option_class import call_option
import Chapter03 as cp3

h5 = pd.HDFStore('C:\\Users\\Marlombrei\\Documents\\Python_For_Finance\\vstoxx_data_31032014.h5','r')
futures_data = h5['futures_data']
options_data = h5['options_data']
h5.close()
options_data['IMP_VOL'] = 0.0
V0 = 17.6639
r = 0.01
tol = 0.5 

for option in options_data.index:
    forward = futures_data[futures_data['MATURITY'] == options_data.loc[option]['MATURITY']]['PRICE'].values[0]
    if (forward * (1-tol) < options_data.loc[option]['STRIKE']):
        imp_vol = cp3.bsm_call_imp_vol(V0,
                                       options_data.loc[option]['STRIKE'],
                                       options_data.loc[option]['TTM'],
                                       r,
                                       options_data.loc[option]['PRICE'],
                                       sigma_est=2.0,
                                       it=100)
        options_data['IMP_VOL'].loc[option] = imp_vol

print (options_data.loc[46170])






















