from __future__ import division, print_function
import pandas as pd
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

#import Chp_15 as chp15

def std_normal_random_numbers(shape, antithetic=True, moment_matching=True, fixed_seed=False):
    """
    comments on the book
    """
    if fixed_seed:
        np.random.seed(1000)
    if antithetic:
        ran = np.random.standard_normal((shape[0], shape[1], shape[2] / 2))
        ran = np.concatenate((ran,-ran), axis=2)
    else:
        ran = np.random.standard_normal(shape)
    
    if moment_matching:
        ran = ran - np.mean(ran)
        ran = ran / np.std(ran)
    
    if shape[0] == 1:
        return ran[0]
    else:
        return ran
    

class simulation_class(object):
    """
    Providing base methods for simulation classes
    """
    def __init__(self, name, mar_env, corr):
        try:
            self.name = name
            self.pricing_date = mar_env.pricing_date
            self.initial_value = mar_env.get_constant('initial_value')
            self.volatility = mar_env.get_constant('volatility')
            self.final_date = mar_env.get_constant('final_date')
            self.currency = mar_env.get_constant('currency')
            self.frequency = mar_env.get_constant('frequency')
            self.paths = mar_env.get_constant('paths')
            self.discount_curve = mar_env.get_constant('discount_curve')
            try:
                # if time_grid in mar_env take this
                # (for portfolio valuation)
                self.time_grid = mar_env.get_list('time_grid')
            except:
                self.time_grid = None
            
            try:
                # if there are special dates, then add these
                self.special_dates = mar_env.get_list('special_dates')
            except:
                self.special_dates = []
            
            self.instrument_values = None
            self.correlated = corr
            if corr is True:
                # only needed in a portfolio context when
                # risk factors are correlated
                self.cholesky_matrix = mar_env.get_list('cholesky_matrix')
                self.rn_set = mar_env.get_list('rn_set')[self.name]
                self.random_numbers = mar_env.get_list('random_numbers')
        except:
            print('Error parsing market environment.')


    def generate_time_grid(self):
        start = self.pricing_date
        end = self.final_date
        
        # pandas date_range function
        # freq = 'B' for Business Day
        # 'W' for Weekly, 'M' for Monthly
        
        time_grid = pd.date_range(start=start,
                                  end=end,
                                  freq=self.frequency).to_pydatetime()
        time_grid = list(time_grid)
        
        #enhance 
        

































































































