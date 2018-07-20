from __future__ import division
import numpy as np
import datetime as dt

dates = [dt.datetime(2015,1,1),
         dt.datetime(2015,7,1),
         dt.datetime(2016,1,1)]

def get_year_delta(date_list, day_count=365):
    start = date_list[0]
    delta_list = [(date - start).days / 365 for date in date_list]
    return np.array(delta_list)

frac = get_year_delta(dates)

class constant_short_rate(object):
    
    def __init__(self, name, short_rate):
        self.name = name
        self.short_rate = short_rate
        if short_rate < 0:
            raise ValueError('Short_rate is negative')
    
    def get_discount_factors(self, date_list, dtobjects=True):
        if dtobjects is True:
            dlist = get_year_delta(date_list)
        else:
            dlist = np.array(date_list)
        
        dflist = np.exp(-self.short_rate * np.sort(dlist))
        return np.array((date_list,dflist)).T
    
csr = constant_short_rate('csr',0.05)
data = csr.get_discount_factors(frac, dtobjects=False)
print(data)


print(np.exp(-0.05*0))

print(np.exp(-0.05*0.49589041))
print(np.exp(-0.05*1))












































































































































































































