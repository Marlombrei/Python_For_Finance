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


pd.options.display.width = 500
pd.options.display.max_rows = 10

def day_of_the_week():
    to = dt.datetime.today().weekday()
    days = {0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
    for key in days:
        if to == key:
            print (days[key])
    
#     wd = lambda x: days[x] #another way of doing this but not using FOR Loop
#     return wd(to)

print (day_of_the_week())

d = dt.datetime(2016,10,31,10,5,30,500000)
print(d)

print([d.replace(second=0, microsecond=0)])

td = d - dt.datetime.now()

print([td])

print(td.days)


print([np.datetime64('2015-10-31')])

print(np.datetime64('2015-10','3D'))

ts = pd.Timestamp('2016-06-30')
print(ts)

d = ts.to_pydatetime()
print(d)

























































































































































