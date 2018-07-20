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



y = np.random.standard_normal(20)

x = range(len(y))

plt.plot(y.cumsum())
plt.grid(True)
plt.xlim(-1,20)
plt.ylim(np.min(y.cumsum())-1,
         np.max(y.cumsum())+1)
plt.show()






























































































