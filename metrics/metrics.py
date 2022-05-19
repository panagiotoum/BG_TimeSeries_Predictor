import pandas as pd
from sklearn import preprocessing
import numpy as np
import math
import scipy.io
from scipy import stats

# RMSE
def rmse(y_actual,y_predicted,N):
    x = sum((y_actual-y_predicted)**2)/N
    rmse = np.sqrt(x)
    return rmse

# MARD
def mard(y_actual,y_predicted,N):
    temp = sum(np.divide(abs(y_predicted-y_actual),y_actual))
    mard_test = (temp/N)
    return mard_test

## CC
def cc(y_actual,y_predicted):
    cc, p_value = scipy.stats.pearsonr(y_predicted,y_actual)
    return cc, p_value
