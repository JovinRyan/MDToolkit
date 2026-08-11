import numpy as np 
import scipy as sc 
import statsmodels.api as sm 
from scipy.signal import savgol_filter

def lowess(x, y, window = 100):
    '''
    '''
    smooth = sm.nonparametric.lowess(endog = y, exog = x, frac = window/len(y))

    return smooth[:, 0], smooth[:, 1]

def savgol():
    '''
    '''