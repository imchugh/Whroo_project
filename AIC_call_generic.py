# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:19:30 2016

@author: imchugh
"""

import numpy as np 

from scipy.optimize import leastsq

from AIC_class_test import NLS

# Create dummy data
a = 2
b = 290
c = 5
d = 50
temp = np.linspace(0, 30, 200)
#signal = a * temp ** b
T_signal = a * np.exp(b * (1 / (10 + 46.02) - 1 / (temp + 46.02)))
sws = np.linspace(0.5, 0.1, 200)
sws_signal_scalar = 1 / (1 + np.exp(c - d * sws))
noise = np.random.normal(0, 0.5, 200)
resp = T_signal * sws_signal_scalar + noise
#a = pd.DataFrame({'Temp': temp, 'respDaily': resp})
 
# Define a simpler model
def linMod(params, temp, yObs):
    a = params[0]
    b = params[1]
 
    yHat = a * temp + b
    err = yObs - yHat
    return(err)

pl = {'a':0, 'b':1}
 
# First, define the likelihood null model
def nonlinMod(params, indeps, yObs):
    a = params[0]
    b = params[1]
 
    yHat = a * indeps**b
    err = yObs - yHat
    return(err)

pnl = {'a':1, 'b':1}

# First, define the likelihood null model
def loytMod(params, indeps, yObs):
    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
 
    temp = indeps[:, 0]
    sws = indeps[:, 1]
    T_func = a * np.exp(b * (1 / (10 + 46.02) - 1 / (temp + 46.02)))
    sws_func = 1 / (1 + np.exp(c - d * sws))
    yHat = T_func * sws_func
    err = yObs - yHat
    return(err)
 
p_lt = {'a': 2, 'b': 300, 'c': 5, 'd': 50}

lMod = NLS(linMod, pl, temp, resp)
 
nlMod = NLS(nonlinMod, pnl, temp, resp)

indeps = np.zeros([200, 2])
indeps[:, 0] = temp
indeps[:, 1] = sws



#ltMod = NLS(loytMod, p_lt, indeps, resp)
