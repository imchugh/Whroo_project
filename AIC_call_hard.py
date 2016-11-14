# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:19:30 2016

@author: imchugh
"""

import pandas as pd
import numpy as np 

from AIC_class_test import NLS

# Create dummy data
a = 2
b = 190
c = 5
d = 50
temp = np.linspace(0, 30, 61)
sws = np.linspace(0.5, 0.1, 61)
T_signal = a * np.exp(b * (1 / (10 + 46.02) - 1 / (temp + 46.02)))
#sws_signal_scalar = 1 / (1 + np.exp(c - d * sws))
noise = np.random.normal(0, 0.1, 61)
resp = T_signal + noise #* sws_signal_scalar 
 
# Define a simpler model
def LTMod(params, indeps, dep):
    a = params[0]
    c = params[1]

    yHat = a * np.exp(b * (1 / (10 + 46.02) - 1 / (indeps + 46.02)))
 
    err = dep - yHat
    return(err)

pl_t = {'a':2, 'b':200}
 
# First, define the likelihood null model
def linMod(params, mass, yObs):
    a = params[0]
    c = params[1]
 
    yHat = a*mass+c
    err = yObs - yHat
    return(err)
 
pl = {'a':1, 'b':1}

ltMod = NLS(LTMod, pl_t, temp, resp)
 
nlMod = NLS(linMod, pl, temp, resp)