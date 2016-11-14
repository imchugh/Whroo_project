# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:19:30 2016

@author: imchugh
"""

import pandas as pd
import numpy as np 

from AIC_class_test import NLS

# Create dummy data
a = 10
b = 1.2
temp = np.linspace(0, 30, 61)
signal = a * temp ** b
noise = np.random.normal(0, 5, 61)
resp = signal + noise
a = pd.DataFrame({'Temp': temp, 'respDaily': resp})
 
# Create the Arrhenius temperature
 
#respData['Ar'] = -1 / (8.617 * 10**-5 * (respData['Temp']+273))

# Define a simpler model
def linMod(params, mass, yObs):
    a = params[0]
    c = params[1]
 
    yHat = a*mass+c
    err = yObs - yHat
    return(err)

pl = {'a':0, 'b':1}
 
# First, define the likelihood null model
def nonlinMod(params, mass, yObs):
    a = params[0]
    c = params[1]
 
    yHat = a*mass**c
    err = yObs - yHat
    return(err)
 
pnl = {'a':1, 'b':1}

lMod = NLS(linMod, pl, temp, resp)
 
nlMod = NLS(nonlinMod, pnl, temp, resp)