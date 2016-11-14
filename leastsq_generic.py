# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:03:19 2016

@author: imchugh
"""

from scipy.optimize import leastsq
from scipy.optimize import least_squares
import numpy as np
from AIC_class_test import NLS

#x = np.arange(0,6e-2,6e-2/30)
#A,k,theta = 10, 1.0/3e-2, np.pi/6
#y_true = A*np.sin(2*np.pi*k*x+theta)
#y_meas = y_true + 2*np.random.randn(len(x))
#
#def residuals(p, y, x):
#    A,k,theta = p
#    err = y-A*np.sin(2*np.pi*k*x+theta)
#    return err
#
#def peval(x, p):
#    return p[0]*np.sin(2*np.pi*p[1]*x+p[2])
#
#p0 = [8, 1/2.3e-2, np.pi/3]
#print np.array(p0)
#
#plsq = leastsq(residuals, p0, args=(y_meas, x))
#print plsq[0]
#
#print np.array([A, k, theta])

T_series = np.linspace(0, 30, 200)
sws_series = np.linspace(0.5, 0.1, 200)
rb, Eo, theta_1, theta_2 = 2, 290, 5, 50 
T_response = rb * np.exp(Eo * (1 / (10 + 46.02) - 1 / (T_series + 46.02)))
sws_response = 1 / (1 + np.exp(theta_1 - theta_2 * sws_series))
y_true = T_response * sws_response
y_meas = y_true + np.random.normal(0, 0.5, 200)

def residuals(p, x, y):
    a, b, c, d = p
    err = y - a * (np.exp(b * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) *
                   (1 / (1 + np.exp(c - d * x[:, 1]))))
    return err

def peval(x, p):
    return (p[0] * np.exp(p[1] * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) *
            (1 / (1 + np.exp(p[2] - p[3] * x[:, 1]))))

p0 = [1, 100, 1, 10]
x = np.empty([200, 2])
x[:, 0] = T_series
x[:, 1] = sws_series
plsq = leastsq(residuals, p0, args = (x, y_meas), full_output = 1)
plsq_2 = least_squares(residuals, p0, args = (x, y_meas), loss = 'linear')
print plsq[0]

params_dict = {var: p0[i] for i, var in enumerate(['rb', 'Eo', 'theta_1', 'theta_2'])}
thisMod = NLS(residuals, params_dict, x, y_meas)