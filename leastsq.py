# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:03:19 2016

@author: imchugh
"""

import numpy as np
import copy as cp
import pdb

import DataIO as io
from AIC_class_test import NLS
import solar_functions as sf
import random_error as rand_err
import respiration_photosynthesis_run as rpr

#------------------------------------------------------------------------------

def filter_data(this_dict):
    
    date_time = this_dict.pop('date_time')
    
    nan_bool_arr = (this_dict['Fsd'] < 5) & (this_dict['ustar'] > 0.32)
    for var in this_dict.keys():
        nan_bool_arr = nan_bool_arr & ~np.isnan(this_dict[var])
            
    for var in this_dict.keys():
        this_dict[var] = this_dict[var][nan_bool_arr]
    
    this_dict['date_time'] = date_time[nan_bool_arr]
       
    return nan_bool_arr
    
#------------------------------------------------------------------------------

def linear(p, x, y):
    a, b = p
    err = y - (a * x[:, 0] + b)
    return err
    
#------------------------------------------------------------------------------

def linear_eval(x, p):
    return (p[0] * x[:, 0] + p[1])

#------------------------------------------------------------------------------

def LT(p, x, y):
    rb, Eo = p
    err = y - rb * np.exp(Eo * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 
    return err
    
#------------------------------------------------------------------------------

def LT_eval(x, p):
    return p[0] * np.exp(p[1] * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 

#------------------------------------------------------------------------------

def LT_H2O(p, x, y):
    rb, Eo, theta_1, theta_2 = p
    err = y - (rb * (np.exp(Eo * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) *
               (1 / (1 + np.exp(theta_1 - theta_2 * x[:, 2])))))
    return err
    
#------------------------------------------------------------------------------

def LT_H2O_eval(x, p):
    return (p[0] * np.exp(p[1] * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) *
            (1 / (1 + np.exp(p[2] - p[3] * x[:, 2]))))

#------------------------------------------------------------------------------

def LT_2comp(p, x, y):
    rb_ag, Eo_ag, rb_bg, Eo_bg = p
    ag_resp = rb_ag * np.exp(Eo_ag * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 
    bg_resp = rb_bg * np.exp(Eo_bg * (1 / (10 + 46.02) - 1 / (x[:, 1] + 46.02)))
    err = y - (ag_resp + bg_resp)
    return err

#------------------------------------------------------------------------------

def LT_2comp_eval(x, p):
    ag_resp = p[0] * np.exp(p[1] * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 
    bg_resp = p[2] * np.exp(p[3] * (1 / (10 + 46.02) - 1 / (x[:, 1] + 46.02)))
    return ag_resp + bg_resp

#------------------------------------------------------------------------------

#
#def LT_H2O_2comp(p, x, y):
#    rb_ag, Eo_ag, rb_bg, Eo_bg, theta_1, theta_2 = p
#    ag_resp = rb_ag * np.exp(Eo_ag * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 
#    bg_resp = rb_bg * np.exp(Eo_bg * (1 / (10 + 46.02) - 1 / (x[:, 1] + 46.02)))
#    sws_resp = 1 / (1 + np.exp(theta_1 - theta_2 * x[:, 2]))
#    err = y - (ag_resp + bg_resp) * sws_resp
#    return err
#
##------------------------------------------------------------------------------
#
#def LT_H2O_2comp(x, p):
#    ag_resp = p[0] * np.exp(p[1] * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 
#    bg_resp = p[2] * np.exp(p[3] * (1 / (10 + 46.02) - 1 / (x[:, 1] + 46.02)))
#    sws_resp = 1 / (1 + np.exp(p[4] - p[5] * x[:, 2]))
#    return (ag_resp + bg_resp) * sws_resp
#
##------------------------------------------------------------------------------

# User options
air_T_var = 'Ta'
do_storage = True
file_path = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L4.nc' 
altitude = 152
latitude = '-36.673215'
longitude = '145.029247'
GMT_zone = 10

# Build configuration dictionary for random error
rand_configs_dict = {'measurement_interval': 30,
                     'pos_averaging_bins': 10,
                     'neg_averaging_bins': 10,
                     'radiation_difference_threshold': 35,
                     'temperature_difference_threshold': 3,
                     'windspeed_difference_threshold': 1,
                     'mean_series': 'NEE_series'}

# Get data
data_dict = io.OzFluxQCnc_to_data_structure(file_path)

## Get model and add mode4l NEE to data dict
#model_dict = rpr.main(do_light_response = True)[0]
#data_dict['NEE_model'] = model_dict['GPP'] + model_dict['Re']
#
## Add storage if using it
#if do_storage:
#    data_dict['NEE_series'] = data_dict['Fc'] + data_dict['Fc_storage_obs']
#else:
#    data_dict['NEE_series'] = data_dict['Fc']
#
## Specify variable to use for temperature
#if air_T_var == 'Flu':
#    data_dict['TempC'] = (data_dict['Flu'] / (5.67*10**-8))**(1.0/4)-273.15
#else:
#    data_dict['TempC'] = data_dict[air_T_var]
#
#data_dict['ws'] = data_dict.pop('Ws')
#
## Truncate dataset to necessary variables
#if do_storage:
#    data_dict['NEE_series'] = data_dict['Fc'] + data_dict['Fc_storage_obs']
#else:
#    data_dict['NEE_series'] = data_dict['Fc']
##data_dict = {var: data_dict[var] for var in ['date_time', 'NEE_series', 'Fsd', 
##                                             'Ta', 'Ts', 'Sws', 'ustar']}
#
#test = rand_err.regress_sigma_delta(data_dict, rand_configs_dict)                                             
                                             
solar_dict = sf.get_ephem_solar(data_dict, lat = latitude, lon = longitude, 
                                alt = altitude, GMT_zone = GMT_zone, 
                                return_var = 'altitude')

sub_dict = cp.deepcopy(data_dict)
sub_dict['solar_elevation'] = np.degrees(solar_dict['altitude'])
nan_bool = filter_data(sub_dict)

# Make independent and dependent variable arrays
x = np.empty([len(sub_dict['NEE_series']), 3])
x[:, 0] = sub_dict['TempC']
x[:, 1] = sub_dict['Ts']
x[:, 2] = sub_dict['Sws']
y = sub_dict['NEE_series']

results_dict = {}

# Do prediction for linear
p0 = [1, 100]
params_dict = {var: p0[i] for i, var in enumerate(['a', 'b'])}
lin_mod = NLS(linear, params_dict, x, y)
lin_pred = linear_eval(x, lin_mod.parmEsts)
results_dict['lin'] = lin_mod

# Do prediction for LT
p0 = [1, 100]
params_dict = {var: p0[i] for i, var in enumerate(['rb', 'Eo'])}
LT_mod = NLS(LT, params_dict, x, y)
LT_pred = LT_eval(x, LT_mod.parmEsts)
results_dict['LT'] = LT_mod

# Do prediction for two-compartment LT_H2O
p0 = [1, 100, 1, 100]
params_dict = {var: p0[i] for i, var in enumerate(['rb_ag', 'Eo_ag', 'rb_bg',
                                                   'Eo_bg'])}
LT_2comp_mod = NLS(LT_2comp, params_dict, x, y)
LT_2comp_pred = LT_2comp_eval(x, LT_2comp_mod.parmEsts)
results_dict['LT_2comp'] = LT_2comp_mod

# Do prediction for LT_H2O
p0 = [1, 100, 1, 10]
params_dict = {var: p0[i] for i, var in enumerate(['rb', 'Eo', 'theta_1', 'theta_2'])}
LT_H2O_mod = NLS(LT_H2O, params_dict, x, y)
LT_H2O_pred = LT_H2O_eval(x, LT_H2O_mod.parmEsts)
results_dict['LT_H2O'] = LT_H2O_mod
                                                   
for mod in results_dict.keys():
    print 'AIC for model {0} = {1}'.format(mod, str(round(results_dict[mod].AIC())))
    print 'RMSE for model {0} = {1}'.format(mod, str(round(results_dict[mod].RMSE, 3)))

## Do prediction for LT without soil moisture
#params_dict = {var: p0[i] for i, var in enumerate(['rb', 'Eo'])}
#LT_noH2O_mod = NLS(L_T_noH2O, params_dict, x, y)
#noH2O_pred = L_T_noH2O_eval(x, LT_noH2O_mod.parmEsts)

#all_recs_years_arr = np.array([date.year for date in data_dict['date_time']])
#years_list = list(set(all_recs_years_arr))

## Iterate through the years
#results_dict = {}
#for year in years_list:
#    
#    # Create a new dictionary containing only the nocturnal data for the relevant year
#    year_index = np.where(all_recs_years_arr == year)
#    year_dict = {var: data_dict[var][year_index] for var in data_dict.keys()}
#
#    # Create a month array and list for indexing and selecting the correct month
#    all_recs_months_arr = np.array([date.month for date in year_dict['date_time']])
#    months_list = list(set(all_recs_months_arr))
#
#    # Create a results_array
#    results_array = np.empty(12)
#    results_array[:] = np.nan
#    results_dict[year] = {}
#
#    # Iterate through the months
#    for month in months_list:
#
#        print 'Year is {0} and month is {1}'.format(str(year), str(month))
#        
#        # Create a month dict
#        month_index = np.where(all_recs_months_arr == month) 
#        month_dict = {var: year_dict[var][month_index] for var in year_dict.keys()}
#        nan_bool = filter_data(month_dict)
#        
#        # Only process if enough data
#        total_data = len(nan_bool)
#        valid_data = len(nan_bool[nan_bool])
#        if valid_data > 0:
#        
#            # Do the fitting and write to results dictionary
#            x = np.empty([len(month_dict['NEE_series']), 2])
#            x[:, 0] = month_dict['Ta']
#            x[:, 1] = month_dict['Sws']
#            y = month_dict['NEE_series']
#            try:
#                LT_mod = NLS(L_T, params_dict, x, y)
#            except:
#                LT_mod = None
#            results_dict[year][month] = LT_mod