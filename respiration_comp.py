# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:51:43 2016

@author: imchugh
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
import matplotlib.pyplot as plt
import copy as cp
import pdb

import DataIO as io
import data_filtering as filt

#------------------------------------------------------------------------------
# Data optimisation algorithms
def TRF_1(data_dict, theta_1, theta_2, Eo, rb):
    
    pot_er = rb * np.exp(Eo * (1 / (10 + 46.02) - 1 / (data_dict['T_above'] + 46.02)))
    
    vwc_ind = 1 / (1 + np.exp(theta_1 - theta_2 * data_dict['Sws']))
    
    er = pot_er * vwc_ind
    
    return er

def TRF_2(data_dict, theta_1, theta_2, Eo_a, rb_a, Eo_b, rb_b):
    
    ag_er = rb_a * np.exp(Eo_a * (1 / (10 + 46.02) - 1 / (data_dict['T_above'] + 46.02)))
    bg_er = rb_b * np.exp(Eo_b * (1 / (10 + 46.02) - 1 / (data_dict['Ts'] + 46.02)))
    
    vwc_ind = 1 / (1 + np.exp(theta_1 - theta_2 * data_dict['Sws']))
    
    er = (ag_er + bg_er) * vwc_ind
    
    return er
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def filter_data(this_dict):
    
    this_dict.pop('date_time')
    
    nan_bool_arr = (this_dict['Fsd'] < 5) & (this_dict['ustar'] > 0.32)
    print len(nan_bool_arr)
    valid_index = np.where((this_dict['Fsd'] < 5) & (this_dict['ustar'] > 0.32))
    filter_arr = this_dict['NEE_series'][nan_bool_arr]
    comp_arr = filt.IQR_filter(filter_arr, inplace = False)
    test_bool_arr = np.nan_to_num(filter_arr) == np.nan_to_num(comp_arr)
    new_bool_arr = nan_bool_arr.copy()
    new_bool_arr[valid_index] = test_bool_arr
    print len(nan_bool_arr)

    for var in this_dict.keys():
        if not var == 'NEE_series':
            nan_bool_arr = nan_bool_arr & ~np.isnan(this_dict[var])
            
    for var in this_dict.keys():
        this_dict[var] = this_dict[var][nan_bool_arr]
        
    return nan_bool_arr, new_bool_arr
#------------------------------------------------------------------------------

# User options
air_T_var = 'Flu'
file_path = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L4.nc' 

# Get data
data_dict = io.OzFluxQCnc_to_data_structure(file_path)

# Specify variable to use for temperature
if air_T_var == 'Flu':
    data_dict['T_above'] = (data_dict['Flu'] / (5.67*10**-8))**(1.0/4)-273.15
else:
    data_dict['T_above'] = data_dict[air_T_var]

# Truncate dataset to necessary variables
data_dict['NEE_series'] = data_dict['Fc'] + data_dict['Fc_storage_obs']
data_dict = {var: data_dict[var] for var in ['date_time', 'NEE_series', 'Fsd', 
                                             'T_above', 'Ts', 'Sws', 'ustar']}

# Do the fit across all data then estimate ER for single compartment model
sub_dict = cp.deepcopy(data_dict)
nan_bool, new_bool_arr = filter_data(sub_dict)
#params_1, test_covar_1 = curve_fit(TRF_1, 
#                                   sub_dict, 
#                                   sub_dict['NEE_series'],
#                                   p0 = [5, 10, 200, 2])
#er_1 = TRF_1(data_dict, params_1[0], params_1[1], params_1[2], params_1[3])
#
#
## Do the fit across all data then estimate ER for two compartment model
#params_2, test_covar_2 = curve_fit(TRF_2, 
#                                   sub_dict, 
#                                   sub_dict['NEE_series'],
#                                   p0 = [5, 10, 200, 2, 200, 2])
#er_2 = TRF_2(data_dict, params_2[0], params_2[1], params_2[2], params_2[3],
#             params_2[4], params_2[5])
#
#rmse = np.sqrt((data_dict['NEE_series'][nan_bool]-er[nan_bool])**2).mean()
#
#r2 = linregress(er_2[nan_bool], data_dict['NEE_series'][nan_bool])[2] ** 2
#
#all_recs_years_arr = np.array([date.year for date in data_dict['date_time']])
#years_list = list(set(all_recs_years_arr))
#
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
#    results_dict[year] = {key: results_array.copy() for key in ['rb_a', 'rb_b']}
#
#    # Iterate through the months
#    for month in months_list:
#        print ('The year is {0}; the month is {1}').format(str(year), str(month))
#        # Create a month dict
#        month_index = np.where(all_recs_months_arr == month) 
#        month_dict = {var: year_dict[var][month_index] for var in year_dict.keys()}
#        nan_bool = filter_data(month_dict)
#        
#        # Only process if enough data
#        if len(nan_bool[nan_bool]) > 0:
#        
#            # Do the fitting and write to results dictionary
#            response_arr = month_dict.pop('NEE_series')
#            params, cov = curve_fit(lambda x, rb_a, rb_b:
#                                    TRF_2(x, params_2[0], params_2[1], 
#                                             params_2[2], rb_a, params_2[4], rb_b),  
#                                    month_dict,
#                                    response_arr, 
#                                    p0 = [params_2[3], params_2[5]])
#            results_dict[year]['rb_a'][month - 1] = params[0]
#            results_dict[year]['rb_b'][month - 1] = params[1]
#            plt.plot(month_dict['T_above'], response_arr, 'o')