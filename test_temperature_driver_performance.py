# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:25:40 2016

@author: imchugh
"""

# Python standard modules
import os
import copy as cp
import numpy as np
import pdb

# My modules
import DataIO as io
import respiration as re
import data_formatting as dt_fm
import datetime as dt
import data_filtering as filt

reload(dt_fm)

###############################################################################
# Function to get data

def get_data(configs_dict, use_storage = True):
    
    data_file = configs_dict['files']['in_file']
    var_list = configs_dict['variables'].values()
    data_dict, attr = io.OzFluxQCnc_to_data_structure(data_file, 
                                                      var_list = var_list, 
                                                      QC_var_list = ['Fc'], 
                                                      return_global_attr = True)
    configs_dict['options']['measurement_interval'] = int(attr['time_step'])

    if use_storage:
        data_dict['Fc'] = data_dict['Fc'] + data_dict['Fc_storage_obs']

    names_dict = dt_fm.get_standard_names(convert_dict = configs_dict['variables'])
    data_dict = dt_fm.rename_data_dict_vars(data_dict, names_dict)
    
    return data_dict    

###############################################################################

config_file = '/home/imchugh/Code/Python/Config_files/Whroo_master_configs.txt'

# Do respiration

re_configs_dict = io.build_algorithm_configs_dict(config_file_location = 
                                                      config_file, 
                                                  algorithm = 
                                                      'respiration_configs')

re_configs_dict['variables'].update({'36m': 'Ta_HMP_32m',
                                     '16m': 'Ta_HMP_16m',
                                     '8m': 'Ta_HMP_8m',
                                     '4m': 'Ta_HMP_4m',
                                     '2m': 'Ta_HMP_2m',
                                     '0.5m': 'Ta_HMP_1m',
                                     'outgoing_longwave_radiation': 'Flu'})

data_dict = get_data(re_configs_dict)
data_dict['Ta_HMP_36m'] = data_dict.pop('Ta_HMP_32m')

filt.screen_low_ustar(data_dict, re_configs_dict['options']['ustar_threshold'],
                      re_configs_dict['options']['noct_threshold'])

re_rslt_dict, re_params_dict, re_error_dict = re.main(data_dict, 
                                                      re_configs_dict['options'])

# Throw out late 2014 data
time_clip_bool_array = data_dict['date_time'] < dt.datetime(2014,6,30)
for key in data_dict.keys():
    data_dict[key] = data_dict[key][time_clip_bool_array]

data_dict['SRT'] = (data_dict['Flu'] / (5.67*10**-8))**(1.0/4)-273.15
    
# Create index for cases where T is valid for all variables
bool_list = []
T_vars = ['Ta_HMP_2m', 'Ta_HMP_4m', 'Ta_HMP_8m', 'Ta_HMP_16m', 
          'Ta_HMP_36m', 'Ts', 'SRT']
for var in T_vars:
    bool_list.append(~np.isnan(data_dict[var]))
master_bool = bool_list[0] & bool_list[1] & bool_list[2] & bool_list[3] & bool_list[4]
NEE_bool = ~np.isnan(data_dict['NEE_series'])

# Remove data where missing, then check which temp series has lowest error
#RMSE_list = []
#for var in T_vars:
#    data_dict['TempC'] = data_dict[var]
#    data_dict[var][np.isnan(master_bool)] = np.nan
#    
#    # Calculate Re                            
#    re_rslt_dict, re_params_dict, re_error_dict = re.main(cp.copy(data_dict), 
#                                                          re_configs_dict['options'])
#                                                          
#    re_est_array = re_rslt_dict['Re'][(data_dict['Fsd'] < 5) & master_bool & NEE_bool]
#    re_obs_array = data_dict['NEE_series'][(data_dict['Fsd'] < 5) & master_bool & NEE_bool]
#    
#    RMSE_list.append(np.sqrt((re_obs_array - re_est_array) **2).mean())
#
#a = {var: RMSE_list[i] for i, var in enumerate(T_vars)}
     
# Test temperature weighting
for var in T_vars:
    rmse = 100
    for weight in np.linspace(0, 1, 11):
        data_dict['TempC'] = weight * data_dict['Ts'] + (1 - weight) * data_dict[var]
        data_dict[var][np.isnan(master_bool)] = np.nan

        # Calculate Re                            
        re_rslt_dict, re_params_dict, re_error_dict = re.main(cp.copy(data_dict), 
                                                              re_configs_dict['options'])
                                                              
        re_est_array = re_rslt_dict['Re'][(data_dict['Fsd'] < 5) & master_bool & NEE_bool]
        re_obs_array = data_dict['NEE_series'][(data_dict['Fsd'] < 5) & master_bool & NEE_bool]
        
        this_rmse = np.sqrt((re_obs_array - re_est_array) **2).mean()
        
        if this_rmse < rmse: 
            rmse = this_rmse
            weight_of_best = weight
        
    print ('For variable {0}, lowest RMSE {1} was found with weighting of {2}'
           .format(var, str(np.round(rmse, 7)), str(weight_of_best)))