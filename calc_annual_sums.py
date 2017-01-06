# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:17:21 2016

@author: imchugh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import copy as cp

import DataIO as io
import data_formatting as dt_fm
import data_filtering as filt
import respiration as re
import photosynthesis as ps

###############################################################################
# Function to get data

def get_data(configs_dict):
    
    data_file = configs_dict['files']['in_file']
    var_list = configs_dict['variables'].values()
    data_dict, attr = io.OzFluxQCnc_to_data_structure(data_file, 
                                                      var_list = var_list, 
                                                      QC_var_list = ['Fc'], 
                                                      return_global_attr = True)

    configs_dict['options']['measurement_interval'] = int(attr['time_step'])

    names_dict = dt_fm.get_standard_names(convert_dict = configs_dict['variables'])
    data_dict = dt_fm.rename_data_dict_vars(data_dict, names_dict)
    
    return data_dict    

###############################################################################  

config_file = '/home/imchugh/Code/Python/Config_files/Whroo_master_configs.txt'

# Get configs dict and combine
re_configs_dict = io.build_algorithm_configs_dict(config_file_location = 
                                                      config_file, 
                                                  algorithm = 
                                                      'respiration_configs')
ps_configs_dict = io.build_algorithm_configs_dict(config_file_location = 
                                                       config_file, 
                                                  algorithm = 
                                                       'photosynthesis_configs')
re_configs_dict['variables'].update(ps_configs_dict['variables'])

# Get data and make a boolean for records with valid entries for all variables
data_dict = get_data(re_configs_dict)
retain_bool = (~np.isnan(data_dict['NEE_series']) & ~np.isnan(data_dict['Sc']))

# Make the unfiltered series and unify cases so n is same for all
C_dict = {}
C_dict['Fc'] = data_dict.pop('NEE_series')
C_dict['Fc_Sc'] = C_dict['Fc'] + data_dict.pop('Sc')
for var in C_dict.keys():
    C_dict[var][~retain_bool] = np.nan

# Now add filtered series

# Make Fc u* filtered lo
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.26, '2012': 0.26, '2013': 0.19, '2014': 0.23},
                      re_configs_dict['options']['noct_threshold'],
                      True)
C_dict['Fc_u*_lo'] = data_dict.pop('NEE_series')

# Make Fc u* filtered med
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.40, '2012': 0.39, '2013': 0.40, '2014': 0.42},
                      re_configs_dict['options']['noct_threshold'],
                      True)
C_dict['Fc_u*_med'] = data_dict.pop('NEE_series')

# Make Fc u* filtered hi
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.52, '2012': 0.52, '2013': 0.61, '2014': 0.61},
                      re_configs_dict['options']['noct_threshold'],
                      True)
C_dict['Fc_u*_hi'] = data_dict.pop('NEE_series')

# Make Fc u* filtered with storage lo
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc_Sc'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.01, '2012': 0.01, '2013': 0, '2014': 0.02},
                      re_configs_dict['options']['noct_threshold'],
                      True)
C_dict['Fc_Sc_u*_lo'] = data_dict.pop('NEE_series')

# Make Fc u* filtered with storage lo
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc_Sc'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.31, '2012': 0.30, '2013': 0.32, '2014': 0.32},
                      re_configs_dict['options']['noct_threshold'],
                      True)
C_dict['Fc_Sc_u*_med'] = data_dict.pop('NEE_series')

# Make Fc u* filtered hi
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc_Sc'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.59, '2012': 0.59, '2013': 0.73, '2014': 0.62},
                      re_configs_dict['options']['noct_threshold'],
                      True)
C_dict['Fc_Sc_u*_hi'] = data_dict.pop('NEE_series')

# Update variables in data dict and configs dict to allow photosynthesis to run
data_dict['PAR'] = data_dict['Fsd'] * 0.46 * 4.6
ps_configs_dict['options']['measurement_interval'] = (
    re_configs_dict['options']['measurement_interval'])

# Now do gap filling
rslt_dict = {}
for var in C_dict.keys():
    
    data_dict['NEE_series'] = cp.deepcopy(C_dict[var])
    try:
        re_rslt_dict, re_params_dict = re.main(data_dict, 
                                               re_configs_dict['options'])[:2]
        ps_rslt_dict = ps.main(data_dict, 
                               ps_configs_dict['options'],
                               re_params_dict)[0]
        this_model_array = re_rslt_dict['Re'] + ps_rslt_dict['GPP']
        idx = np.isnan(data_dict['NEE_series'])
        data_dict['NEE_series'][idx] = this_model_array[idx]
        C_dict[var] = data_dict.pop('NEE_series')
    except:
        print 'Couldnt fill {0}'.format(var)
        continue

# Do calculations: sums and means for all groups
annual_dict = {}
df = pd.DataFrame(C_dict, index = data_dict['date_time'])
years_list = list(set([this_date.year for this_date in data_dict['date_time']]))
for this_year in years_list:
    annual_dict[this_year] = (df.loc[str(this_year)] * 1800 * 10**-6 * 12).sum()
