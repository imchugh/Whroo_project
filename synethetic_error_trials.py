#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:51:46 2016

@author: imchugh
"""
import pdb
import copy as cp
import numpy as np
import matplotlib.pyplot as plt

import DataIO as io
import data_formatting as dt_fm
import data_filtering as filt
import respiration as re
import photosynthesis as ps
import random_error as rand_err

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

config_file = '/home/imchugh/Code/Python/Config_files/Whroo_master_configs.txt'

###############################################################################
# Function for respiration

def do_respiration():

    filt.screen_low_ustar(data_dict, re_configs_dict['options']['ustar_threshold'],
                          re_configs_dict['options']['noct_threshold'])
    
    re_rslt_dict, re_params_dict, re_error_dict = re.main(data_dict, 
                                                          re_configs_dict['options'])

###############################################################################
# Do plotting

def plot_rb(rb_dict):
    
    return
    
###############################################################################
# Do respiration

re_configs_dict = io.build_algorithm_configs_dict(config_file_location = 
                                                      config_file, 
                                                  algorithm = 
                                                      'respiration_configs')

data_dict = get_data(re_configs_dict)

filt.screen_low_ustar(data_dict, re_configs_dict['options']['ustar_threshold'],
                      re_configs_dict['options']['noct_threshold'])

re_rslt_dict, re_params_dict, re_error_dict = re.main(data_dict, 
                                                      re_configs_dict['options'])

###############################################################################
# Constants

num_trials = 10000
###############################################################################
# Do photosynthesis

ps_configs_dict = io.build_algorithm_configs_dict(config_file_location = 
                                                       config_file, 
                                                  algorithm = 
                                                       'photosynthesis_configs')

data_dict = get_data(ps_configs_dict)

data_dict['PAR'] = data_dict['Fsd'] * 0.46 * 4.6

li_rslt_dict, li_params_dict, li_error_dict = ps.main(data_dict, 
                                                      ps_configs_dict['options'], 
                                                      re_params_dict)

###############################################################################
# Do random error

rand_configs_dict = io.build_algorithm_configs_dict(config_file_location = 
                                                        config_file, 
                                                    algorithm = 
                                                       'random_error_configs')

data_dict = get_data(rand_configs_dict)
data_dict['NEE_model'] = re_rslt_dict['Re'] + li_rslt_dict['GPP']

fig, stats_dict, bins = rand_err.regress_sigma_delta(data_dict, 
                                                     rand_configs_dict['options'])

sigma_delta = rand_err.estimate_sigma_delta(data_dict['NEE_model'], stats_dict)

###############################################################################
# Get year indices

data_years_array = np.array([this_date.year for this_date in data_dict['date_time']])
Eo_years_array = np.array([this_date.year for this_date in re_params_dict['date']])
years_list = list(set(data_years_array))
data_index_dict = {}
Eo_index_dict = {}
for this_year in years_list:
    data_index_dict[this_year] = np.where(data_years_array == this_year)
    Eo_index_dict[this_year] = np.where(Eo_years_array == this_year)

###############################################################################
# Set up results dicts    
    
dummy_arr = np.empty(num_trials)
Csum_dict = {this_year: dummy_arr.copy() for this_year in years_list}
dummy_arr = np.zeros(len(re_params_dict['rb']))
rb_dict = {this_var: dummy_arr for this_var in ['rb_sum', 'rb_sum_sq']}
dummy_arr = np.zeros(4)
Eo_dict = {this_var: dummy_arr.copy() for this_var in ['Eo_sum', 'Eo_sum_sq']}
             
###############################################################################
# Do MC simulation

for this_trial in xrange(num_trials):

    error = rand_err.estimate_random_error(sigma_delta)
    synth_dict = cp.copy(data_dict)
    this_bool = np.isnan(synth_dict['NEE_series'])
    synth_dict['NEE_series'] = synth_dict.pop('NEE_model') + error
    synth_dict['NEE_series'][this_bool] = np.nan
    re_synth_rslt_dict, re_synth_params_dict, re_synth_error_dict = (
        re.main(synth_dict, re_configs_dict['options']))
    
    for this_year in years_list:
        
        this_arr = re_synth_rslt_dict['Re'][data_index_dict[this_year]] 
        ans = (this_arr * 1800 * 12 * 10**-6).sum()
        Csum_dict[this_year][this_trial] = ans

        this_arr = re_synth_params_dict['Eo'][Eo_index_dict[this_year]]
        Eo_dict['Eo_sum'][years_list.index(this_year)] = (
            Eo_dict['Eo_sum'][years_list.index(this_year)] + this_arr.mean())
        Eo_dict['Eo_sum_sq'][years_list.index(this_year)] = (
            Eo_dict['Eo_sum_sq'][years_list.index(this_year)] + this_arr.mean()**2)

    rb_dict['rb_sum'] = rb_dict['rb_sum'] + re_synth_params_dict['rb']
    rb_dict['rb_sum_sq'] = rb_dict['rb_sum_sq'] + re_synth_params_dict['rb']**2
    

st_dev_rb = np.sqrt((rb_dict['rb_sum_sq'] - (rb_dict['rb_sum'])**2 / num_trials) / num_trials)
st_dev_Eo = np.sqrt((Eo_dict['Eo_sum_sq'] - (Eo_dict['Eo_sum'])**2 / num_trials) / num_trials)

###############################################################################

