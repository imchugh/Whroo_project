# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:03:19 2016

@author: imchugh
"""

import numpy as np
import copy as cp
import pdb
import matplotlib.pyplot as plt
from scipy.stats import linregress

import DataIO as io
from AIC_class_test import NLS
import solar_functions as sf
import respiration as re_LT
import respiration_H2O as re_LT_H2O
import respiration_nonlin_H2O as re_nl_H2O
import respiration_lin_H2O as re_lin_H2O
import data_filtering as filt
import data_formatting as dt_fm

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

#------------------------------------------------------------------------------

def filter_data(this_dict):
    
    date_time = this_dict.pop('date_time')
    
    nan_bool_arr = this_dict['Fsd'] < 5
    try:
        for var in this_dict.keys():
            nan_bool_arr = nan_bool_arr & ~np.isnan(this_dict[var])
    except:
        pdb.set_trace()
    for var in this_dict.keys():
        this_dict[var] = this_dict[var][nan_bool_arr]
    
    this_dict['date_time'] = date_time[nan_bool_arr]
       
    return nan_bool_arr
    
###############################################################################
# Simple stats
# My AIC
def AIC(n_obs, n_params, rss):

    return (n_obs + n_obs * np.log(2 * np.pi) + n_obs *
            np.log(rss / n_obs) + 2 * (n_params + 1))

# My RSS
def RSS(y, y_hat):
    
    return ((y - y_hat)**2).sum()

###############################################################################
# Fit functions

def linear(p, x, y):
    a, b = p
    err = y - (a * x[:, 0] + b)
    return err

#------------------------------------------------------------------------------

def non_linear(p, x, y):
    a, b = p
    err = y - a * np.exp(x[:, 0] * b)
    return err
  
#------------------------------------------------------------------------------

def LT(p, x, y):
    rb, Eo = p
    err = y - rb * np.exp(Eo * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 
    return err

#------------------------------------------------------------------------------    
    
def LT_2comp(p, x, y):
    rb_ag, Eo_ag, rb_bg, Eo_bg = p
    ag_resp = rb_ag * np.exp(Eo_ag * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 
    bg_resp = rb_bg * np.exp(Eo_bg * (1 / (10 + 46.02) - 1 / (x[:, 1] + 46.02)))
    err = y - (ag_resp + bg_resp)
    return err
   
#------------------------------------------------------------------------------

def lin_H2O(p, x, y):
    a, b, theta_1, theta_2 = p
    err = y - ((a * x[:, 0] + b) *
               (1 / (1 + np.exp(theta_1 - theta_2 * x[:, 2]))))
    return err

#------------------------------------------------------------------------------

def non_lin_H2O(p, x, y):
    a, b, theta_1, theta_2 = p
    err = y - ((a * np.exp(x[:, 0] * b)) *
               (1 / (1 + np.exp(theta_1 - theta_2 * x[:, 2]))))
    return err

#------------------------------------------------------------------------------
    
def LT_H2O(p, x, y):
    rb, Eo, theta_1, theta_2 = p
    err = y - (rb * np.exp(Eo * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) *
               (1 / (1 + np.exp(theta_1 - theta_2 * x[:, 2]))))
    return err
    
#------------------------------------------------------------------------------ 

def LT_H2O_alt(p, x, y):
    rb, Eo, theta_v, c = p
    err = (y - rb * np.exp(Eo * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) *
           (np.exp(-0.5 * (abs(x[:, 2] - theta_v) / c)**2)))
    return err
    
#------------------------------------------------------------------------------ 

def LT_H2O_2comp(p, x, y):
    rb_ag, Eo_ag, rb_bg, Eo_bg, theta_1, theta_2 = p
    ag_resp = rb_ag * np.exp(Eo_ag * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 
    bg_resp = rb_bg * np.exp(Eo_bg * (1 / (10 + 46.02) - 1 / (x[:, 1] + 46.02)))
    sws_resp = 1 / (1 + np.exp(theta_1 - theta_2 * x[:, 2]))
    err = y - (ag_resp + bg_resp) * sws_resp
    return err

#------------------------------------------------------------------------------
    
###############################################################################
# Eval functions

def linear_eval(x, p):
    return (p[0] * x[:, 0] + p[1])

#------------------------------------------------------------------------------

def non_linear_eval(x, p):
    return p[0] * np.exp(x[:, 0] * p[1])

#------------------------------------------------------------------------------
    
def LT_eval(x, p):
    return p[0] * np.exp(p[1] * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 

#------------------------------------------------------------------------------
    
def LT_2comp_eval(x, p):
    ag_resp = p[0] * np.exp(p[1] * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 
    bg_resp = p[2] * np.exp(p[3] * (1 / (10 + 46.02) - 1 / (x[:, 1] + 46.02)))
    return ag_resp + bg_resp

#------------------------------------------------------------------------------    

def lin_H2O_eval(x, p):
    return (p[0] * x[:, 0] + p[1]) * (1 / (1 + np.exp(p[2] - p[3] * x[:, 2])))

#------------------------------------------------------------------------------  

def non_lin_H2O_eval(x, p):
    return (p[0] * np.exp(x[:, 0] * p[1])) * (1 / (1 + np.exp(p[2] - p[3] * x[:, 2])))

#------------------------------------------------------------------------------      

def LT_H2O_eval(x, p):
    return (p[0] * np.exp(p[1] * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) *
            (1 / (1 + np.exp(p[2] - p[3] * x[:, 2]))))

#------------------------------------------------------------------------------    

def LT_H2O_2comp_eval(x, p):
    ag_resp = p[0] * np.exp(p[1] * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) 
    bg_resp = p[2] * np.exp(p[3] * (1 / (10 + 46.02) - 1 / (x[:, 1] + 46.02)))
    sws_resp = 1 / (1 + np.exp(p[4] - p[5] * x[:, 2]))
    return (ag_resp + bg_resp) * sws_resp

#------------------------------------------------------------------------------     
    
def LT_H2O_alt_eval(x, p):
    return (p[0] * np.exp(p[1] * (1 / (10 + 46.02) - 1 / (x[:, 0] + 46.02))) *
            (np.exp(-0.5 * (abs(x[:, 2] - p[2]) / p[3])**2)))
    
###############################################################################
# Plot data

def plot_AIC_step_dependence(step_rslt_dict):
    
    fig, ax1 = plt.subplots(1, figsize = (12, 8))
    fig.patch.set_facecolor('white')
    ax2 = ax1.twinx()
    ax1.set_ylabel('$AIC$', fontsize = 18)
    ax1.set_xlabel('$Step\/(Days)$', fontsize = 18)
    ax1.tick_params(axis = 'x', labelsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax2.set_ylabel('$Parameters$', fontsize = 18)
    ax2.tick_params(axis = 'x', labelsize = 14)
    ax2.tick_params(axis = 'y', labelsize = 14)
    ser_1 = ax1.semilogx(step_rslt_dict['AIC'].keys()[1:], 
                         step_rslt_dict['AIC'].values()[1:], 's', 
                         ms = 10, color = 'grey',
                         label = 'AIC')
    ser_2 = ax2.semilogx(step_rslt_dict['n_params'].keys()[1:], 
                         step_rslt_dict['n_params'].values()[1:], 'o',
                         ms = 10, mfc = 'None', mew = 1, color = 'black', 
                         label = 'Parameters')
    [plt.setp(ax.get_yticklabels()[0], visible = False) for ax in [ax1, ax2]]
    all_ser = ser_1 + ser_2
    labs = [ser.get_label() for ser in all_ser]
    plt.legend(all_ser, labs, frameon = False, numpoints = 1)
    
    return

# User options
air_T_var = 'Ta'
do_storage = True
file_path = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L4.nc' 
altitude = 152
latitude = '-36.673215'
longitude = '145.029247'
GMT_zone = 10
config_file = '/home/imchugh/Code/Python/Config_files/Whroo_master_configs.txt'
plot = True

###############################################################################
# Prepare data

re_configs_dict = io.build_algorithm_configs_dict(config_file_location = 
                                                      config_file, 
                                                  algorithm = 
                                                      'respiration_configs')

data_dict = get_data(re_configs_dict)

filt.screen_low_ustar(data_dict, re_configs_dict['options']['ustar_threshold'],
                      re_configs_dict['options']['noct_threshold'])

solar_dict = sf.get_ephem_solar(data_dict, lat = latitude, lon = longitude, 
                                alt = altitude, GMT_zone = GMT_zone, 
                                return_var = 'altitude')

# Subset data
sub_dict = cp.deepcopy(data_dict)
sub_dict['solar_elevation'] = np.degrees(solar_dict['altitude'])
nan_bool = filter_data(sub_dict)

###############################################################################
# Do step trials with differing number of days

temp_configs_dict = cp.deepcopy(re_configs_dict)
step_rslt_dict = {'n_params': {}, 'AIC': {}}
n_obs = len(nan_bool[nan_bool])
for window in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30, 60, 120, 360]:
    
    temp_configs_dict['options']['step_size_days'] = window
    temp_configs_dict['options']['window_size_days'] = window
    
    this_rslt_dict, this_params_dict = re_LT_H2O.main(data_dict, 
                                                      temp_configs_dict['options'])[:2]
    
    steps = len(this_params_dict['rb_error_code'][this_params_dict['rb_error_code'] != 20])
    n_params = 9 + steps
    rss_step_model = RSS(data_dict['NEE_series'][nan_bool], 
                         this_rslt_dict['Re'][nan_bool])
    AIC_step_model = AIC(n_obs, n_params, rss_step_model)
    step_rslt_dict['n_params'][window] = n_params
    step_rslt_dict['AIC'][window] = AIC_step_model

###############################################################################
# Do step model trials for different respiration models

# Set some stuff up
step_init_list = [7, 365]
name_prefix_list = ['step', 'annual']
stat_rslt_dict = {'AIC': {}, 'RMSE': {}, 'r2': {}}

# Do 7 day
temp_configs_dict = cp.deepcopy(re_configs_dict) 
mod_list = ['LT', 'LT_H2O', 'lin_H2O', 'non_lin_H2O']
for i, step in enumerate(step_init_list):
    temp_configs_dict['options']['step_size_days'] = step
    temp_configs_dict['options']['window_size_days'] = step
    for j, model in enumerate([re_LT, re_LT_H2O, re_lin_H2O, re_nl_H2O]):
        model_name = '{0}_{1}'.format(name_prefix_list[i], mod_list[j])
        this_rslt_dict, this_params_dict = model.main(data_dict, 
                                                      temp_configs_dict['options'])[:2]
        steps = len(this_params_dict['rb_error_code'][this_params_dict['rb_error_code'] != 20])
        n_params = 9 + steps
        rss_step_model = RSS(data_dict['NEE_series'][nan_bool], 
                             this_rslt_dict['Re'][nan_bool])
        rmse_step_model = np.sqrt(rss_step_model / len(nan_bool[nan_bool]))
        AIC_step_model = AIC(n_obs, n_params, rss_step_model)
        r2_step_model = linregress(data_dict['NEE_series'][nan_bool], 
                                   this_rslt_dict['Re'][nan_bool])[2]**2
        stat_rslt_dict['RMSE'][model_name] = rmse_step_model
        stat_rslt_dict['AIC'][model_name] = AIC_step_model
        stat_rslt_dict['r2'][model_name] = r2_step_model

###############################################################################
# Do global fits

# Make independent and dependent variable arrays
x = np.empty([len(sub_dict['NEE_series']), 3])
x[:, 0] = sub_dict['TempC']
x[:, 1] = sub_dict['Ts']
x[:, 2] = sub_dict['Sws']
y = sub_dict['NEE_series']

# Make results dicts
model_dict = {}
pred_dict = {}

# Do prediction for linear
p0 = [1, 100]
params_dict = {var: p0[i] for i, var in enumerate(['a', 'b'])}
lin_mod = NLS(linear, params_dict, x, y)
model_dict['lin'] = lin_mod
pred_dict['lin'] = linear_eval(x, lin_mod.parmEsts) 

# Do prediction for non-linear
p0 = [1, 1]
params_dict = {var: p0[i] for i, var in enumerate(['a', 'b'])}
non_lin_mod = NLS(non_linear, params_dict, x, y)
model_dict['non_lin'] = non_lin_mod
pred_dict['non_lin'] = non_linear_eval(x, lin_mod.parmEsts)

# Do prediction for LT
p0 = [1, 100]
params_dict = {var: p0[i] for i, var in enumerate(['rb', 'Eo'])}
LT_mod = NLS(LT, params_dict, x, y)
model_dict['LT'] = LT_mod
pred_dict['LT'] = LT_eval(x, LT_mod.parmEsts)

# Do prediction for two-compartment LT_H2O
p0 = [1, 100, 1, 100]
params_dict = {var: p0[i] for i, var in enumerate(['rb_ag', 'Eo_ag', 'rb_bg',
                                                   'Eo_bg'])}
LT_2comp_mod = NLS(LT_2comp, params_dict, x, y)
model_dict['LT_2comp'] = LT_2comp_mod
pred_dict['LT_2comp'] = LT_2comp_eval(x, LT_2comp_mod.parmEsts)

# Do prediction for lin_H2O
p0 = [1, 100, 1, 10]
params_dict = {var: p0[i] for i, var in enumerate(['a', 'b', 'theta_1', 'theta_2'])}
lin_H2O_mod = NLS(lin_H2O, params_dict, x, y)
model_dict['lin_H2O'] = lin_H2O_mod
pred_dict['lin_H2O'] = lin_H2O_eval(x, lin_H2O_mod.parmEsts)

# Do prediction for non_lin_H2O
p0 = [1, 0.01, 1, 10]
params_dict = {var: p0[i] for i, var in enumerate(['a', 'b', 'theta_1', 'theta_2'])}
non_lin_H2O_mod = NLS(non_lin_H2O, params_dict, x, y)
model_dict['non_lin_H2O'] = non_lin_H2O_mod
pred_dict['non_lin_H2O'] = non_lin_H2O_eval(x, non_lin_H2O_mod.parmEsts)

# Do prediction for LT_H2O
p0 = [1, 100, 1, 10]
params_dict = {var: p0[i] for i, var in enumerate(['rb', 'Eo', 'theta_1', 'theta_2'])}
LT_H2O_mod = NLS(LT_H2O, params_dict, x, y)
model_dict['LT_H2O'] = LT_H2O_mod
pred_dict['LT_H2O'] = LT_H2O_eval(x, LT_H2O_mod.parmEsts)

# Do prediction for LT_H2O_alt
p0 = [1, 100, 1, 10]
params_dict = {var: p0[i] for i, var in enumerate(['rb', 'Eo', 'theta_v', 'c'])}
LT_H2O_alt_mod = NLS(LT_H2O_alt, params_dict, x, y)
model_dict['LT_H2O_alt'] = LT_H2O_alt_mod
pred_dict['LT_H2O_alt'] = LT_H2O_eval(x, LT_H2O_alt_mod.parmEsts)

# Do prediction for LT_H2O_2comp
p0 = [1, 100, 1, 100, 1, 10]
params_dict = {var: p0[i] for i, var in enumerate(['rb_ag', 'Eo_ag', 'rb_bg', 
                                                   'Eo_bg', 'theta_1', 'theta_2'])}
LT_H2O_2comp_mod = NLS(LT_H2O_2comp, params_dict, x, y)
model_dict['LT_H2O_2comp'] = LT_H2O_2comp_mod
pred_dict['LT_H2O_2comp'] = LT_H2O_2comp_eval(x, LT_H2O_2comp_mod.parmEsts)

# Do stats 
var_arr = np.array(model_dict.keys() + stat_rslt_dict['AIC'].keys())
AIC_arr = np.array([mod.AIC() for mod in model_dict.values()] + 
                   stat_rslt_dict['AIC'].values())
sort_index = np.argsort(AIC_arr)
AIC_arr = AIC_arr[sort_index]
var_arr = var_arr[sort_index]
delta_arr = AIC_arr - AIC_arr.min()
weights_sum = np.exp(-delta_arr / 2).sum()
weights_arr = np.exp(-delta_arr / 2) / weights_sum
