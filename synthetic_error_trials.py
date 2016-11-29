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
import datetime as dt
import math
import pandas as pd
from scipy.stats import linregress

import DataIO as io
import data_formatting as dt_fm
import data_filtering as filt
import respiration_new as re
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

###############################################################################
# Do plotting

def plot_rb(data_dict):

    fig, ax = plt.subplots(1, figsize = (12, 8))
    fig.patch.set_facecolor('white')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    #ax.set_xlabel('$Date$', fontsize = 18)
    ax.set_ylabel(r'$rb\/(\mu mol C\/m^{-2}\/s^{-1})$', fontsize = 18)
    ax.set_xlim([dt.date(2012, 1, 1), dt.date(2015, 1, 1)])
    min_y = 0
    max_y = math.ceil((data_dict['rb'] + 
                       data_dict['rb_SD']).max())
    ax.set_ylim([0, max_y])
    
    tick_locs = [i for i in data_dict['date'] if 
                 (i.month == 1 or i.month == 7) and i.day == 1]
    tick_labs = ['January' if i.month == 1 else 'July' for i in tick_locs]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labs, rotation = 45, fontsize = 14)
    year_pos = [i for i in tick_locs if i.month == 7]
    
    text_vert_coord = (max_y - min_y) * 0.95
    for this_date in year_pos:
        ax.text(this_date, text_vert_coord, dt.datetime.strftime(this_date, '%Y'), 
                fontsize = 20, horizontalalignment = 'center',
                verticalalignment = 'center')
    
    vert_lines = [i for i in tick_locs if i.month == 1]
    for line in vert_lines:
        ax.axvline(line, color = 'black', ls = ':')             
    
    # Box to demarcate missing data
    ax.axhspan(0.6, 2.6, 0.553, 0.603, edgecolor = 'black', ls = '--', fill = False)
    ax.annotate('Missing\ndata', 
                xy = (dt.datetime(2013, 9, 23), 2.6), 
                xytext = (dt.datetime(2013, 9, 23), 3.2),
                textcoords='data', verticalalignment='center',
                horizontalalignment = 'center',
                arrowprops = dict(arrowstyle="->"), fontsize = 18)
        
    ax.plot(data_dict['date'], data_dict['rb'], 
            color = 'black', lw = 1.5)
    ax.fill_between(data_dict['date'], 
                    data_dict['rb'] + 2 *
                    data_dict['rb_SD'], 
                    data_dict['rb'] - 2 *
                    data_dict['rb_SD'],
                    facecolor = '0.75', edgecolor = 'None')
    
    fig.tight_layout()
    fig.show()
    
    return

def plot_noise(data_dict, synth_dict):

    num_cats = 30
    
    df = pd.DataFrame(data_dict)
    df['NEE_synth'] = synth_dict['NEE_series']
    df['NEE_model'] = synth_dict['NEE_model']
    df = df[df.Fsd < 5]
    df = df[df.ustar > 0.32]
    df = df[['TempC', 'NEE_series', 'NEE_synth', 'NEE_model', 'Sws']]
    df.dropna(inplace=True)
    
    # Put into temperature categories
    df.sort_values(by = 'TempC')
    df['TempC_cat'] = pd.qcut(df.TempC, num_cats, 
                      labels = np.linspace(1, num_cats, num_cats))
    
    # Do grouping
    mean_df = df.groupby('TempC_cat').mean()
    std_df = df.groupby('TempC_cat').std()
    
    # Calculate stats
    r2_mod_modnoise = np.round(linregress(df.NEE_model, df.NEE_synth).rvalue ** 2, 2)
    r2_mod_obs = np.round(linregress(df.NEE_model, df.NEE_series).rvalue ** 2, 2)
    
    # Now plot
    fig, ax = plt.subplots(1, figsize = (12, 8))
    fig.patch.set_facecolor('white')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([-6, 10])
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    ax.set_xlabel('$Air\/temperature\/(^oC)$', fontsize = 18)
    ax.set_ylabel(r'$NEE\/(\mu mol C\/m^{-2}\/s^{-1})$', fontsize = 18)
    ax.axhline(0, color = 'black')
    
    color_real = 'blue'
    color_synth = 'red'
    ax.plot(df['TempC'], df['NEE_series'], 
            'o', ms = 3, alpha = 0.2, mfc = 'None', mec = color_real, label = '')
    ax.plot(df['TempC'], df['NEE_synth'], 
            'o', ms = 3, alpha = 0.2, mfc = 'None', mec = color_synth, label = '')    
    ax.plot(mean_df['TempC'], mean_df['NEE_series'], color = color_real, lw = 3,
            label = '$observations\/(r^2\/=\/{0})$'.format(str(r2_mod_obs)))
    ax.plot(mean_df['TempC'], mean_df['NEE_series'] + std_df['NEE_series'], 
            ls = ':', lw = 3, color = color_real, label = '')
    ax.plot(mean_df['TempC'], mean_df['NEE_series'] - std_df['NEE_series'], 
            ls = ':', lw = 3, color = color_real, label = '')
    ax.plot(mean_df['TempC'], mean_df['NEE_synth'], color = color_synth, lw = 3,
            label = '$model\/+\/noise\/(r^2\/=\/{0})$'.format(str(r2_mod_modnoise)))
    ax.plot(mean_df['TempC'], mean_df['NEE_synth'] + std_df['NEE_synth'], 
            ls = ':', lw = 3, color = color_synth, label = '')
    ax.plot(mean_df['TempC'], mean_df['NEE_synth'] - std_df['NEE_synth'], 
            ls = ':', lw = 3, color = color_synth, label = '')
    ax.legend(loc = [0.6, 0.1], frameon = False, fontsize = 18)
    
    return

###############################################################################
# Constants
num_trials = 1
config_file = '/home/imchugh/Code/Python/Config_files/Whroo_master_configs.txt'
    
##############################################################################
# Do respiration

re_configs_dict = io.build_algorithm_configs_dict(config_file_location = 
                                                      config_file, 
                                                  algorithm = 
                                                      'respiration_configs')

re_data_dict = get_data(re_configs_dict)

filt.screen_low_ustar(re_data_dict, re_configs_dict['options']['ustar_threshold'],
                      re_configs_dict['options']['noct_threshold'])

re_rslt_dict, re_params_dict, re_error_dict = re.main(re_data_dict, 
                                                      re_configs_dict['options'])

###############################################################################
# Do photosynthesis

ps_configs_dict = io.build_algorithm_configs_dict(config_file_location = 
                                                       config_file, 
                                                  algorithm = 
                                                       'photosynthesis_configs')

ps_data_dict = get_data(ps_configs_dict)

ps_data_dict['PAR'] = ps_data_dict['Fsd'] * 0.46 * 4.6

li_rslt_dict, li_params_dict, li_error_dict = ps.main(ps_data_dict, 
                                                      ps_configs_dict['options'], 
                                                      re_params_dict)

###############################################################################
# Do random error

rand_configs_dict = io.build_algorithm_configs_dict(config_file_location = 
                                                        config_file, 
                                                    algorithm = 
                                                        'random_error_configs')

rand_data_dict = get_data(rand_configs_dict)
rand_data_dict['NEE_model'] = re_rslt_dict['Re'] + li_rslt_dict['GPP']

fig, stats_dict, bins = rand_err.regress_sigma_delta(rand_data_dict, 
                                                     rand_configs_dict['options'])

sigma_delta = rand_err.estimate_sigma_delta(rand_data_dict['NEE_model'], stats_dict)

###############################################################################
# Get year indices

data_years_array = np.array([this_date.year for this_date in re_data_dict['date_time']])
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

    synth_dict = cp.deepcopy(re_data_dict)
    synth_dict['error'] = rand_err.estimate_random_error(sigma_delta)
    this_bool = np.isnan(re_data_dict['NEE_series'])
    synth_dict['NEE_model'] = rand_data_dict['NEE_model']
    synth_dict['NEE_series'] = synth_dict['NEE_model'] + synth_dict['error']
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
    

re_synth_params_dict['rb_SD'] = np.sqrt((rb_dict['rb_sum_sq'] - 
                                         (rb_dict['rb_sum'])**2 / num_trials) /
                                        num_trials)
re_synth_params_dict['Eo_SD'] = np.sqrt((Eo_dict['Eo_sum_sq'] - 
                                         (Eo_dict['Eo_sum'])**2 / num_trials) / 
                                        num_trials)

###############################################################################
# Do plotting

plot_noise(re_data_dict, synth_dict)


