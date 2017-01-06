#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 11:54:23 2016

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
import respiration_H2O as re

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
num_cats = 50
config_file = '/home/imchugh/Code/Python/Config_files/Whroo_master_configs.txt'

# Do respiration

re_configs_dict = io.build_algorithm_configs_dict(config_file_location = 
                                                      config_file, 
                                                  algorithm = 
                                                      'respiration_configs')

data_dict = get_data(re_configs_dict, False)

filt_data_dict = cp.deepcopy(data_dict)
filt_data_dict['NEE_series'] = filt_data_dict['NEE_series'] + filt_data_dict['Sc']
filt.screen_low_ustar(filt_data_dict, re_configs_dict['options']['ustar_threshold'],
                      True)

re_rslt_dict, re_params_dict, re_error_dict = re.main(filt_data_dict, 
                                                      re_configs_dict['options'])

# Add Re and generate advection estimate
data_dict['Re'] = re_rslt_dict['Re']
data_dict['advection'] = (data_dict['Re'] - data_dict['NEE_series'] - 
                          data_dict['Sc'])

# Remove daytime, missing or filled data where relevant
df = pd.DataFrame(data_dict, index = data_dict['date_time'])
df['Re'] = re_rslt_dict['Re']
sub_df = cp.deepcopy(df)
sub_df = sub_df[sub_df.Fsd < 5]
sub_df = sub_df[['NEE_series', 'Sc', 'ustar', 'Re', 'advection']]
sub_df.dropna(inplace = True)  

# Categorise data into ustar bins then do means and Sds grouped by categories
sub_df['ustar_cat'] = pd.qcut(sub_df.ustar, num_cats, 
                              labels = np.linspace(1, num_cats, num_cats))
means_df = sub_df.groupby('ustar_cat').mean()
CI_df = (sub_df[['NEE_series','Sc','Re','advection', 'ustar_cat']]
         .groupby('ustar_cat').std() / 
         np.sqrt(sub_df[['NEE_series','Sc','Re','advection', 'ustar_cat']]
         .groupby('ustar_cat').count()) * 2)

# Create plot
fig, ax1 = plt.subplots(1, figsize = (12, 8))
fig.patch.set_facecolor('white')
ax1 = plt.gca()
ax1.plot(means_df.ustar, means_df.Re, linestyle = '--', 
         label = '$\hat{ER}$', color = 'black')
ax1.plot(means_df.ustar, means_df.NEE_series, linestyle = '-', 
         label = '$F_{c}$', color = 'black')
ax1.plot(means_df.ustar, means_df.Sc, linestyle = ':', 
         label = '$S_{c}$', color = 'black')                     
ax1.plot(means_df.ustar, means_df.advection, 
         linestyle = '-', label = '$A_c$', color = 'grey')
x = means_df.ustar
ylo = means_df.advection - CI_df.advection
yhi = means_df.advection + CI_df.advection
ax1.fill_between(x, ylo, yhi, where=yhi>=ylo, facecolor='0.8', 
                 edgecolor='None', interpolate=True)
ax1.axvline(x = 0.42, color  = 'grey')
ax1.axvline(x = 0.32, color  = 'grey')
ax1.axhline(y = 0, color  = 'black', linestyle = '-')
ax1.set_ylabel(r'$CO_2\/respiration\/(\mu mol\/m^{-2}\/s^{-1})$', fontsize = 22)
ax1.set_xlabel('$u_{*}\/(m\/s^{-1})$', fontsize = 22)
ax1.tick_params(axis = 'x', labelsize = 14)
ax1.tick_params(axis = 'y', labelsize = 14)
ax1.yaxis.set_ticks_position('left')
ax1.xaxis.set_ticks_position('bottom')    
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.setp(ax1.get_yticklabels()[0], visible = False)
ax1.legend(fontsize = 18, loc = [0.76,0.4], numpoints = 1, frameon = False)
fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
            'basic C paper/Images/ustar_vs_Fc_and_storage_advection.png',
            bbox_inches='tight',
            dpi = 300) 
plt.show()