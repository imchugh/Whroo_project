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
re_configs_dict['variables']['alternative_carbon_storage'] = 'Fc_storage'

# Get data and make a boolean for records with valid entries for all variables
data_dict = get_data(re_configs_dict)
retain_bool = (~np.isnan(data_dict['NEE_series']) & ~np.isnan(data_dict['Sc']) &
               ~np.isnan(data_dict['Fc_storage']))

# Make the unfiltered series and unify cases so n is same for all
C_dict = {}
C_dict['Fc'] = data_dict.pop('NEE_series')
C_dict['Fc_Sc'] = C_dict['Fc'] + data_dict.pop('Sc')
C_dict['Fc_Sc_pt'] = C_dict['Fc'] + data_dict.pop('Fc_storage')
for var in C_dict.keys():
    C_dict[var][~retain_bool] = np.nan

# Now add filtered series

# Make u* filtered with no storage
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.40, '2012': 0.39, '2013': 0.40, '2014': 0.42},
                      re_configs_dict['options']['noct_threshold'],
                      True)
C_dict['Fc_u*'] = data_dict.pop('NEE_series')

# Make u* filtered with no storage and only nocturnal ustar filtering
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.40, '2012': 0.39, '2013': 0.40, '2014': 0.42},
                      re_configs_dict['options']['noct_threshold'],
                      False)
C_dict['Fc_u*_noct'] = data_dict.pop('NEE_series')

# Make u* filtered with storage
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc_Sc'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.31, '2012': 0.30, '2013': 0.32, '2014': 0.32},
                      re_configs_dict['options']['noct_threshold'],
                      True)
C_dict['Fc_Sc_u*'] = data_dict.pop('NEE_series')

# Make u* filtered with storage and only nocturnal filter
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc_Sc'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.31, '2012': 0.30, '2013': 0.32, '2014': 0.32},
                      re_configs_dict['options']['noct_threshold'],
                      False)
C_dict['Fc_Sc_u*_noct'] = data_dict.pop('NEE_series')

# Make u* filtered with storage
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc_Sc_pt'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.31, '2012': 0.30, '2013': 0.32, '2014': 0.32},
                      re_configs_dict['options']['noct_threshold'],
                      True)
C_dict['Fc_Sc_pt_u*'] = data_dict.pop('NEE_series')

# Make u* filtered with storage and only nocturnal filter
data_dict['NEE_series'] = cp.deepcopy(C_dict['Fc_Sc_pt'])
filt.screen_low_ustar(data_dict, 
                      {'2011': 0.31, '2012': 0.30, '2013': 0.32, '2014': 0.32},
                      re_configs_dict['options']['noct_threshold'],
                      False)
C_dict['Fc_Sc_pt_u*_noct'] = data_dict.pop('NEE_series')

# Update variables in data dict and configs dict to allow photosynthesis to run
data_dict['PAR'] = data_dict['Fsd'] * 0.46 * 4.6
ps_configs_dict['options']['measurement_interval'] = (
    re_configs_dict['options']['measurement_interval'])

# Now do gap filling
rslt_dict = {}
for var in C_dict.keys():
    
    data_dict['NEE_series'] = cp.deepcopy(C_dict[var])
    re_rslt_dict, re_params_dict = re.main(data_dict, 
                                           re_configs_dict['options'])[:2]
    ps_rslt_dict = ps.main(data_dict, 
                           ps_configs_dict['options'],
                           re_params_dict)[0]
    this_model_array = re_rslt_dict['Re'] + ps_rslt_dict['GPP']
    idx = np.isnan(data_dict['NEE_series'])
    data_dict['NEE_series'][idx] = this_model_array[idx]
    C_dict[var] = data_dict.pop('NEE_series')
    
# Do calculations: sums and means for all groups
annual_dict = {}
df = pd.DataFrame(C_dict, index = data_dict['date_time'])
years_list = list(set([this_date.year for this_date in data_dict['date_time']]))
for this_year in years_list:
    annual_dict[this_year] = (df.loc[str(this_year)] * 1800 * 10**-6 * 12).sum()
#
diurnal_df = df.groupby([lambda x: x.hour, lambda y: y.minute]).mean()
diurnal_df = pd.concat([diurnal_df.iloc[24:], diurnal_df.iloc[:24]])
diurnal_df.index = np.linspace(0, 23.5, 48)

# Instantiate plot
fig, ax = plt.subplots(1, 1, figsize = (16, 8))
fig.patch.set_facecolor('white')
ax.set_xlim([0, 24])
ax.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,24])
ax.set_xticklabels([12,14,16,18,20,22,0,2,4,6,8,10,12], fontsize = 14)
ax.set_ylim([-10, 4])
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis = 'x', labelsize = 14)
ax.tick_params(axis = 'y', labelsize = 14)
ax.set_xlabel(r'$Time\/(hours)$', fontsize = 18)
ax.set_ylabel(r'$NEE\/(\mu mol\/CO_2\/m^{-2}\/s^{-1})$', fontsize = 18)
plt.setp(ax.get_yticklabels()[0], visible = False)
x = diurnal_df.index
vars_list = ['Fc', 'Fc_Sc', 'Fc_Sc_u*', 'Fc_u*', 'Fc_Sc_pt_u*']
names_list = ['$F_c$', '$F_c\/+\/S_c$', '$(F_c\/+\/S_c)_{u_*corr}$', 
              '$F_{c\_u_*corr}$', '$(F_c\/+\/S_{c\_pt})_{u_*corr}$']
y1 = diurnal_df[vars_list[0]]
y2 = diurnal_df[vars_list[1]]
y3 = diurnal_df[vars_list[2]]
y4 = diurnal_df[vars_list[3]]
y5 = diurnal_df[vars_list[4]]
ax.plot(x, y1, color = 'grey', linewidth = 2, label = names_list[0])
ax.plot(x, y4, color = 'black', linewidth = 2, ls = ':', 
        label = names_list[3])
ax.plot(x, y2, color = 'black', linewidth = 2, linestyle = '--', 
        label = names_list[1])
ax.plot(x, y3, color = 'black', linewidth = 2, label = names_list[2])
ax.fill_between(x, y1, y2, where=y2>=y1, facecolor='blue', alpha = 0.3,
                edgecolor='None',interpolate=True)
ax.fill_between(x, y1, y2, where=y1>=y2, facecolor='blue', alpha = 0.3, 
                edgecolor='None',interpolate=True)
ax.fill_between(x, y2, y3, where=y3>=y2, facecolor='red', alpha = 0.3,
                edgecolor='None',interpolate=True)
ax.plot(x, y5, color = 'black', marker = 'o', ms = 8,
        mfc = 'None', ls = '', mew = 1, label = names_list[4])

ax.legend(fontsize = 16, loc = [0.1, 1], frameon = False, ncol = 5, 
          numpoints = 1)
ax.axhline(y = 0, color = 'black', linestyle = '-')

ax.annotate('$Advection$' , 
            xy = (16, 1.8), 
            xytext = (20, 3),
            textcoords='data', verticalalignment='center',
            horizontalalignment = 'center',
            arrowprops = dict(arrowstyle="->"), fontsize = 18)
ax.annotate('$Storage$' , 
            xy = (8, 1.8), 
            xytext = (4, 3),
            textcoords='data', verticalalignment='center',
            horizontalalignment = 'center',
            arrowprops = dict(arrowstyle="->"), fontsize = 18)

ax_inset = plt.axes([.337, .22, .35, .35], axisbg = 'white')
ax_inset.plot(x, diurnal_df['Fc_u*'] - diurnal_df['Fc_u*_noct'], 
              color = 'black', ls = ':')
ax_inset.plot(x, diurnal_df['Fc_Sc_u*'] - diurnal_df['Fc_Sc_u*_noct'], 
              color = 'black')
ax_inset.plot(x, diurnal_df['Fc_Sc_pt_u*'] - diurnal_df['Fc_Sc_pt_u*_noct'], 
              color = 'black', ls = '', marker = 'o', mfc = 'None', mew = 1)
ax_inset.xaxis.set_ticks_position('bottom')
ax_inset.yaxis.set_ticks_position('left')
ax_inset.set_xticks([0,4,8,12,16,20,24])
ax_inset.set_xticklabels([12,16,20,0,4,8,12])
ax_inset.set_xlim([0, 24])
ax_inset.set_ylim([-0.8, 0.4])
ax_inset.text(2, -0.45, 'Difference between series with\nand without daytime $u_*$ filtering')

fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
            'basic C paper/Images/diurnal_view_of_advection.png',
            bbox_inches='tight',
            dpi = 300)     
plt.show()