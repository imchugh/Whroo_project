# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:17:21 2016

@author: imchugh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import respiration_photosynthesis_run as rp_run
import DataIO as io



  
f = '/home/imchugh/Code/Python/Config_files/Whroo_master_configs.txt'
var_list = ['Fc', 'Fc_Sc', 'Fc_u*', 'Fc_Sc_u*']
stor_list = [False, True, False, True]
ustar_list = [0,
              0, 
              {'2011': 0.41,
               '2012': 0.40,
               '2013': 0.42,
               '2014': 0.42},
              {'2011': 0.31,
               '2012': 0.30,
               '2013': 0.32,
               '2014': 0.32}]
# Get the uncorrected data and gap-fill Fc
df = pd.DataFrame()
for i, var in enumerate(var_list):
    
    temp_dict = rp_run.main(use_storage = stor_list[i],
                            storage_var = 'Fc_storage_obs',
                            ustar_threshold = ustar_list[i],
                            config_file = f,
                            do_light_response = True)[0]
    temp_dict['NEE_est'] = temp_dict['Re'] + temp_dict['GPP']
    temp_dict['NEE_filled'] = temp_dict['NEE_series']
    temp_dict['NEE_filled'][np.isnan(temp_dict['NEE_filled'])] = \
        temp_dict['NEE_est'][np.isnan(temp_dict['NEE_filled'])]
    df[var] = temp_dict['NEE_filled']
    if i == 0:
        df.index = temp_dict['date_time']
 
# Do calculations of means for all groups
diurnal_df = df.groupby([lambda x: x.hour, lambda y: y.minute]).mean()
diurnal_df = pd.concat([diurnal_df.iloc[24:], diurnal_df.iloc[:24]])
diurnal_df.index = np.linspace(0, 23.5, 48)
diurnal_df['Fc_Sc_u*'].iloc[36:] = diurnal_df['Fc_Sc'].iloc[36:]

# Set plot iterables
vars_list = ['Fc', 'Fc_Sc', 'Fc_u*', 'Fc_Sc_u*']
             
names_dict = ['$F_c$', '$F_c\/+\/S_c$', '$F_{c\_u_*corr}$', '$(F_c\/+\/S_c)_{u_*corr}$']

lines_dict = {'Fc': ':',
              'Fc_Sc': '--',
              'Fc_u*': '-.',
              'Fc_Sc_u*': '-'}

# Instantiate plot
fig, ax = plt.subplots(1, 1, figsize = (12, 8))
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
x = diurnal_df.index
var1 = vars_list[0]
var2 = vars_list[1]
var3 = vars_list[2]
var4 = vars_list[3]
y1 = diurnal_df[var1]
y2 = diurnal_df[var2]
y3 = diurnal_df[var3]
y4 = diurnal_df[var4]
ax.plot(x, y1, color = 'grey', linewidth = 2, label = names_dict[0])
ax.plot(x, y2, color = 'black', linewidth = 2, linestyle = '--', label = names_dict[1])
ax.plot(x[9:42], y3.iloc[9:42], color = 'grey', linestyle = ':', 
        linewidth = 2, label = names_dict[2])
ax.plot(x, y4, color = 'black', linewidth = 2, label = names_dict[3])
ax.fill_between(x, y1, y2, where=y2>=y1, facecolor='blue', alpha = 0.3,
                edgecolor='None',interpolate=True)
ax.fill_between(x, y1, y2, where=y1>=y2, facecolor='blue', alpha = 0.3, 
                edgecolor='None',interpolate=True)
ax.fill_between(x, y2, y4, where=y3>=y2, facecolor='red', alpha = 0.3,
                edgecolor='None',interpolate=True)

#ax.legend(fontsize = 18, loc = 'upper left', frameon = False)
ax.axhline(y = 0, color = 'black', linestyle = '-')

ax.annotate('$Advection$' , 
            xy = (16, 1.8), 
            xytext = (20, 3),
            textcoords='data', verticalalignment='center',
            horizontalalignment = 'center',
            arrowprops = dict(arrowstyle="->"), fontsize = 16)
ax.annotate('$Storage$' , 
            xy = (8, 1.8), 
            xytext = (4, 3),
            textcoords='data', verticalalignment='center',
            horizontalalignment = 'center',
            arrowprops = dict(arrowstyle="->"), fontsize = 16)


fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
            'basic C paper/Images/diurnal_view_of_advection.png',
            bbox_inches='tight',
            dpi = 300)     
plt.show()