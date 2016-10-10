# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:32:39 2016

@author: imchugh
"""

# Standard modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# My modules
import respiration_photosynthesis_run as rp_run
import utility_funcs

reload(rp_run)
  
f = '/home/imchugh/Code/Python/Config_files/Whroo_master_configs.txt'
var_list = ['Fc', 'Fc_Sc', 'Fc_Sc_u*']
stor_list = [False, True, True]
ustar_list = [0,
              0, 
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
diurnal_df.index = np.linspace(0, 23.5, 48)

# Set plot iterables
vars_dict = {1: ['Fc', 'Fc_Sc'],
             2: ['Fc_Sc', 'Fc_Sc_u*']}
             
names_dict = {1: ['$F_c$', '$F_c\/+\/S_c$'],
              2: ['$F_c\/+\/S_c$', '$(F_c\/+\/S_c)_{u_*corr}$']}

lines_dict = {'Fc': ':',
              'Fc_Sc': '--',
              'Fc_Sc_u*': '-'}

# Instantiate plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 6), sharex = True)
fig.patch.set_facecolor('white')
fig_labels = ['a)', 'b)', 'c)', 'd)']

for i, ax in enumerate((ax1, ax2)):

    counter = i + 1
    ax.set_xlim([0, 24])
    ax.set_ylim([-10, 4])
    ax.set_xticks([0,4,8,12,16,20,24])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$Time\/(hours)$', fontsize = 18)
    if counter % 2 != 0:
        ax.set_ylabel(r'$NEE\/(\mu mol\/CO_2\/m^{-2}\/s^{-1})$', fontsize = 18)
    x = diurnal_df.index
    var1 = vars_dict[counter][0]
    var2 = vars_dict[counter][1]
    y1 = diurnal_df[var1]
    y2 = diurnal_df[var2]
    ax.plot(x, y1, color = 'black', linestyle = lines_dict[var1], 
            linewidth = 2, label = names_dict[counter][0])
    ax.plot(x, y2, color = 'black', linestyle = lines_dict[var2], 
            linewidth = 2, label = names_dict[counter][1])
    ax.fill_between(x, y1, y2, where=y2>=y1, facecolor='0.8', 
                    edgecolor='None',interpolate=True)
    ax.fill_between(x, y1, y2, where=y1>=y2, facecolor='0.8', 
                    edgecolor='None',interpolate=True)

    ax.legend(fontsize = 18, loc = 'lower right', frameon = False)
    ax.axhline(y = 0, color = 'black', linestyle = '-')
    ax.text(-2.6, 3.8, fig_labels[i], fontsize = 12)

#fig.savefig(utility_funcs.set_output_path('diurnal_NEE_effects_of_storage_correction.png'),
#            bbox_inches='tight',
#            dpi = 300)     
plt.show()