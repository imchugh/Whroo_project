# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:35:06 2016

@author: imchugh
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

import DataIO as io

   
"""
This script plots Sc as a function of time of day; it also shows the major 
respiration drivers on the right hand axis!
"""    

file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L6.nc'

storage_vars = ['Fc_storage_obs_1', 'Fc_storage_obs_2', 
                'Fc_storage_obs_3', 'Fc_storage_obs_4', 'Fc_storage_obs_5', 
                'Fc_storage_obs_6', 'Fc_storage_obs', 'Fc_storage']    

df = io.OzFluxQCnc_to_data_structure(file_in, 
                                     var_list = (storage_vars + 
                                                 ['ustar','Ta', 'Fc',
                                                 'Fc_storage_obs', 'Flu',
                                                 'Fsd']), 
                                     output_structure='pandas')

diurnal_df = df.groupby([lambda x: x.hour, lambda y: y.minute]).mean()
diurnal_df['Fc_storage_obs_std'] = df['Fc_storage_obs'].groupby([lambda x: x.hour, 
                                                         lambda y: y.minute]).std()
diurnal_df.index = np.linspace(0, 23.5, 48)

var_names = ['0-0.5m', '0.5-2m', '2-4m', '4-8m', '8-16m', '16-36m', '0-36m', 'EC_36m']

# Create plot
fig = plt.figure(figsize = [13,8])
ax1 = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right = 0.75)
fig.patch.set_facecolor('white')
colour_idx = np.linspace(0, 1, 6)
ax2 = ax1.twinx()
ax3 = ax1.twinx()
offset = 80
new_fixed_axis = ax3.get_grid_helper().new_fixed_axis
ax3.axis["right"] = new_fixed_axis(loc="right",
                                    axes=ax3,
                                    offset=(offset, 0))
ax1.set_xlim([0, 24])
ax1.set_xticks([0,4,8,12,16,20,24])
ax1.tick_params(axis = 'x', labelsize = 50)
ax1.tick_params(axis = 'y', labelsize = 18)
ax2.tick_params(axis = 'y', labelsize = 14)
ax1.set_xlabel('$Time\/(hours)$')
ax1.set_ylabel('$S_c\/(\mu mol\/CO_2\/m^{-2}\/s^{-1})$')
ax1.axis['bottom'].label.set_fontsize(20)
ax1.axis['left'].label.set_fontsize(20)
ax1.xaxis.set_ticks_position('bottom')
ax2.set_ylabel('$u_{*}\/(m\/s^{-1})$')
ax2.axis['right'].label.set_fontsize(20)
ax3.set_ylabel('$Fsd\/(W\/m^{-2})$')
ax3.axis['right'].label.set_fontsize(20)
series_list = []
for i, var in enumerate(storage_vars[0:-2]):
    series_list.append(ax1.plot(diurnal_df.index, diurnal_df[var], 
                                color = plt.cm.cool(colour_idx[i]), 
                                label = var_names[i + 1]))
series_list.append(ax1.plot(diurnal_df.index, diurnal_df.Fc_storage_obs, 
                            color = '0.5', label = var_names[-2]))
series_list.append(ax1.plot(diurnal_df.index, diurnal_df.Fc_storage, 
                            color = '0.5', linestyle = '-.', lw = 2,
                            label = var_names[-1]))
series_list.append(ax2.plot(diurnal_df.index, diurnal_df.ustar, 
                            color = 'black', label = '$u_*$'))
series_list.append(ax3.plot(diurnal_df.index, diurnal_df.Fsd, 
                            color = 'blue', label = '$Fsd$'))                                
ax1.axhline(0, color = '0.5')
ax1.axvline(5, color = 'black', linestyle = ':')
ax1.axvline(20, color = 'black', linestyle = ':')
ax2.axhline(0.42, linestyle = '--', color = 'black')    
plt.setp(ax1.get_yticklabels()[0], visible = False)
plt.setp(ax2.get_yticklabels()[0], visible = False)
labs = [ser[0].get_label() for ser in series_list]
lst = [i[0] for i in series_list]
ax1.legend(lst, labs, fontsize = 14, loc = [0.06,0.74], 
           numpoints = 1, ncol = 2, frameon = False)
plt.tight_layout()
plt.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
            'basic C paper/Images/diurnal_storage.png',
#                bbox_inches='tight',
            dpi = 300) 
plt.show()