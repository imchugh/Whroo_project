# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:15:35 2016

@author: imchugh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb
import matplotlib


import DataIO as io


def get_data(var_list = None):
    
    reload(io)    
    
    file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L6.nc'
    
    return io.OzFluxQCnc_to_data_structure(file_in, var_list = var_list, 
                                           output_structure = 'pandas')



"""
This script plots CO2 mixing ratio as a function of time of day
"""    

# Set var lists    
vars_list = ['ps', 'Ta', 'Fsd', 'ustar',
             'Fc_storage_obs', 'Fc_storage']

# Get data
df = get_data(vars_list)

# Calculate weighted sum then remove weighting from individual layers            
diurnal_df = df.groupby([lambda x: x.hour, lambda y: y.minute]).mean()
diurnal_df.index = np.arange(48) / 2.0

diurnal_df = pd.concat([diurnal_df.iloc[24:], diurnal_df.iloc[:24]])
diurnal_df.index = np.linspace(0, 23.5, 48)
diurnal_df['Fsd'] = np.where(diurnal_df['Fsd'] < 0, 0, diurnal_df['Fsd'])

# Plot it
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 10))
fig.patch.set_facecolor('white')

# Subplot 1
x_axis = diurnal_df.index
ax1.set_xlim([0, 24])
ax1.set_ylim([10, 22])
xticks = ax1.xaxis.get_major_ticks()
[i.set_visible(False) for i in xticks]
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis = 'x', labelsize = 14)
ax1.tick_params(axis = 'y', labelsize = 14)
ax1.set_ylabel('$T_a\/(^oC)$', fontsize = 20)
ax1.plot(diurnal_df.index, diurnal_df.Ta, color = 'red', label = '$T_a$')

ax1_r = ax1.twinx()
ax1_r.set_xlim([0, 24])
ax1_r.set_ylim([0, 700])
ax1_r.spines['top'].set_visible(False)
ax1_r.spines['bottom'].set_visible(False)
ax1_r.tick_params(axis = 'y', labelsize = 14)
ax1_r.set_ylabel('$Fsd\/(W\/m^{-2})$', fontsize = 20)
ax1_r.plot(diurnal_df.index, diurnal_df.Fsd, color = 'blue', label = '$Fsd$')


# Subplot 2
ax2.set_xlim([0, 24])
ax2.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22,24])
ax2.set_xticklabels([12,14,16,18,20,22,0,2,4,6,8,10,12], fontsize = 14)
ax2.set_ylim([-2, 2])
ax2.set_yticks([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
ax2.tick_params(axis = 'x', labelsize = 14)
ax2.tick_params(axis = 'y', labelsize = 14)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.axhline(y = diurnal_df['Fc_storage_obs'].mean(), color = 'black')
ax2.set_xlabel('$Time\/(hours)$', fontsize = 20)
ax2.set_ylabel('$S_c\/(\mu mol\/CO_2\/m^{-2}\/s^{-1})$', fontsize = 20)
s1 = ax2.plot(x_axis, diurnal_df.Fc_storage_obs, color = 'black', label = '$S_c$')

ax2_r = ax2.twinx()
ax2_r.spines['top'].set_visible(False)
ax2_r.tick_params(axis = 'y', labelsize = 14)
ax2_r.set_ylabel('$u_*\/(m\/s^{-1})$', fontsize = 20)
s2 = ax2_r.plot(diurnal_df.index, diurnal_df.ustar, color = 'black', linestyle = ':', 
         label = '$u_*$')
ax2_r.plot([8, 8.5, 9, 9.5, 10, 10.5], [0.42, 0.42, 0.42, 0.42, 0.42, 0.42,],
           color = 'black')
ax2_r.text(11, 0.41, '$u_*\/threshold\/(F_c)$', fontsize = 18)

all_s = s1 + s2
labs = [s.get_label() for s in all_s]
ax2.legend(all_s, labs, loc = [0.75, 0.77], frameon = False, fontsize = 18)

#new_frame = pd.DataFrame({'a': np.tile(diurnal_df.Fc_storage_obs.mean(), 12), 
#                          'b': diurnal_df.Fc_storage_obs.iloc[7:19]}, 
#                         index = diurnal_df.index[7:19])
#this_x = new_frame.index
#this_y1 = new_frame['a']
#this_y2 = new_frame['b']
#ax2.fill_between(this_x, this_y1, this_y2, where=this_y1<=this_y2, 
#                 facecolor = 'grey', alpha = 0.3, edgecolor = 'None', interpolate = True)
#
#new_frame = pd.DataFrame({'a': np.tile(diurnal_df.Fc_storage_obs.mean(), 9), 
#                          'b': diurnal_df.Fc_storage_obs.iloc[34:43]}, 
#                         index = diurnal_df.index[34:43])
#add_frame = pd.DataFrame({'a': diurnal_df.Fc_storage_obs.mean(),
#                          'b': -1.56},
#                          index = [21.4])
#new_frame = new_frame.append(add_frame)
#this_x = new_frame.index
#this_y1 = new_frame['a']
#this_y2 = new_frame['b']
#ax2.fill_between(this_x, this_y1, this_y2, where=this_y1>=this_y2, 
#                 facecolor = 'grey', alpha = 0.3, 
#                 edgecolor = 'None', interpolate = True)
#                 
#
#
#transFigure = fig.transFigure.inverted()
#L1_coord1 = transFigure.transform(ax1.transData.transform([9, 395]))
#L1_coord2 = transFigure.transform(ax2.transData.transform([9, diurnal_df['Fc_storage_obs'].mean()]))
#line1 = matplotlib.lines.Line2D((L1_coord1[0], L1_coord2[0]),
#                                (L1_coord1[1], L1_coord2[1]),
#                                transform=fig.transFigure, color = 'black',
#                                linestyle = ':')
#L2_coord1 = transFigure.transform(ax1.transData.transform([21.4, 395]))
#L2_coord2 = transFigure.transform(ax2.transData.transform([21.4, -1.6]))
#line2 = matplotlib.lines.Line2D((L2_coord1[0], L2_coord2[0]),
#                                (L2_coord1[1], L2_coord2[1]),
#                                transform=fig.transFigure, color = 'black',
#                                linestyle = ':')
                                
#fig.lines = line1, line2
        