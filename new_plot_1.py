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
vars_list = ['Cc_LI840_32m', 'Cc_LI840_16m', 'Cc_LI840_8m', 'Cc_LI840_4m',
             'Cc_LI840_2m', 'Cc_LI840_1m', 'Cc_7500_Av', 'ps', 'Ta', 'Fsd',
             'Fc_storage_obs', 'Fc_storage']
new_list = ['LI840_0.5m', 'LI840_2m', 'LI840_4m', 'LI840_8m', 'LI840_16m', 
            'LI840_36m']

# Get data
df = get_data(vars_list)

# Convert LI7500 CO2 density to mixing ratio    
df['C_mol'] = df.Cc_7500_Av / (44*10**3)
df['air_mol'] = df.ps *10 ** 3 / (8.3143 * (273.15 + df.Ta))
df['LI7500_36m'] = df.C_mol / df.air_mol * 10**6

# Calculate layer mixing ratio averages, weights and make new names
layer_names_list = []
weight_val_list = []
work_df = pd.DataFrame(index = df.index)
total_weight = 0
for i, var in enumerate(vars_list[:len(new_list)]):
    prev_var = False if i == 0 else var        
    prev_val = 0 if i == 0 else val
    new_var = new_list[i]
    a = new_var.split('_')[-1]
    try:
        val = int(a.split('m')[0])
    except:
        val = float(a.split('m')[0])
    try:
        weight_val = val - prev_val
    except:
        pdb.set_trace()
    weight_val_list.append(weight_val)
    total_weight = total_weight + weight_val
    new_name = 'LI840_' + str(prev_val) + '-' + str(val) + 'm'
    layer_names_list.append(new_name)
    if prev_var:
        work_df[new_name] = ((df[var] + df[prev_var]) / 2) * weight_val
    else:
        work_df[new_name] = df[var] * weight_val

# Calculate weighted sum then remove weighting from individual layers            
work_df['LI840_0-36m'] = work_df.sum(axis = 1) / total_weight
work_df['Fsd'] = df['Fsd']
work_df['Fc_storage_calc'] = df['Fc_storage']
work_df['Fc_storage_obs'] = df['Fc_storage_obs']
for i, var in enumerate(layer_names_list):
    work_df[var] = work_df[var] / weight_val_list[i]
work_df['LI7500_36m'] = df['LI7500_36m']
work_df.dropna(inplace = True)

# Create a diurnal average df
layer_names_list = layer_names_list + ['LI840_0-36m', 'LI7500_36m', 'Fsd', 
                                       'Fc_storage_obs', 'Fc_storage_calc']
diurnal_df = work_df[layer_names_list].groupby([lambda x: x.hour, 
                                                lambda y: y.minute]).mean()
diurnal_df.index = np.arange(48) / 2.0

diurnal_df = pd.concat([diurnal_df.iloc[24:], diurnal_df.iloc[:24]])
diurnal_df.index = np.linspace(0, 23.5, 48)

# Plot it
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (12, 10))
fig.patch.set_facecolor('white')

# Subplot 1
x_axis = diurnal_df.index
prof_mean = diurnal_df['LI840_0-36m'].mean()
EC_mean = diurnal_df['LI7500_36m'].mean()
base_line = 395
ax1.set_xlim([0, 24])
ax1.tick_params(axis = 'y', labelsize = 14)
ax1.yaxis.set_ticks_position('left')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
xticks = ax1.xaxis.get_major_ticks()
[i.set_visible(False) for i in xticks]
ax1.set_ylim([386, 404])
ax1.set_yticks([384, 388, 392, 396, 400, 404, 408])
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.tick_params(axis = 'y', labelsize = 14)
ax1.set_ylabel('$CO_2\/(ppm)$', fontsize = 20)
weight_series = diurnal_df['LI840_0-36m'] + base_line - prof_mean
ax1.plot(x_axis, weight_series, label = '$Profile$', color = 'black')
EC_series = diurnal_df['LI7500_36m'] + base_line - EC_mean
ax1.plot(x_axis, EC_series, label = '$EC$', color = '0.5')
ax1.axhline(base_line, color = 'black')
ax1.legend(loc = [0.06, 0.82], frameon = False, fontsize = 18)

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
ax2.plot(x_axis, diurnal_df.Fc_storage_obs, color = 'black', label = '$Profile$')
ax2.plot(x_axis, diurnal_df['Fc_storage_calc'], color = '0.5', label = '$EC$')
#ax2.legend(loc = [0.06, 0.82], frameon = False, fontsize = 18)

new_frame = pd.DataFrame({'a': np.tile(diurnal_df.Fc_storage_obs.mean(), 12), 
                          'b': diurnal_df.Fc_storage_obs.iloc[7:19]}, 
                         index = diurnal_df.index[7:19])
this_x = new_frame.index
this_y1 = new_frame['a']
this_y2 = new_frame['b']
ax2.fill_between(this_x, this_y1, this_y2, where=this_y1<=this_y2, 
                 facecolor = 'grey', alpha = 0.3, edgecolor = 'None', interpolate = True)

new_frame = pd.DataFrame({'a': np.tile(diurnal_df.Fc_storage_obs.mean(), 9), 
                          'b': diurnal_df.Fc_storage_obs.iloc[34:43]}, 
                         index = diurnal_df.index[34:43])
add_frame = pd.DataFrame({'a': diurnal_df.Fc_storage_obs.mean(),
                          'b': -1.56},
                          index = [21.4])
new_frame = new_frame.append(add_frame)
this_x = new_frame.index
this_y1 = new_frame['a']
this_y2 = new_frame['b']
ax2.fill_between(this_x, this_y1, this_y2, where=this_y1>=this_y2, 
                 facecolor = 'grey', alpha = 0.3, 
                 edgecolor = 'None', interpolate = True)
                 


transFigure = fig.transFigure.inverted()
L1_coord1 = transFigure.transform(ax1.transData.transform([9, 395]))
L1_coord2 = transFigure.transform(ax2.transData.transform([9, diurnal_df['Fc_storage_obs'].mean()]))
line1 = matplotlib.lines.Line2D((L1_coord1[0], L1_coord2[0]),
                                (L1_coord1[1], L1_coord2[1]),
                                transform=fig.transFigure, color = 'black',
                                linestyle = ':')
L2_coord1 = transFigure.transform(ax1.transData.transform([21.4, 395]))
L2_coord2 = transFigure.transform(ax2.transData.transform([21.4, -1.6]))
line2 = matplotlib.lines.Line2D((L2_coord1[0], L2_coord2[0]),
                                (L2_coord1[1], L2_coord2[1]),
                                transform=fig.transFigure, color = 'black',
                                linestyle = ':')
                                
fig.lines = line1, line2
        