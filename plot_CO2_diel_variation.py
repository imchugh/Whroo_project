# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

import utility_funcs as util
"""
Created on Mon Oct 10 16:24:41 2016

@author: imchugh
"""

# Set var lists    
vars_list = ['Cc_LI840_32m', 'Cc_LI840_16m', 'Cc_LI840_8m', 'Cc_LI840_4m',
             'Cc_LI840_2m', 'Cc_LI840_1m', 'Cc_7500_Av', 'ps', 'Ta', 'Fsd']
new_list = ['LI840_0.5m', 'LI840_2m', 'LI840_4m', 'LI840_8m', 'LI840_16m', 
            'LI840_36m']

# Get data
df = util.get_data(vars_list)

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
for i, var in enumerate(layer_names_list):
    work_df[var] = work_df[var] / weight_val_list[i]
work_df['LI7500_36m'] = df['LI7500_36m']
work_df.dropna(inplace = True)

# Create a diurnal average df
layer_names_list = layer_names_list + ['LI840_0-36m', 'LI7500_36m', 'Fsd']
diurnal_df = work_df[layer_names_list].loc['2012'].groupby([lambda x: x.hour, 
                                                lambda y: y.minute]).mean()
diurnal_df.index = np.arange(48) / 2.0

# Plot it
fig, ax = plt.subplots(1, 1, figsize = (12, 8))
fig.patch.set_facecolor('white')
colour_idx = np.linspace(0, 1, 6)
x_axis = diurnal_df.index
base_line = diurnal_df['LI840_0-36m'].mean()

align_dict = {'buildup': {'arrow_y': 395,
                          'text_x': 11,
                          'text_y': 403},
              'drawdown': {'arrow_y': 395,
                           'text_x': 19,
                           'text_y': 403}}
ax.set_xticks([0,4,8,12,16,20,24])
ax.set_xlim([0, 24])
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis = 'x', labelsize = 14)
ax.tick_params(axis = 'y', labelsize = 14)
ax.set_xlabel('$Time\/(hours)$', fontsize = 20)
ax.set_ylabel('$CO_2\/(ppm)$', fontsize = 20)
for i, var in enumerate(layer_names_list[:-3]):
    color = plt.cm.cool(colour_idx[i])
    this_series = diurnal_df[var] #- diurnal_df[var].mean() + base_line
    label = var.split('_')[-1]
    ax.plot(x_axis, this_series, color = color, linewidth = 0.5, label = label)
weight_series = diurnal_df['LI840_0-36m'] #- diurnal_df['LI840_0-36m'].mean() + base_line
ax.plot(x_axis, weight_series, label = '0-36m', color = '0.5',
        linewidth = 2)
EC_series = diurnal_df['LI7500_36m'] - diurnal_df['LI7500_36m'].mean() + base_line
ax.plot(x_axis, EC_series, label = 'EC_36m', color = 'black',
        linewidth = 2)
ax.axhline(base_line, color = '0.5', linewidth = 2)
ax.axvline(6, color = 'black', linestyle = ':')    
ax.axvline(18, color = 'black', linestyle = ':')
ax.legend(loc = [0.32, 0.82], frameon = False, ncol = 2)
ax.annotate('$CO_2\/drawdown$ \n $termination$ \n $(2100)$' , 
            xy = (21, align_dict['drawdown']['arrow_y']), 
            xytext = (align_dict['drawdown']['text_x'], 
                      align_dict['drawdown']['text_y']),
            textcoords='data', verticalalignment='center',
            horizontalalignment = 'center',
            arrowprops = dict(arrowstyle="->"), fontsize = 16)
ax.annotate('$CO_2\/buildup$ \n $termination$ \n $(0900)$' , 
            xy = (9.4, align_dict['buildup']['arrow_y']), 
            xytext = (align_dict['buildup']['text_x'], 
                      align_dict['buildup']['text_y']),
            textcoords='data', verticalalignment='center',
            horizontalalignment = 'center',
            arrowprops = dict(arrowstyle="->"), fontsize = 16)