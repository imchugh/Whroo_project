# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:42:40 2016

@author: imchugh
"""

import DataIO as io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2015_L3.nc'

df = io.OzFluxQCnc_to_data_structure(file_in, output_structure = 'pandas')

df['SRT'] = (df.Flu / (5.67*10**-8))**(1.0/4)-273.15

lst = ['Ta_HMP_1m', 'Ta_HMP_2m', 'Ta_HMP_4m', 'Ta_HMP_8m', 'Ta_HMP_16m', 
       'Ta_HMP_32m', 'ustar', 'SRT', 'Fsd', 'Ts']

sub_df = df.loc[:'2014-06-30', lst]
                                
num_cats = 20   

# Remove daytime, missing or filled data where relevant
sub_df = sub_df[sub_df.Fsd < 5]
sub_df.dropna(inplace = True)    

# Categorise data
sub_df['ustar_cat'] = pd.qcut(sub_df.ustar, num_cats, labels = np.linspace(1, num_cats, num_cats))
new_df = sub_df.groupby('ustar_cat').mean()

plot_list = [i for i in lst if not i in ['Fsd', 'ustar']]
adj = 5.4

#weight_df = pd.DataFrame(index=new_df.index)
#weight_df['0-2'] = new_df.Ta_HMP_2m * 2
#weight_df['2-4'] = (new_df.Ta_HMP_2m + new_df.Ta_HMP_4m) / 2 * 2
#weight_df['4-8'] = (new_df.Ta_HMP_4m + new_df.Ta_HMP_8m) / 2 * 4
#weight_df['8-16'] = (new_df.Ta_HMP_8m + new_df.Ta_HMP_16m) / 2 * 8
#weight_df['16-36'] = (new_df.Ta_HMP_16m + new_df.Ta_HMP_32m) / 2 * 20
#weight_df['all'] = weight_df.sum(axis = 1) / 36

daily_df = df.loc[:'2014-06-30', lst].groupby([lambda x: x.hour, lambda y: y.minute]).mean()
daily_df['time'] = np.arange(48.0) / 2
daily_df = pd.concat([daily_df.iloc[24:], daily_df.iloc[:24]])
daily_df = daily_df.reset_index()

colour_idx = np.linspace(0, 1, 6)
names = ['0.5m', '2m', '4m', '8m', '16m', '36m']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (18, 8))
fig.patch.set_facecolor('white')
for ax in [ax1, ax2]:
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 18)
    ax.tick_params(axis = 'y', labelsize = 18)
ax1.set_ylim([2,14])
ax1.set_yticks([4, 6, 8, 10, 12, 14])
ax1.set_yticklabels([4, 6, 8, 10, 12, 14])
ax1.set_xlabel('$u_*\/(m\/s^{-1})$', fontsize = 22)
ax1.set_ylabel('$Temperature\/(^oC)$', fontsize = 22)
ax2.set_ylim([8,24])
ax2.set_yticks([10, 12, 14, 16, 18, 20, 22, 24])
ax2.set_yticklabels([10, 12, 14, 16, 18, 20, 22, 24])
ax2.set_xlabel('$Time\/(hours)$', fontsize = 22)
ax2.set_ylabel('$Temperature\/(^oC)$', fontsize = 22)
ax2.set_xlim([0, 48])
ax2.set_xticks([0, 8, 16, 24, 32, 40, 48])
ax2.set_xticklabels([12, 16, 20, 0, 4, 8, 12], fontsize = 14)
for i, var in enumerate([var for var in plot_list if 'Ta_HMP' in var]):
    if var == 'Ta_HMP_1m':
        this_adj = adj
    else:
        this_adj = 0
    ax1.plot(new_df.ustar, new_df[var] + this_adj, 
             color = plt.cm.cool(colour_idx[i]), label = names[i])
    ax2.plot(daily_df.index, daily_df[var] + this_adj, 
             color = plt.cm.cool(colour_idx[i]))
ax1.plot(new_df.ustar, new_df['Ts'], color = 'blue', label = '-0.05m')
ax1.plot(new_df.ustar, new_df['SRT'], color = 'black', label = 'SRT')
ax2.plot(daily_df.index, daily_df['Ts'], color = 'blue')
ax2.plot(daily_df.index, daily_df['SRT'], color = 'black')
ax1.legend(loc = [0.44, 0.1], frameon = False, ncol = 2, fontsize = 18)
ax1.text(-0.1, 13.92, 'a)', horizontalalignment = 'center', fontsize = 18)
ax2.text(-5, 23.9, 'b)', horizontalalignment = 'center', fontsize = 18)

plt.show()

