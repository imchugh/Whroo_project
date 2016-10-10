# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:03:18 2016

@author: imchugh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

target = '/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo basic C paper/Data/data.csv'

data_df = pd.read_csv(target)
data_df.drop('Unnamed: 0', inplace=True, axis = 1)

NEE_lst = [var for var in data_df.columns if 'NEE' in var]
sigdel_lst = [var for var in data_df.columns if 'sig' in var]
pairs_lst = [[NEE_lst[0], sigdel_lst[0]], 
             [NEE_lst[1], sigdel_lst[1]], 
             [NEE_lst[2], sigdel_lst[2]]]
myorder = [0, 2, 1]
pairs_lst = [pairs_lst[i] for i in myorder]


day_list = []
night_list = []

fig, ax1 = plt.subplots(1, 1, figsize = (12, 8))
fig.patch.set_facecolor('white')
ax1.xaxis.set_ticks_position('bottom')
ax1.set_ylim([0, 6])
ax1.yaxis.set_ticks_position('left')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.tick_params(axis = 'y', labelsize = 14)
ax1.tick_params(axis = 'x', labelsize = 14)

ax1.set_ylabel('$\sigma[\delta]\/(\mu mol\/CO_2\/m^{-2}\/s^{-1})$', fontsize = 18)
ax2 = ax1.twinx()
ax2.spines['right'].set_position('zero')
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_ylim(ax1.get_ylim())
ax2.tick_params(axis = 'y', labelsize = 14)
plt.setp(ax2.get_yticklabels()[0], visible = False)

markers = ['s', '^', 'h']
colors = ['0.5', '0', '1']
labels = ['$F_c$', '$F_c\/+\/S_c$', '$F_c\/+\/S_{c\_pt}$']

for i, pair in enumerate(pairs_lst):
    day_params = np.polyfit(data_df.loc[:10, pair[0]], 
                            data_df.loc[:10, pair[1]],
                            1)
    day_NEE_vals = np.append(data_df.loc[:9, pair[0]], 0)
    day_sigdel_vals = np.polyval(day_params, day_NEE_vals)
    night_params = np.polyfit(data_df.loc[10:, pair[0]], 
                              data_df.loc[10:, pair[1]],
                              1)
    night_NEE_vals = np.append(0, data_df.loc[10:, pair[0]])
    night_sigdel_vals = np.polyval(night_params, night_NEE_vals)
    ax1.plot(data_df[pair[0]], data_df[pair[1]], markers[i], 
             markersize = 10, markerfacecolor = colors[i], label = labels[i])
    if not i == 2:             
        ax1.plot(day_NEE_vals, day_sigdel_vals, color = colors[i])
        ax1.plot(night_NEE_vals, night_sigdel_vals, color = colors[i])
ax1.legend(loc = 'upper left', numpoints = 1, frameon = False, fontsize = 18)

fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
            'basic C paper/Images/random_error.png',
            bbox_inches='tight',
            dpi = 300) 