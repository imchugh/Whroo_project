#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 14:58:20 2016

@author: imchugh
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy as cp

import DataIO as io

fc_ustar = 0.42
fc_sc_ustar = 0.31

file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L6.nc'
    
df = io.OzFluxQCnc_to_data_structure(file_in, 
                                     var_list = ['ustar','Ta', 'Fc',
                                                 'Fc_storage_obs', 'Flu',
                                                 'Fsd'], 
                                     output_structure='pandas')

df.Fsd = df.Fsd * 4.6 * 0.46
df['Fc_filtered'] = cp.deepcopy(df['Fc'])
df['Fc_filtered'][df['ustar'] < fc_ustar] = np.nan
df['Fc_Sc_filtered'] = cp.deepcopy(df['Fc'] + df['Fc_storage_obs'])
df['Fc_Sc_filtered'][df['ustar'] < fc_sc_ustar] = np.nan

diurnal_df = df.groupby([lambda x: x.hour, lambda y: y.minute]).mean()
diurnal_df.index = np.linspace(0, 23.5, 48)

# Create plot
fig = plt.figure(figsize = (12, 8))
fig.patch.set_facecolor('white')
ax1 = plt.gca()
ax1.set_xlim([8, 16])
ax1.set_ylim([-0.01,-0.002])
ax1.set_xticks([8,9,10,11,12,13,14,15,16])
ax1.tick_params(axis = 'x', labelsize = 14)
ax1.tick_params(axis = 'y', labelsize = 14)
ax1.set_xlabel('$Time (hours)$', fontsize = 22)
ax1.set_ylabel('$RUE\/(\mu mol\/CO_2\/\mu mol\/photon^{-1}\/m^{-2}\/s^{-1})$', fontsize = 22)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
series_list = []
ax1.plot(diurnal_df.index, diurnal_df.Fc / diurnal_df.Fsd, 
         color = 'black', lw = 2, label = '$F_c$')
ax1.plot(diurnal_df.index, diurnal_df.Fc_filtered / diurnal_df.Fsd, 
         color = 'grey', lw = 2, label = '$F_{c\_u_{*}corr}$')
ax1.plot(diurnal_df.index, (diurnal_df.Fc + 
         diurnal_df.Fc_storage_obs) / diurnal_df.Fsd, 
         color = 'black', linestyle = ':', lw = 2, label = '$F_c\/+\/S_c$')
ax1.plot(diurnal_df.index, diurnal_df.Fc_Sc_filtered / diurnal_df.Fsd, 
         color = 'black', linestyle = '-.', lw = 2,
         label = '$(F_c\/+\/S_c)_{u_{*}corr}$')

ax1.legend(fontsize = 18, loc = [0.75,0.77], 
           numpoints = 1, frameon = False)    
plt.setp(ax1.get_yticklabels()[0], visible = False)
plt.tight_layout()
fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
            'basic C paper/Images/RUE_and_Sc_on_Fc.png',
            bbox_inches='tight',
            dpi = 300) 
plt.show()