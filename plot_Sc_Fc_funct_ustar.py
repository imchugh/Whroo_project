#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import DataIO as io

num_cats = 30
correct_storage = False

# Make variable lists
use_vars = ['Fc_storage_obs', 'Fc_storage_obs_1', 'Fc_storage_obs_2', 'Fc_storage_obs_3', 
            'Fc_storage_obs_4', 'Fc_storage_obs_5', 'Fc_storage_obs_6', 
            'Fc', 'Fc_QCFlag', 'ustar', 'ustar_QCFlag', 'Fsd', 'Ta', 'Sws']

# Get data
file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L4.nc'

df = io.OzFluxQCnc_to_data_structure(file_in, var_list = use_vars, 
                                     output_structure = 'pandas')

# Remove daytime, missing or filled data where relevant
sub_df = df[use_vars]
sub_df = sub_df[sub_df.Fc_QCFlag == 0]
sub_df = sub_df[sub_df.ustar_QCFlag == 0]    
sub_df = sub_df[sub_df.Fsd < 5]
sub_df.drop(['Fc_QCFlag','ustar_QCFlag'], axis = 1, inplace = True)
sub_df.dropna(inplace = True)    

# Categorise data
sub_df['ustar_cat'] = pd.qcut(sub_df.ustar, num_cats, labels = np.linspace(1, num_cats, num_cats))
new_df = sub_df.groupby('ustar_cat').mean()

# Generate regression statistics and apply to low u* data for low levels 
# (0-0.5, 0.5-2, 2-4, 4-8) if correct_storage = True
if correct_storage:                 
    stats_df = pd.DataFrame(columns=use_vars[1:5], index = ['a', 'b'])
    for var in stats_df.columns:
        coeffs = np.polyfit(new_df['Fc_storage_obs_5'][(new_df.ustar < 0.4) & (new_df.ustar > 0.2)],
                            new_df[var][(new_df.ustar < 0.4) & (new_df.ustar > 0.2)],
                            1)
        stats_df.loc['a', var] = coeffs[0]
        stats_df.loc['b', var] = coeffs[1]
                     
    for var in stats_df.columns:
        new_df[var][new_df['ustar'] < 0.2] = (new_df['Fc_storage_obs_5'][new_df['ustar'] < 0.2] * 
                                              stats_df.loc['a', var] + stats_df.loc['b', var])
    
    new_df['Fc_storage_obs'] = new_df[use_vars[1:7]].sum(axis = 1)

# Create plot
fig = plt.figure(figsize = (12, 8))
fig.patch.set_facecolor('white')
ax1 = plt.gca()
ax2 = ax1.twinx()
ser_1 = ax1.plot(new_df.ustar, new_df.Fc, 's', label = '$F_{c}$',
                 markersize = 10, color = 'grey')
ser_2 = ax1.plot(new_df.ustar, new_df.Fc_storage_obs, 'o', label = '$S_{c}$', 
                 markersize = 10, markeredgecolor = 'black', 
                 markerfacecolor = 'none', mew = 1)
ser_3 = ax1.plot(new_df.ustar, new_df.Fc + new_df.Fc_storage_obs, '^', 
                 label = '$F_{c}\/+\/S_{c}$',
                 markersize = 10, color = 'black')
ser_4 = ax2.plot(new_df.ustar, new_df.Ta, color = 'black', linestyle = ':',
                 label = '$T_{a}$')
ser_5 = ax2.plot(new_df.ustar, new_df.Sws * 100, color = 'black', 
                 linestyle = '-.', label = '$VWC$')                     
ax1.axvline(x = 0.42, color = 'grey')
ax1.axvline(x = 0.32, color = 'grey')
ax1.axhline(y = 0, color  = 'black', linestyle = '-')
ax1.set_ylabel(r'$CO_2\/exchange\/(\mu mol\/m^{-2}\/s^{-1})$', fontsize = 22)
ax2.set_ylabel('$T_{a}\/(^{o}C)\//\/VWC\/(m^{3}\/m^{-3}\/$'+'x'+'$\/10^{2})$', 
               fontsize = 20)
ax2.set_ylim([-5,25])
ax1.set_xlabel('$u_{*}\/(m\/s^{-1})$', fontsize = 22)
ax1.tick_params(axis = 'x', labelsize = 14)
ax1.tick_params(axis = 'y', labelsize = 14)
ax2.tick_params(axis = 'y', labelsize = 14)
ax1.xaxis.set_ticks_position('bottom')
ax1.annotate('$u_{*th}\/(F_c)$', 
             xy = (0.42, 1), 
             xytext = (0.6, 
                       1),
             textcoords='data', verticalalignment='center',
             horizontalalignment = 'center',
             arrowprops = dict(arrowstyle="->"), fontsize = 18)
ax1.annotate('$u_{*th}\/(F_c\/+\/S_c)$', 
             xy = (0.32, 0.8), 
             xytext = (0.6, 
                       0.8),
             textcoords='data', verticalalignment='center',
             horizontalalignment = 'center',
             arrowprops = dict(arrowstyle="->"), fontsize = 18)
[plt.setp(ax.get_yticklabels()[0], visible = False) for ax in [ax1, ax2]]
all_ser = ser_1 + ser_2 + ser_3 + ser_4 + ser_5
labs = [ser.get_label() for ser in all_ser]
ax1.legend(all_ser, labs, fontsize = 18, loc = [0.75,0.28], numpoints = 1,
           frameon = False)
fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
            'basic C paper/Images/ustar_vs_Fc_and_storage.png',
            bbox_inches='tight',
            dpi = 300) 
plt.show()