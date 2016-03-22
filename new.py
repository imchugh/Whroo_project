# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:58:39 2016

@author: imchugh
"""

import pandas as pd
import numpy as np
import datetime as dt
import pdb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import DataIO as io
import solar_functions as sf

file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L6.nc'

ustar_threshold = 0.42

data_d1 = io.OzFluxQCnc_to_data_structure(file_in, 
                                          var_list = ['Fc', 'ustar', 'Fsd', 
                                                      'Ta', 'ps'],
                                          QC_accept_codes=[0])
data_d2 = io.OzFluxQCnc_to_data_structure(file_in, var_list = ['Fc_storage'])

df = pd.DataFrame(data_d1)
df['Fc_storage'] = data_d2['Fc_storage']
df['time_since_sunset'] = 0

sunrise_d = sf.get_ephem_solar(data_d1, '-36.673215', '145.029247', 150, 10, 'rise')
sunset_d = sf.get_ephem_solar(data_d1, '-36.673215', '145.029247', 150, 10, 'set')

for i, this_date in enumerate(sunrise_d['date'][:-1]):
    set_date = dt.datetime.combine(this_date, sunset_d['set'][i])
    rise_date = dt.datetime.combine(this_date + dt.timedelta(1),
                                    sunrise_d['rise'][i + 1])
    this_index = np.where((data_d1['date_time'] > set_date) & 
                          (data_d1['date_time'] < rise_date))[0]
    count_array = np.linspace(1, len(this_index), len(this_index))
    df['time_since_sunset'].iloc[this_index] = count_array

df = df[(df.time_since_sunset > 0) & (df.ustar > 0.42)]

#df.sort('Ta')
df['Ta_cat'] = pd.qcut(df.Ta, 50, 
                       labels = np.linspace(1, 50, 50))

rslt_df = df.groupby('Ta_cat').mean()

#hi_ustar_df = df[df.ustar > ustar_threshold]
#hi_means_df = hi_ustar_df.groupby('time_since_sunset').mean()
#hi_count_df = hi_ustar_df.groupby('time_since_sunset').count()
#
#lo_ustar_df = df[df.ustar < ustar_threshold]
#lo_means_df = lo_ustar_df.groupby('time_since_sunset').mean()
#lo_count_df = lo_ustar_df.groupby('time_since_sunset').count()
#
#final_df = pd.DataFrame({'count': hi_means_df.index[1:-1],
#                         'hi_Sc_mean': hi_means_df.Fc_storage[1:-1], 
#                         'hi_Sc_count': hi_count_df.Fc_storage[1:-1],
#                         'hi_Ta_mean': hi_means_df.Ta[1:-1],
#                         'lo_Sc_mean': lo_means_df.Fc_storage[1:-1], 
#                         'lo_Sc_count': lo_count_df.Fc_storage[1:-1]})
#
#fig = plt.figure(figsize = (12, 9))
#fig.patch.set_facecolor('white')
#gs = gridspec.GridSpec(2, 1, height_ratios=[3,1.5])
#ax1 = plt.subplot(gs[0])
#ax1.set_xlim([0.5,14])
#ax1.set_ylabel('$S_c\/(\mu mol C\/m^{-2} s^{-1})$', fontsize = 18)
#ax1.xaxis.set_ticks_position('bottom')
#ax1.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#ax1.set_xticklabels(['',2,'',4,'',6,'',8,'',10,'',12,'',14])
#ax1.tick_params(axis = 'x', labelsize = 14)
#ax1.tick_params(axis = 'y', labelsize = 14)
#x = final_df.index / 2.0
#y1 = np.zeros(28)
#y2 = final_df.hi_Sc_mean
#y3 = final_df.hi_Ta_mean
#y4 = final_df.hi_Sc_count.cumsum() / (final_df.hi_Sc_count).sum() * 100
#ax1.plot(x, y2, color = 'black', marker = 's', markerfacecolor = 'black',
#         markeredgecolor = 'black')
#ax1.fill_between(x, y1, y2, where=y2>=y1, facecolor='0.4', edgecolor='None',
#                 interpolate=True)
#ax1.fill_between(x, y1, y2, where=y2<y1, facecolor='0.8', edgecolor='None',
#                 interpolate=True)
#ax1.axhline(0, color = 'black')
#ax2 = ax1.twinx()
#ax2.set_xlim([0.5,14])
#ax2.set_ylabel('$Ta\/(^{o}C)$', fontsize = 18)
#ax2.tick_params(axis = 'y', labelsize = 14)
#ax2.plot(x, y3, color = 'black', linestyle=':')
#ax3 = plt.subplot(gs[1])
#ax3.set_ylabel('$\%\/ obs$', fontsize = 18)                 
#ax3.set_xlim([0.5, 14])
#ax3.xaxis.set_ticks_position('bottom')
#ax3.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#ax3.set_xticklabels(['',2,'',4,'',6,'',8,'',10,'',12,'',14])
#ax3.yaxis.set_ticks_position('left')
#ax3.set_xlabel('$Time\/after\/sunset\/(hrs)$', fontsize = 18)
#ax3.tick_params(axis = 'x', labelsize = 14)
#ax3.tick_params(axis = 'y', labelsize = 14)
#ax3.plot(x, y4, color = 'black')