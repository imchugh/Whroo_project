# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 03:27:01 2016

@author: imchugh
"""
import pandas as pd
import numpy as np
import copy as cp
import datetime as dt
import pdb

import DataIO as io
import solar_functions as sf

add_hours = 3

file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L6.nc'

ustar_threshold = 0.42

data_d1 = io.OzFluxQCnc_to_data_structure(file_in, 
                                          var_list = ['Fc', 'ustar', 'Fsd', 
                                                      'Ta', 'ps'],
                                          QC_accept_codes=[0])
data_d2 = io.OzFluxQCnc_to_data_structure(file_in, var_list = ['Fc_storage'])

df = pd.DataFrame(data_d1)
df['Fc_storage'] = data_d2['Fc_storage']

sunrise_d = sf.get_ephem_solar(data_d1, '-36.673215', '145.029247', 150, 10, 'rise')
sunset_d = sf.get_ephem_solar(data_d1, '-36.673215', '145.029247', 150, 10, 'set')

sunset_d['set_shift'] = [dt.time(this_time.hour + add_hours, this_time.minute, 
                                 this_time.second) 
                         for this_time in sunset_d['set']]

dates_array = np.array([this_dt.date() for this_dt in data_d1['date_time']])
rslt_list = []
for i, date in enumerate(sunrise_d['date']):
    rise_time = sunrise_d['rise'][i]
    set_time = sunset_d['set_shift'][i]
    this_index = np.where(dates_array == date)
    these_times = np.array([this_date.time() for 
                            this_date in data_d1['date_time'][this_index]])
    day_index = np.where((these_times < rise_time) | (these_times > set_time), 1, 0)
    rslt_list.append(day_index)
df['night_index'] = np.concatenate(rslt_list)

ustar_index = np.where((df.ustar > ustar_threshold) & (df.night_index == 1))[0]

rslt_df = pd.DataFrame(index = ustar_index, 
                       columns = ['count', 'Sc', 'Fc', 
                                  'ustar', 'ustar_diff', 
                                  'Ta'],
                       dtype = float)

for i in ustar_index:

    this_ustar = 0
    j = 0

    while True:
        j = j + 1
        if i - j < 0:
            j = j - 1
            break
        if not df.loc[i - j, 'night_index']:
            j = j - 1
            break
        this_ustar = df.loc[i - j, 'ustar']
        if this_ustar > ustar_threshold:
            j = j - 1
            break
        this_Fc = df.loc[i - j, 'Fc']
        this_Sc = df.loc[i - j, 'Fc_storage']
        if np.isnan(this_Fc) or np.isnan(this_Sc) or np.isnan(this_ustar):
            j = j - 1
            break

    rslt_df.loc[i, 'count'] = j
    if j:        
        rslt_df.loc[i, 'ustar'] = df.loc[i, 'ustar']
        rslt_df.loc[i, 'Fc'] = df.loc[i, 'Fc']
        rslt_df.loc[i, 'Sc'] = df.loc[i, 'Fc_storage']
        rslt_df.loc[i, 'Ta'] = df.loc[i, 'Ta']
        rslt_df.loc[i, 'ustar_diff'] = (df.loc[i, 'ustar'] - 
                                        df.loc[i - j: i - 1, 'ustar'].mean())

one_rslt_df = rslt_df[rslt_df['count'] == 1]
one_rslt_df['ustar_diff_cat'] = pd.qcut(one_rslt_df.ustar_diff, 10, 
                                        labels = np.linspace(1, 10, 10))
                                        
                                        
#test = one_rslt_df.groupby('ustar_diff_cat').mean()
                                        
        
#
#rslt_df.dropna(inplace = True)
#
#lst = []
#for i in range(1, int(rslt_df['count'].max()) + 1):
#    temp_array = np.array(rslt_df[rslt_df['count'] == i].index)
#    lst.append(df.loc[temp_array, 'Fc_storage'].mean())
#    
#this_array = np.array(lst)
    
    