# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:42:40 2016

@author: imchugh
"""

import DataIO as io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#def plot_Ta():

"""
This script plots nocturnal Fc and Sc as a function of ustar; it also shows 
the major respiration drivers on the right hand axis 
"""

file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2015_L3.nc'

df = io.OzFluxQCnc_to_data_structure(file_in, output_structure = 'pandas')

ta_vars = [var for var in df.columns if 'Ta' in var]
ts_vars = [var for var in df.columns if 'Ts' in var]

lst = ['Ta_HMP_2m', 'Ta_HMP_4m', 'Ta_HMP_8m', 'Ta_HMP_16m', 
       'Ta_HMP_32m', 'ustar', 'Ts', 'Fsd', 'Flu']

sub_df = df.loc[:'2014-06-30', lst]
                                
num_cats = 30   

# Remove daytime, missing or filled data where relevant
sub_df = sub_df[sub_df.Fsd < 5]
sub_df.dropna(inplace = True)    

# Categorise data
sub_df['ustar_cat'] = pd.qcut(sub_df.ustar, num_cats, labels = np.linspace(1, num_cats, num_cats))
new_df = sub_df.groupby('ustar_cat').mean()

# Plot it
fig, ax = plt.subplots(1, 1, figsize = (12, 8))
fig.patch.set_facecolor('white')

new_df.Flu = (new_df.Flu / (5.67*10**-8))**(1.0/4)-273.15

plot_list = [i for i in lst if not i in ['Fsd', 'ustar']]

adj = 1.5

weight_df = pd.DataFrame(index=new_df.index)
weight_df['0-2'] = new_df.Ta_HMP_2m * 2
weight_df['2-4'] = (new_df.Ta_HMP_2m + new_df.Ta_HMP_4m) / 2 * 2
weight_df['4-8'] = (new_df.Ta_HMP_4m + new_df.Ta_HMP_8m) / 2 * 4
weight_df['8-16'] = (new_df.Ta_HMP_8m + new_df.Ta_HMP_16m) / 2 * 8
#weight_df['16-36'] = (new_df.Ta_HMP_16m + new_df.Ta_HMP_32m) / 2 * 20
weight_df['all'] = weight_df.sum(axis = 1) / 16

new_df['Ta_weighted'] = weight_df['all']
plot_list.append('Ta_weighted')
plot_list.append('Ts')

for var in plot_list:
    if var == 'Flu':
        new_df[var] = new_df[var] + adj
    ax.plot(new_df.ustar, new_df[var], label = var)

ax.plot(new_df.ustar, new_df.Flu, label = 'Flu')

plt.legend(loc='lower right')
plt.show()

