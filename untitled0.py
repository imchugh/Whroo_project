# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 13:42:40 2016

@author: imchugh
"""

import DataIO as io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#def plot_Ta():

"""
This script plots nocturnal Fc and Sc as a function of ustar; it also shows 
the major respiration drivers on the right hand axis 
"""

file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2015_L3.nc'

df = io.OzFluxQCnc_to_data_structure(file_in, output_structure = 'pandas')

df['SRT'] = (df.Flu / (5.67*10**-8))**(1.0/4)-273.15

ta_vars = [var for var in df.columns if 'Ta' in var]
ts_vars = [var for var in df.columns if 'Ts' in var]

lst = ['Ta_HMP_2m', 'Ta_HMP_4m', 'Ta_HMP_8m', 'Ta_HMP_16m', 
       'Ta_HMP_32m', 'ustar', 'Fsd']

sub_df = df.loc[:'2014-06-30', lst]
                                
num_cats = 20   

# Remove daytime, missing or filled data where relevant
sub_df.dropna(inplace = True)

diel_df = sub_df.groupby([lambda x: x.hour, lambda y: y.minute]).mean()

x = np.arange(48.0) / 2
y = np.array([2, 4, 8, 16, 36])

X, Y = np.meshgrid(x, y)

Z = np.array(diel_df.drop(['ustar', 'Fsd'], axis = 1)).T

levels = np.linspace(5.5,20,30)

fig, ax = plt.subplots(1, 1, figsize = (12, 8))
fig.patch.set_facecolor('white')

im = plt.imshow(Z, interpolation='bilinear', origin='lower',
                cmap=cm.gray, extent=(0, 23.5, 0, 36)) 
#ax.set_xlim([0, 24])
#ax.set_ylim([0, 36])
CS = ax.contour(X, Y, Z, levels = levels)
ax.clabel(CS, inline=1, fontsize=10)
