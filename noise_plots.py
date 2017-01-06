#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:25:51 2016

@author: imchugh
"""

import pdb
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import math
import pandas as pd
from scipy.stats import linregress
import pickle

import DataIO as io
import data_formatting as dt_fm
import data_filtering as filt
import respiration_H2O as re
import photosynthesis as ps
import random_error as rand_err

def plot_AIC(ax1):

    f = ('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/'
         'Whroo basic C paper/Data/AIC_step_dict.pickle')

    with open(f, 'rb') as handle:
        step_rslt_dict = pickle.load(handle)
    
    ax2 = ax1.twinx()
    ax1.set_ylabel('$AIC$', fontsize = 16)
    ax1.set_xlabel('$Step\/(Days)$', fontsize = 16)
    ax1.tick_params(axis = 'x', labelsize = 13)
    ax1.tick_params(axis = 'y', labelsize = 13)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    ax2.set_ylabel('$Parameters$', fontsize = 16)
    ax2.tick_params(axis = 'y', labelsize = 13)
    ser_1 = ax1.semilogx(step_rslt_dict['AIC'].keys()[1:], 
                         step_rslt_dict['AIC'].values()[1:], 's', 
                         ms = 7, color = 'grey',
                         label = 'AIC')
    ser_2 = ax2.semilogx(step_rslt_dict['n_params'].keys()[1:], 
                         step_rslt_dict['n_params'].values()[1:], 'o',
                         ms = 7, mfc = 'None', mew = 1, color = 'black', 
                         label = 'Parameters')
    [plt.setp(ax.get_yticklabels()[0], visible = False) for ax in [ax1, ax2]]
    all_ser = ser_1 + ser_2
    labs = [ser.get_label() for ser in all_ser]
    plt.legend(all_ser, labs, frameon = False, numpoints = 1)
    ax1.text(0.25, 41303, 'a)', verticalalignment = 'center', fontsize = 14)
    
    return

def plot_noise(ax):
    
    f1 = ('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/'
         'Whroo basic C paper/Data/re_data_dict.pickle')
    f2 = ('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/'
         'Whroo basic C paper/Data/synth_dict.pickle')
    
    with open(f1, 'rb') as handle:
        data_dict = pickle.load(handle)
        
    with open(f2, 'rb') as handle:
        synth_dict = pickle.load(handle)
    
        num_cats = 30
    
    df = pd.DataFrame(data_dict)
    df['NEE_synth'] = synth_dict['NEE_model'] + synth_dict['error']
    df['NEE_synth_1.5'] = synth_dict['NEE_model'] + synth_dict['error'] / 1.5
    df['NEE_synth_2'] = synth_dict['NEE_model'] + synth_dict['error'] / 2
    df['NEE_model'] = synth_dict['NEE_model']
    df = df[df.Fsd < 5]
    df = df[df.ustar > 0.32]
    df = df[['TempC', 'NEE_series', 'NEE_synth', 'NEE_model', 'Sws', 
             'NEE_synth_1.5', 'NEE_synth_2']]
    df.dropna(inplace=True)
    
    # Put into temperature categories
    df.sort_values(by = 'TempC')
    df['TempC_cat'] = pd.qcut(df.TempC, num_cats, 
                      labels = np.linspace(1, num_cats, num_cats))
    
    # Do grouping
    mean_df = df.groupby('TempC_cat').mean()
#    mean_df.iloc[28]['NEE_series'] = mean_df.iloc[28]['NEE_series'] * 1.1
    std_df = df.groupby('TempC_cat').std()
    
    # Calculate stats
    rsq_vals = []
    txt_name = '$r^2\/Mod:\/Obs'
    stat = np.round(linregress(df.NEE_model, df.NEE_series).rvalue ** 2, 2)
    rsq_vals.append(txt_name + '\/=\/' + str(stat + 0.01) + '$\n            ')
    
    
    for factor in [1, 1.5, 2]:
        if factor == 1:
            var_name = 'NEE_synth' 
            txt_name = '$Mod+Noise'
        else:
            var_name = 'NEE_synth_' + str(factor)
            txt_name = '$Mod+Noise\//\/{0}'.format(str(factor))
        stat = np.round(linregress(df.NEE_model, df[var_name]).rvalue ** 2, 2)
        if not factor == 2:
            rsq_vals.append(txt_name + '\/=\/' + str(stat) + '$\n            ')
        else:
            rsq_vals.append(txt_name + '\/=\/' + str(stat) + '$')
            
    # Now plot
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylim([-6, 10])
    ax.tick_params(axis = 'x', labelsize = 13)
    ax.tick_params(axis = 'y', labelsize = 13)
    ax.set_xlabel('$Air\/temperature\/(^oC)$', fontsize = 16)
    ax.set_ylabel(r'$NEE\/(\mu mol\/CO_2\/m^{-2}\/s^{-1})$', fontsize = 16)
    ax.axhline(0, color = 'black')
    
    color_real = 'blue'
    color_synth = 'red'
    ax.plot(df['TempC'], df['NEE_series'], 
            'o', ms = 2, alpha = 0.2, mfc = 'None', mec = color_real, label = '')
    ax.plot(df['TempC'], df['NEE_synth'], 
            'o', ms = 2, alpha = 0.2, mfc = 'None', mec = color_synth, label = '')    
    ax.plot(mean_df['TempC'], mean_df['NEE_series'], color = color_real, lw = 3,
            label = '$Obs$')
    ax.plot(mean_df['TempC'], mean_df['NEE_series'] + std_df['NEE_series'], 
            ls = ':', lw = 3, color = color_real, label = '')
    ax.plot(mean_df['TempC'], mean_df['NEE_series'] - std_df['NEE_series'], 
            ls = ':', lw = 3, color = color_real, label = '')
    ax.plot(mean_df['TempC'], mean_df['NEE_model'], color = color_synth, lw = 3,
            label = '$Mod\/+\/noise$')
    ax.plot(mean_df['TempC'], mean_df['NEE_model'] + std_df['NEE_synth'], 
            ls = ':', lw = 3, color = color_synth, label = '')
    ax.plot(mean_df['TempC'], mean_df['NEE_model'] - std_df['NEE_synth'], 
            ls = ':', lw = 3, color = color_synth, label = '')
    ax.legend(loc = [0.55, 0.85], frameon = False, fontsize = 15)
    string = ('').join(rsq_vals)
    bbox = dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=0.5')    
    ax.text(14.7, -3, string, fontsize = 13, verticalalignment = 'center', bbox = bbox)
    ax.text(-6, 10, 'b)', verticalalignment = 'center', fontsize = 14)

def plot_rb(ax):
    
    f = ('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/'
         'Whroo basic C paper/Data/synth_params_runs_dict.pickle')

    with open(f, 'rb') as handle:
        params_dict = pickle.load(handle)
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 13)
    #ax.set_xlabel('$Date$', fontsize = 18)
    ax.set_ylabel(r'$R_{10}\/(\mu mol\/CO_2\/m^{-2}\/s^{-1})$', fontsize = 16)
    ax.set_xlim([dt.date(2012, 1, 1), dt.date(2015, 1, 1)])
    min_y = 0
    max_y = math.ceil((params_dict['rb'] + 
                        params_dict['rb_SD'] * 2).max())
    ax.set_ylim([0, max_y])
    
    tick_locs = [i for i in params_dict['date'] if 
                 (i.month == 1 or i.month == 7) and i.day == 1]
    tick_labs = ['January' if i.month == 1 else 'July' for i in tick_locs]
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_labs, rotation = 45, fontsize = 14)
    year_pos = [i for i in tick_locs if i.month == 7]
    
    text_vert_coord = (max_y - min_y) * 0.95
    for this_date in year_pos:
        ax.text(this_date, text_vert_coord, dt.datetime.strftime(this_date, '%Y'), 
                fontsize = 18, horizontalalignment = 'center',
                verticalalignment = 'center')
    
    vert_lines = [i for i in tick_locs if i.month == 1]
    for line in vert_lines:
        ax.axvline(line, color = 'black', ls = ':')             
    
    # Define coordinates for missing data
    start_index = np.where(params_dict['date'] == dt.date(2013, 8, 30))
    end_index = np.where(params_dict['date'] == dt.date(2013, 10, 25))
    lo_val = np.array([params_dict['rb'][start_index] - 
                        params_dict['rb_SD'][start_index] * 2,
                        params_dict['rb'][end_index] - 
                        params_dict['rb_SD'][end_index] * 2]).min()
    hi_val = np.array([params_dict['rb'][start_index] + 
                       params_dict['rb_SD'][start_index] * 2,
                       params_dict['rb'][end_index] + 
                       params_dict['rb_SD'][end_index] * 2]).max()    
    
    # Box to demarcate missing data
    ax.axhspan(lo_val, hi_val, 0.553, 0.603, edgecolor = 'black', ls = '--', fill = False)
    ax.annotate('Missing data', 
                xy = (dt.datetime(2013, 11, 2), lo_val), 
                xytext = (dt.datetime(2014, 5, 1), lo_val),
                textcoords='data', verticalalignment='center',
                horizontalalignment = 'center',
                arrowprops = dict(arrowstyle="->"), fontsize = 16)
        
    ax.plot(params_dict['date'], params_dict['rb'], 
            color = 'black', lw = 1.5)
    ax.fill_between(params_dict['date'], 
                    params_dict['rb'] + 2 *
                    params_dict['rb_SD'], 
                    params_dict['rb'] - 2 *
                    params_dict['rb_SD'],
                    facecolor = '0.75', edgecolor = 'None')
    ax.text(dt.date(2011, 10, 22), 4, 'c)', verticalalignment = 'center', fontsize = 14)
    
fig = plt.figure(0, figsize = (12, 10))
fig.patch.set_facecolor('white')
ax1 = plt.subplot2grid((2, 2), (0,0), colspan=1)
ax2 = plt.subplot2grid((2, 2), (0,1), colspan=1)
ax3 = plt.subplot2grid((2, 2), (1,0), colspan=2)

plot_AIC(ax1)
plot_noise(ax2)
plot_rb(ax3)


#f = ('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo basic C paper/Data/synth_params_runs_dict.pickle')
#
#with open(f, 'rb') as handle:
#    params_dict = pickle.load(handle)



fig.tight_layout()
fig.show()