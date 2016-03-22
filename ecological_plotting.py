# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:36:03 2016

@author: imchugh
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime_functions as dtf

import DataIO as io
import respiration_photosynthesis_run as rp_run 

def plot_NEE_cml():
    
    file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L6.nc'
    Fc_df = io.OzFluxQCnc_to_data_structure(file_in, var_list = ['Fc'],
                                            QC_accept_codes = [0], 
                                            output_structure = 'pandas')    
    anc_df = io.OzFluxQCnc_to_data_structure(file_in, var_list = ['ustar', 
                                                                  'Fsd',
                                                                  'NEE_SOLO'],
                                             output_structure = 'pandas')      
    
    df = Fc_df.join(anc_df)
    df['Fc_screen'] = df.Fc
    df['Fc_screen'][(df.ustar < 0.42) & (df.Fsd < 5)] = np.nan
    
    model_dict = rp_run.main()[0]
    df['NEE_est'] = model_dict['Re'] + model_dict['GPP']
    df['Fc_fill'] = df['Fc_screen']
    df['Fc_fill'][np.isnan(df['Fc_fill'])] = df['NEE_est']

    # Do daily sums
    daily_df = (df['Fc_fill'].groupby([lambda x: x.year, 
                    lambda y: y.dayofyear]).mean() * 0.0864 * 12)


    # Split into years and align
    years = list(set(daily_df.index.levels[0]))
    new_index = np.arange(1,367)
    yr_daily_df = pd.concat([daily_df.loc[yr].reindex(new_index) 
                             for yr in years], axis = 1)
    years_list = [str(yr) for yr in years]
    yr_daily_df.columns = years_list
    
    # Do cumulative NEE (drop 2011)
    cml_df = pd.concat([yr_daily_df[var].cumsum() for var in years_list], axis = 1)
    cml_df.drop('2011', axis = 1, inplace = True)

    fig, ax = plt.subplots(1, 1, figsize = (12, 8))
    fig.patch.set_facecolor('white')
    colors = ['0.5', 'black', 'black']
    styles = ['-', '--', ':']
    ax.set_xlim([0, 366])
    ax.set_xticks(dtf.get_DOY_first_day_of_month(2013))
    ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 
                        'J', 'A', 'S', 'O', 'N', 'D'], fontsize = 14)
    ax.set_xlabel('$Month$', fontsize = 18)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis = 'y', labelsize = 14)
    ax.set_ylabel('$Cumulative\/NEE\/(gC\/m^{-2})$', fontsize = 18)
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axhline(0, color = 'Black', linewidth = 0.5)
    for i, col in enumerate(cml_df.columns):
        ax.plot(cml_df.index, cml_df[col], label = col, color = colors[i], 
                linestyle = styles[i])
    ax.legend(frameon = False, loc = [0.57, 0.97], numpoints = 1, ncol = 3)
    fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
                'basic C paper/Images/Cumulative NEE.png',
                bbox_inches='tight',
                dpi = 300) 
    plt.show()    
    
    return