# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:28:46 2015

@author: imchugh
"""

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pdb

import respiration_photosynthesis_run as rp_run
import DataIO as io
import datetime_functions as dtf
import solar_functions as sf

reload (dtf)

def get_data(var_list = None):
    
    reload(io)    
    
    file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L6.nc'
    
    return io.OzFluxQCnc_to_data_structure(file_in, var_list = var_list, 
                                           output_structure = 'pandas')

def plot_CO2_diurnal():

    """
    This script plots CO2 mixing ratio as a function of time of day
    """    

    # Set var lists    
    vars_list = ['Cc_LI840_32m', 'Cc_LI840_16m', 'Cc_LI840_8m', 'Cc_LI840_4m',
                 'Cc_LI840_2m', 'Cc_LI840_1m', 'Cc_7500_Av', 'ps', 'Ta', 'Fsd']
    new_list = ['LI840_0.5m', 'LI840_2m', 'LI840_4m', 'LI840_8m', 'LI840_16m', 
                'LI840_36m']

    # Get data
    df = get_data(vars_list)

    # Convert LI7500 CO2 density to mixing ratio    
    df['C_mol'] = df.Cc_7500_Av / (44*10**3)
    df['air_mol'] = df.ps *10 ** 3 / (8.3143 * (273.15 + df.Ta))
    df['LI7500_36m'] = df.C_mol / df.air_mol * 10**6

    # Calculate layer mixing ratio averages, weights and make new names
    layer_names_list = []
    weight_val_list = []
    work_df = pd.DataFrame(index = df.index)
    total_weight = 0
    for i, var in enumerate(vars_list[:len(new_list)]):
        prev_var = False if i == 0 else var        
        prev_val = 0 if i == 0 else var
        new_var = new_list[i]
        a = new_var.split('_')[-1]
        try:
            val = int(a.split('m')[0])
        except:
            val = float(a.split('m')[0])
        weight_val = val - prev_val
        weight_val_list.append(weight_val)
        total_weight = total_weight + weight_val
        new_name = 'LI840_' + str(prev_val) + '-' + str(val) + 'm'
        layer_names_list.append(new_name)
        if prev_var:
            work_df[new_name] = ((df[var] + df[prev_var]) / 2) * weight_val
        else:
            work_df[new_name] = df[var] * weight_val

    # Calculate weighted sum then remove weighting from individual layers            
    work_df['LI840_0-36m'] = work_df.sum(axis = 1) / total_weight
    work_df['Fsd'] = df['Fsd']
    for i, var in enumerate(layer_names_list):
        work_df[var] = work_df[var] / weight_val_list[i]
    work_df['LI7500_36m'] = df['LI7500_36m']
    work_df.dropna(inplace = True)

    # Create a diurnal average df
    layer_names_list = layer_names_list + ['LI840_0-36m', 'LI7500_36m', 'Fsd']
    diurnal_df = work_df[layer_names_list].loc['2012'].groupby([lambda x: x.hour, 
                                                    lambda y: y.minute]).mean()
    diurnal_df.index = np.arange(48) / 2.0

    # Plot it
    fig, ax = plt.subplots(1, 1, figsize = (12, 8))
    fig.patch.set_facecolor('white')
    colour_idx = np.linspace(0, 1, 6)
    x_axis = diurnal_df.index
    base_line = diurnal_df['LI840_0-36m'].mean()
    print base_line
    align_dict = {'buildup': {'arrow_y': 395,
                              'text_x': 11,
                              'text_y': 403},
                  'drawdown': {'arrow_y': 395,
                               'text_x': 19,
                               'text_y': 403}}
    ax.set_xticks([0,4,8,12,16,20,24])
    ax.set_xlim([0, 24])
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    ax.set_xlabel('$Time\/(hours)$', fontsize = 20)
    ax.set_ylabel('$CO_2\/(ppm)$', fontsize = 20)
    for i, var in enumerate(layer_names_list[:-3]):
        color = plt.cm.cool(colour_idx[i])
        this_series = diurnal_df[var] #- diurnal_df[var].mean() + base_line
        label = var.split('_')[-1]
        ax.plot(x_axis, this_series, color = color, linewidth = 0.5, label = label)
    weight_series = diurnal_df['LI840_0-36m'] #- diurnal_df['LI840_0-36m'].mean() + base_line
    ax.plot(x_axis, weight_series, label = '0-36m', color = '0.5',
            linewidth = 2)
    EC_series = diurnal_df['LI7500_36m'] - diurnal_df['LI7500_36m'].mean() + base_line
    ax.plot(x_axis, EC_series, label = 'EC_36m', color = 'black',
            linewidth = 2)
    ax.axhline(base_line, color = '0.5', linewidth = 2)
    ax.axvline(6, color = 'black', linestyle = ':')    
    ax.axvline(18, color = 'black', linestyle = ':')
    ax.legend(loc = [0.32, 0.82], frameon = False, ncol = 2)
    ax.annotate('$CO_2\/drawdown$ \n $termination$ \n $(2100)$' , 
                xy = (21, align_dict['drawdown']['arrow_y']), 
                xytext = (align_dict['drawdown']['text_x'], 
                          align_dict['drawdown']['text_y']),
                textcoords='data', verticalalignment='center',
                horizontalalignment = 'center',
                arrowprops = dict(arrowstyle="->"), fontsize = 16)
    ax.annotate('$CO_2\/buildup$ \n $termination$ \n $(0900)$' , 
                xy = (9.4, align_dict['buildup']['arrow_y']), 
                xytext = (align_dict['buildup']['text_x'], 
                          align_dict['buildup']['text_y']),
                textcoords='data', verticalalignment='center',
                horizontalalignment = 'center',
                arrowprops = dict(arrowstyle="->"), fontsize = 16)
                
    return

def plot_CO2_profiles_diurnal():
    
    # Set var lists    
    vars_list = ['Cc_LI840_32m', 'Cc_LI840_16m', 'Cc_LI840_8m', 'Cc_LI840_4m',
                 'Cc_LI840_2m', 'Cc_LI840_1m', 'Fsd']
    new_list = ['LI840_0.5m', 'LI840_2m', 'LI840_4m', 'LI840_8m', 'LI840_16m', 
                'LI840_36m', 'Fsd']
    # Get data
    df = get_data(vars_list)
    df.columns = new_list
    
    # Group
    diurnal_df = df.groupby([lambda x: x.hour, lambda y: y.minute]).mean()
    
    return diurnal_df

def plot_NEE_diurnal():
    
    reload(rp_run)    
    
    f = '/home/imchugh/Code/Python/Config_files/master_configs_2.txt'
    var_list = ['Fc', 'Fc_u*', 'Fc_Sc', 'Fc_Sc_u*']
    ustar_list = [0, 
                  {2011: 0.40,
                   2012: 0.39,
                   2013: 0.40,
                   2014: 0.42}, 
                  0, 
                  {2011: 0.31,
                   2012: 0.30,
                   2013: 0.32,
                   2014: 0.32}]
    stor_list = [False, False, True, True]

    # Get the uncorrected data and gap-fill Fc
    df = pd.DataFrame()
    for i, var in enumerate(var_list):
        
        temp_dict = rp_run.main(use_storage = stor_list[i],
                                ustar_threshold = ustar_list[i],
                                config_file = f,
                                do_light_response = True)[0]
        temp_dict['NEE_est'] = temp_dict['Re'] + temp_dict['GPP']
        temp_dict['NEE_filled'] = temp_dict['NEE_series']
        temp_dict['NEE_filled'][np.isnan(temp_dict['NEE_filled'])] = \
            temp_dict['NEE_est'][np.isnan(temp_dict['NEE_filled'])]
        df[var] = temp_dict['NEE_filled']
        if i == 0:
            df.index = temp_dict['date_time']
        
    # Do calculations of means for all groups
    diurnal_df = df.groupby([lambda x: x.hour, lambda y: y.minute]).mean()
    diurnal_df.index = np.linspace(0, 23.5, 48)

    # Set plot iterables
    vars_dict = {1: ['Fc', 'Fc_Sc'],
                 2: ['Fc_Sc', 'Fc_Sc_u*'],
                 3: ['Fc', 'Fc_Sc_u*'],
                 4: ['Fc_u*', 'Fc_Sc_u*']}
                 
    names_dict = {1: ['$F_c$', '$F_c\_S_c$'],
                  2: ['$F_c\_S_c$', '$F_c\_S_c\_u_*$'],
                  3: ['$F_c$', '$F_c\_S_c\_u_*$'],
                  4: ['$F_c\_u_*$', '$F_c\_S_c\_u_*$']}

    lines_dict = {'Fc': ':',
                  'Fc_u*': '-.',
                  'Fc_Sc': '--',
                  'Fc_Sc_u*': '-'}


    # Instantiate plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize = (16, 12))
    fig.patch.set_facecolor('white')
    fig_labels = ['a)', 'b)', 'c)', 'd)']

    for i, ax in enumerate((ax1, ax2, ax3, ax4)):

        counter = i + 1
        ax.set_xlim([0, 24])
        ax.set_ylim([-10, 4])
        ax.set_xticks([0,4,8,12,16,20,24])
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if counter > 2:        
            ax.set_xlabel(r'$Time\/(hours)$', fontsize = 18)
        if counter % 2 != 0:
            ax.set_ylabel(r'$NEE\/(\mu mol C\/m^{-2} s^{-1})$', fontsize = 18)
        x = diurnal_df.index
        var1 = vars_dict[counter][0]
        var2 = vars_dict[counter][1]
        y1 = diurnal_df[var1]
        y2 = diurnal_df[var2]
        ax.plot(x, y1, color = 'black', linestyle = lines_dict[var1], 
                linewidth = 2, label = names_dict[counter][0])
        ax.plot(x, y2, color = 'black', linestyle = lines_dict[var2], 
                linewidth = 2, label = names_dict[counter][1])
        ax.fill_between(x, y1, y2, where=y2>=y1, facecolor='0.8', 
                        edgecolor='None',interpolate=True)
        ax.fill_between(x, y1, y2, where=y1>=y2, facecolor='0.8', 
                        edgecolor='None',interpolate=True)

        ax.legend(fontsize = 18, loc = 'lower right', frameon = False)
        ax.axhline(y = 0, color = 'black', linestyle = '-')
        ax.text(-2.6, 3.8, fig_labels[i], fontsize = 12)

    fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
                'basic C paper/Images/diurnal_NEE_effects_of_correction.png',
                bbox_inches='tight',
                dpi = 300)     
    plt.show()

def plot_RUE_dependence_on_Sc():
    
    file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L6.nc'
        
    df = io.OzFluxQCnc_to_data_structure(file_in, 
                                         var_list = ['ustar','Ta', 'Fc',
                                                     'Fc_storage_obs', 'Flu',
                                                     'Fsd'], 
                                         output_structure='pandas')

    df.Fsd = df.Fsd * 4.6 * 0.46
    diurnal_df = df.groupby([lambda x: x.hour, lambda y: y.minute]).mean()
    diurnal_df.index = np.linspace(0, 23.5, 48)
    
    # Create plot
    fig = plt.figure(figsize = (12, 8))
    fig.patch.set_facecolor('white')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.set_xlim([8, 16])
    ax1.set_ylim([-0.01,-0.002])
    ax2.set_ylim([0,1])
    ax1.set_xticks([8,9,10,11,12,13,14,15,16])
    ax1.tick_params(axis = 'x', labelsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax2.tick_params(axis = 'y', labelsize = 14)
    ax1.set_xlabel('$Time (hours)$', fontsize = 22)
    ax1.set_ylabel('$RUE\/(\mu mol C\/\mu mol\/photon^{-1}\/m^{-2} s^{-1})$', fontsize = 22)
    ax2.set_ylabel('$S_c\//\/F_c$', fontsize = 22)
    series_list = []
    series_1 = ax1.plot(diurnal_df.index, (diurnal_df.Fc + 
                                           diurnal_df.Fc_storage_obs) / diurnal_df.Fsd, 
                        color = '0.5', label = '$S_c$')
    series_2 = ax1.plot(diurnal_df.index, diurnal_df.Fc / diurnal_df.Fsd, 
                        color = 'black', label = '$F_c\/+\/S_c$')
    series_3 = ax2.plot(diurnal_df.index, diurnal_df.Fc_storage_obs / diurnal_df.Fc, 
                        color = 'black', label = '$S_c\//\/F_c$', linestyle = ':')
    series_list = series_1 + series_2 + series_3                        
    labs = [ser.get_label() for ser in series_list]
    ax1.legend(series_list, labs, fontsize = 18, loc = [0.75,0.77], 
               numpoints = 1, frameon = False)    
    plt.setp(ax1.get_yticklabels()[0], visible = False)
    plt.setp(ax2.get_yticklabels()[0], visible = False)
    plt.tight_layout()
    fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
                'basic C paper/Images/RUE_and_Sc_on_Fc.png',
                bbox_inches='tight',
                dpi = 300) 
    plt.show()

def plot_Sc_Ac_contributions():

    """
    This script plots a stacked plot showing contributions of Sc and Ac to 
    shortfall between Fc and ER as a function of ustar
    """    

    num_cats = 50   
    
    # Make variable lists
    storage_vars = ['Fc_storage_obs', 'Fc_storage_obs_1', 'Fc_storage_obs_2', 
                    'Fc_storage_obs_3', 'Fc_storage_obs_4', 'Fc_storage_obs_5', 
                    'Fc_storage_obs_6']
    anc_vars = ['ustar', 'ustar_QCFlag', 'Fsd', 'Fsd_QCFlag', 'Ta', 'Ta_QCFlag',
                'Fc', 'Fc_QCFlag']

    # Get data
    df = get_data(storage_vars + anc_vars)
    test_dict = rp_run.main()[0]
    df['Fre_lt'] = test_dict['Re']
    anc_vars.append('Fre_lt')
    
    # Remove daytime, missing or filled data where relevant
    sub_df = df[storage_vars + anc_vars]
    sub_df = sub_df[sub_df.ustar_QCFlag == 0]    
    sub_df = sub_df[sub_df.Fsd_QCFlag == 0]    
    sub_df = sub_df[sub_df.Fc_QCFlag == 0]  
    sub_df = sub_df[sub_df.Fsd < 5]
    sub_df.dropna(inplace = True)    

    # Categorise data
    sub_df['ustar_cat'] = pd.qcut(sub_df.ustar, num_cats, labels = np.linspace(1, num_cats, num_cats))
    new_df = sub_df[['ustar', 'Fc_storage_obs', 'Fc_storage_obs_1', 'Fc_storage_obs_2', 
                     'Fc_storage_obs_3', 'Fc_storage_obs_4', 'Fc_storage_obs_5', 
                     'Fc_storage_obs_6', 'ustar_cat', 'Ta', 'Fc', 'Fre_lt']].groupby('ustar_cat').mean()
    new_df['Fc_storage_obs_std'] = sub_df[['Fc_storage_obs','ustar_cat']].groupby('ustar_cat').std()

    Sc_Ac_area = (new_df.Fre_lt[new_df.ustar<0.42].mean() - 
                  new_df.Fc[new_df.ustar<0.42].mean())
                  
    Sc_propn_total = new_df.Fc_storage_obs[new_df.ustar<0.42].mean() / Sc_Ac_area
    Ac_propn_total = 1 - Sc_propn_total

    # Create plot
    fig = plt.figure(figsize = (8, 8))
    fig.patch.set_facecolor('white')
    ax1 = plt.gca()
    ax1.plot(new_df.ustar, new_df.Fc_storage_obs + new_df.Fc, linestyle = ':', 
             label = '$S_{c}$', color = 'black')
    ax1.plot(new_df.ustar, new_df.Fc, linestyle = '-', 
             label = '$F_{c}$', color = 'black')
    ax1.plot(new_df.ustar, new_df.Fre_lt, linestyle = '--', 
             label = '$\widehat{ER}$', color = 'black')

    x = new_df.ustar
    y1 = new_df.Fc
    y2 = new_df.Fc_storage_obs + new_df.Fc
    ax1.fill_between(x, y1, y2, where=y2>=y1, facecolor='0.8', edgecolor='None',
                     interpolate=True)
    y1 = new_df.Fc_storage_obs + new_df.Fc
    y2 = new_df.Fre_lt
    ax1.fill_between(x, y1, y2, where=y2>=y1, facecolor='0.6', edgecolor='None',
                 interpolate=True)
    ax1.axhline(y = 0, color  = 'black', linestyle = '-')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.set_ylabel(r'$C\/flux\/(\mu mol C\/m^{-2} s^{-1})$', fontsize = 22)
    ax1.set_xlabel('$u_{*}\/(m\/s^{-1})$', fontsize = 22)
    ax1.set_xlim([0, 0.42])
    ax1.tick_params(axis = 'x', labelsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.text(0.08, 1.73, '$Av_c\/+\/Ah_c$\n$=\/$' + str(round(Ac_propn_total, 2)), 
             fontsize = 22, horizontalalignment = 'center')
    ax1.text(0.1, 1.1, '$S_c\/=\/$' + str(round(Sc_propn_total, 2)), 
             fontsize = 22)
    ax1.text(0.3, 0.5, '$F_c$', fontsize = 22)
    plt.setp(ax1.get_yticklabels()[0], visible = False)
    plt.tight_layout()   
    fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
                'basic C paper/Images/ustar_vs_Fc_and_storage_advection1.png',
                bbox_inches='tight',
                dpi = 300) 
    plt.show()
    
    return

def plot_Sc_all_levels_funct_ustar(correct_storage = False):
    """
    This script plots all levels of the profile system as a function of ustar;
    the decline in storage of the levels below 8m with decreasing u* is thought
    to be associated with horizontal advection due to drainage currents, so 
    there is a kwarg option to output additional series where the low level storage 
    estimates are increased by regressing those series on the 8-16m level 
    (between u* = 0.4 and u* = 0.2m s-1) and extrapolating the regression to 
    0.2-0m s-1;
    """
    
    num_cats = 30   

    # Make variable lists
    storage_vars = ['Fc_storage_obs', 'Fc_storage_obs_1', 'Fc_storage_obs_2', 
                    'Fc_storage_obs_3', 'Fc_storage_obs_4', 'Fc_storage_obs_5', 
                    'Fc_storage_obs_6',]
    anc_vars = ['ustar', 'ustar_QCFlag', 'Fsd', 'Ta', 'Fc', 'Fre_lt']    
    var_names = ['0-36m', '0-0.5m', '0.5-2m', '2-4m', '4-8m', '8-16m', '16-36m'] 
    
    # Get data
    df = get_data(storage_vars + anc_vars)
    test_dict = rp_run.main()[0]
    df['Fre_lt'] = test_dict['Re']    
    
    # Remove daytime, missing or filled data where relevant
    sub_df = df[storage_vars + anc_vars]
    sub_df = sub_df[sub_df.ustar_QCFlag == 0]    
    sub_df = sub_df[sub_df.Fsd < 5]
    sub_df.drop(['ustar_QCFlag', 'Fsd'], axis = 1, inplace = True)
    sub_df.dropna(inplace = True)    
    
    # Add a variable
    sub_df['Re_take_Fc'] = sub_df['Fre_lt'] - sub_df['Fc']
    
    # Categorise data
    sub_df['ustar_cat'] = pd.qcut(sub_df.ustar, num_cats, 
                                  labels = np.linspace(1, num_cats, num_cats))
    means_df = sub_df.groupby('ustar_cat').mean()

    # Calculate uncertainty
    error_df = (sub_df[['Fc_storage_obs', 'Re_take_Fc', 'ustar_cat']]
                .groupby('ustar_cat').std() / 
                np.sqrt(sub_df[['Fc_storage_obs', 'Re_take_Fc', 'ustar_cat']]
                        .groupby('ustar_cat').count())) * 2

    # Generate regression statistics and apply to low u* data for low levels 
    if correct_storage:
        stats_df = pd.DataFrame(columns=storage_vars[1:5], index = ['a', 'b'])
        for var in stats_df.columns:
            coeffs = np.polyfit(means_df['Fc_storage_obs_5'][(means_df.ustar < 0.4) & 
                                                         (means_df.ustar > 0.2)],
                                means_df[var][(means_df.ustar < 0.4) & 
                                              (means_df.ustar > 0.2)],
                                1)
            stats_df.loc['a', var] = coeffs[0]
            stats_df.loc['b', var] = coeffs[1]
    
        corr_df = means_df.copy()
        for var in stats_df.columns:
            corr_df[var][corr_df['ustar'] < 0.2] = (corr_df['Fc_storage_obs_5']
                                                    [corr_df['ustar'] < 0.2] * 
                                                    stats_df.loc['a', var] + 
                                                    stats_df.loc['b', var])
        corr_df['Fc_storage_obs'] = corr_df[storage_vars[1:]].sum(axis = 1)
        error_df['Fc_storage_obs_corrected'] = error_df['Fc_storage_obs']
        error_df['Fc_storage_obs_corrected'][corr_df['ustar'] < 0.2] = (
            error_df['Fc_storage_obs'] * 
            corr_df[storage_vars[0]][corr_df['ustar'] < 0.2] /
            means_df[storage_vars[0]][means_df['ustar'] < 0.2])
        corr_df = corr_df[corr_df.ustar < 0.25]
        means_df['Fc_storage_obs_corrected'] = pd.concat(
            [corr_df[storage_vars[0]][means_df.ustar < 0.25],
             means_df[storage_vars[0]][means_df.ustar > 0.25]])
    
    # Create plot
    fig = plt.figure(figsize = (12, 8))
    fig.patch.set_facecolor('white')
    ax1 = plt.gca()
    colour_idx = np.linspace(0, 1, 6)
    vars_dict = {'Fc_storage_obs_corrected': 'blue', 'Re_take_Fc': 'grey'}
    for i, var in enumerate(storage_vars[1:]):
        ax1.plot(means_df.ustar, means_df[var], color = plt.cm.cool(colour_idx[i]), 
                 label = var_names[i + 1])
    ax1.plot(means_df.ustar, means_df[storage_vars[0]], label = var_names[0], 
             color = 'blue')
    if correct_storage:
        ax1.plot(corr_df.ustar, corr_df[storage_vars[0]], color = 'blue', 
                 linestyle = '--')
        ax1.plot(means_df.ustar, means_df.Re_take_Fc, label = '$\widehat{ER}\/-\/F_{c}$',
                 linestyle = '-', color = 'black')
        for var in vars_dict.keys():
            x = means_df.ustar         
            y1 = means_df[var]
            yhi = y1 + error_df[var]
            ylo = y1 - error_df[var]
            ax1.fill_between(x, ylo, yhi, where = yhi >= ylo, facecolor=vars_dict[var], 
                             edgecolor='None', alpha=0.3, interpolate=True)
        x = corr_df.ustar         
        y1 = means_df[storage_vars[0]][means_df.ustar < 0.25]
        y2 = corr_df[storage_vars[0]]
        ax1.fill_between(x, y1, y2, where = y2 >= y1, hatch = '.', alpha = 0.5,
                         interpolate=True, edgecolor = 'black', color='None',
                         linewidth = 0.0)
        for i, var in enumerate(storage_vars[1:]):
            ax1.plot(corr_df.ustar, corr_df[var], color = plt.cm.cool(colour_idx[i]), 
                     linestyle = '--')
    ax1.axhline(y = 0, color  = 'black', linestyle = '-')
    ax1.set_ylabel(r'$S_c\/(\mu mol C\/m^{-2} s^{-1})$', fontsize = 22)
    ax1.set_xlabel('$u_{*}\/(m\/s^{-1})$', fontsize = 22)
    ax1.tick_params(axis = 'x', labelsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.setp(ax1.get_yticklabels()[0], visible = False)
    ax1.legend(fontsize = 16, loc = [0.83,0.6], numpoints = 1, frameon = False)
    fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
                'basic C paper/Images/ustar_vs_storage.png',
                bbox_inches='tight',
                dpi = 300) 
    plt.show()
    
    return

def plot_Sc_dependence_time_after_sunset():
    
    file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L6.nc'

    ustar_threshold = 0.42
    
    data_d1 = io.OzFluxQCnc_to_data_structure(file_in, 
                                              var_list = ['Fc', 'ustar', 'Fsd', 
                                                          'Ta', 'ps'],
                                              QC_accept_codes=[0])
    data_d2 = io.OzFluxQCnc_to_data_structure(file_in, var_list = ['Fc_storage_obs'])
    
    df = pd.DataFrame(data_d1)
    df['Fc_storage_obs'] = data_d2['Fc_storage_obs']
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
    
    hi_ustar_df = df[df.ustar > ustar_threshold]
    hi_means_df = hi_ustar_df.groupby('time_since_sunset').mean()
    hi_count_df = hi_ustar_df.groupby('time_since_sunset').count()
    
    lo_ustar_df = df[df.ustar < ustar_threshold]
    lo_means_df = lo_ustar_df.groupby('time_since_sunset').mean()
    lo_count_df = lo_ustar_df.groupby('time_since_sunset').count()
    
    final_df = pd.DataFrame({'count': hi_means_df.index[1:-1],
                             'hi_Sc_mean': hi_means_df.Fc_storage_obs[1:-1], 
                             'hi_Sc_count': hi_count_df.Fc_storage_obs[1:-1],
                             'hi_Ta_mean': hi_means_df.Ta[1:-1],
                             'lo_Sc_mean': lo_means_df.Fc_storage_obs[1:-1], 
                             'lo_Sc_count': lo_count_df.Fc_storage_obs[1:-1]})
    
    fig = plt.figure(figsize = (12, 9))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1.5])
    ax1 = plt.subplot(gs[0])
    ax1.set_xlim([0.5,14])
    ax1.set_ylabel('$S_c\/(\mu mol C\/m^{-2} s^{-1})$', fontsize = 18)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    ax1.set_xticklabels(['',2,'',4,'',6,'',8,'',10,'',12,'',14])
    ax1.tick_params(axis = 'x', labelsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 14)
    x = final_df.index / 2.0
    y1 = np.zeros(28)
    y2 = final_df.hi_Sc_mean
    y3 = final_df.hi_Ta_mean
    y4 = final_df.hi_Sc_count.cumsum() / (final_df.hi_Sc_count).sum() * 100
    ax1.plot(x, y2, color = 'black', marker = 's', markerfacecolor = 'black',
             markeredgecolor = 'black')
    ax1.fill_between(x, y1, y2, where=y2>=y1, facecolor='0.4', edgecolor='None',
                     interpolate=True)
    ax1.fill_between(x, y1, y2, where=y2<y1, facecolor='0.8', edgecolor='None',
                     interpolate=True)
    ax1.axhline(0, color = 'black')
    ax2 = ax1.twinx()
    ax2.set_xlim([0.5,14])
    ax2.set_ylabel('$Ta\/(^{o}C)$', fontsize = 18)
    ax2.tick_params(axis = 'y', labelsize = 14)
    ax2.plot(x, y3, color = 'black', linestyle=':')
    ax3 = plt.subplot(gs[1])
    ax3.set_ylabel('$\%\/ obs$', fontsize = 18)                 
    ax3.set_xlim([0.5, 14])
    ax3.xaxis.set_ticks_position('bottom')
    ax3.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    ax3.set_xticklabels(['',2,'',4,'',6,'',8,'',10,'',12,'',14])
    ax3.yaxis.set_ticks_position('left')
    ax3.set_xlabel('$Time\/after\/sunset\/(hrs)$', fontsize = 18)
    ax3.tick_params(axis = 'x', labelsize = 14)
    ax3.tick_params(axis = 'y', labelsize = 14)
    ax3.plot(x, y4, color = 'black')
    fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
                'basic C paper/Images/Sc_dependence_time_after_sunset.png',
                bbox_inches='tight',
                dpi = 300)     
    
    return

def plot_Sc_Fc_Ac_funct_ustar():
    
    """
    This script plots Sc and Fc as a function of ustar, but additionally
    calculates estimated advection as a residual (where a model is used to 
    estimate ER
    """    
    
    num_cats = 50   

    # Make variable lists
    use_vars = ['Fc_storage_obs', 'Fc', 'ustar', 'ustar_QCFlag', 
                'Fsd', 'Ta', 'Fc_QCFlag', 'Sws']
    
    # Get data
    df = get_data(use_vars)
    test_dict = rp_run.main(True)[0]
    df['Fre_lt'] = test_dict['Re']
    use_vars.append('Fre_lt')

    # Remove daytime, missing or filled data where relevant
    sub_df = df[use_vars]
    sub_df = sub_df[sub_df.ustar_QCFlag == 0]    
    sub_df = sub_df[sub_df.Fc_QCFlag == 0]  
    sub_df = sub_df[sub_df.Fsd < 5]
    sub_df.dropna(inplace = True)    

    # Generate advection estimate
    sub_df['advection'] = sub_df['Fre_lt'] - sub_df['Fc'] - sub_df['Fc_storage_obs']

    # Categorise data into ustar bins then do means and Sds grouped by categories
    sub_df['ustar_cat'] = pd.qcut(sub_df.ustar, num_cats, labels = np.linspace(1, num_cats, num_cats))
    means_df = sub_df.groupby('ustar_cat').mean()
    CI_df = (sub_df[['Fc','Fc_storage_obs','Fre_lt','advection', 'ustar_cat']]
             .groupby('ustar_cat').std() / 
             np.sqrt(sub_df[['Fc','Fc_storage_obs','Fre_lt','advection', 'ustar_cat']]
             .groupby('ustar_cat').count()) * 2)
    
    # Create plot
    fig = plt.figure(figsize = (12, 8))
    fig.patch.set_facecolor('white')
    ax1 = plt.gca()
    ax1.plot(means_df.ustar, means_df.Fre_lt, linestyle = '--', 
             label = '$\hat{ER}$', color = 'black')
    ax1.plot(means_df.ustar, means_df.Fc, linestyle = '-', 
             label = '$F_{c}$', color = 'black')
    ax1.plot(means_df.ustar, means_df.Fc_storage_obs, linestyle = ':', 
             label = '$S_{c}$', color = 'black')                     
    ax1.plot(means_df.ustar, means_df.advection, 
             linestyle = '-', label = '$Av_{c}\/+\/Ah_{c}$', color = 'grey')
    x = means_df.ustar
    ylo = means_df.advection - CI_df.advection
    yhi = means_df.advection + CI_df.advection
    ax1.fill_between(x, ylo, yhi, where=yhi>=ylo, facecolor='0.8', 
                     edgecolor='None', interpolate=True)
    ax1.axvline(x = 0.42, color  = 'black', linestyle = '-.')
    ax1.axhline(y = 0, color  = 'black', linestyle = '-')
    ax1.set_ylabel(r'$C\/flux\/(\mu mol C\/m^{-2} s^{-1})$', fontsize = 22)
    ax1.set_xlabel('$u_{*}\/(m\/s^{-1})$', fontsize = 22)
    ax1.tick_params(axis = 'x', labelsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.setp(ax1.get_yticklabels()[0], visible = False)
    ax1.legend(fontsize = 18, loc = [0.76,0.4], numpoints = 1, frameon = False)
    fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
                'basic C paper/Images/ustar_vs_Fc_and_storage_advection.png',
                bbox_inches='tight',
                dpi = 300) 
    plt.show()
    
    return

def plot_Sc_Fc_funct_ustar(correct_storage = False):

    """
    This script plots nocturnal Fc and Sc as a function of ustar; it also shows 
    the major respiration drivers on the right hand axis 
    """
    
    num_cats = 30   
    
    # Make variable lists
    use_vars = ['Fc_storage_obs', 'Fc_storage_obs_1', 'Fc_storage_obs_2', 'Fc_storage_obs_3', 
                'Fc_storage_obs_4', 'Fc_storage_obs_5', 'Fc_storage_obs_6', 
                'Fc', 'Fc_QCFlag', 'ustar', 'ustar_QCFlag', 'Fsd', 'Ta', 'Sws']

    # Get data
    df = get_data(use_vars)
    
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
    ax1.axvline(x = 0.42, color  = 'black', linestyle = '--')
    ax1.axhline(y = 0, color  = 'black', linestyle = '-')
    ax1.set_ylabel(r'$C\/flux\/(\mu mol C\/m^{-2} s^{-1})$', fontsize = 22)
    ax2.set_ylabel('$T_{a}\/(^{o}C)\//\/VWC\/(m^{3}m^{-3}\/$'+'x'+'$\/10^{2})$', 
                   fontsize = 20)
    ax2.set_ylim([-5,25])
    ax1.set_xlabel('$u_{*}\/(m\/s^{-1})$', fontsize = 22)
    ax1.tick_params(axis = 'x', labelsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax2.tick_params(axis = 'y', labelsize = 14)
    ax1.xaxis.set_ticks_position('bottom')
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
    
    return

def plot_Sc_diurnal_with_ustar():
    
    """
    This script plots Sc as a function of time of day; it also shows the major 
    respiration drivers on the right hand axis!
    """    
    
    file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L6_stor.nc'
    
    storage_vars = ['Fc_storage_obs', 'Fc_storage_obs_1', 'Fc_storage_obs_2', 
                    'Fc_storage_obs_3', 'Fc_storage_obs_4', 'Fc_storage_obs_5', 
                    'Fc_storage_obs_6',]    
    
    df = io.OzFluxQCnc_to_data_structure(file_in, 
                                         var_list = (storage_vars + 
                                                     ['ustar','Ta', 'Fc',
                                                     'Fc_storage_obs', 'Flu',
                                                     'Fsd']), 
                                         output_structure='pandas')

    diurnal_df = df.groupby([lambda x: x.hour, lambda y: y.minute]).mean()
    diurnal_df['Fc_storage_obs_std'] = df['Fc_storage_obs'].groupby([lambda x: x.hour, 
                                                             lambda y: y.minute]).std()
    diurnal_df.index = np.linspace(0, 23.5, 48)
    day_ind = diurnal_df[diurnal_df.Fsd > 5].index

    storage_mean = diurnal_df.Fc_storage_obs.mean()
    var_names = ['0-36m', '0-0.5m', '0.5-2m', '2-4m', '4-8m', '8-16m', '16-36m']
    
    # Create plot
    fig = plt.figure(figsize = (12, 8))
    fig.patch.set_facecolor('white')
    colour_idx = np.linspace(0, 1, 6)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.set_xlim([0, 24])
    ax1.set_xticks([0,4,8,12,16,20,24])
    ax1.tick_params(axis = 'x', labelsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax2.tick_params(axis = 'y', labelsize = 14)
    ax1.set_xlabel('$Time\/(hours)$', fontsize = 20)
    ax1.set_ylabel('$S_c\/(\mu mol C\/m^{-2} s^{-1})$', fontsize = 20)
    ax2.set_ylabel('$u_{*}\/(m\/s^{-1})$', fontsize = 20)
    series_list = []
    for i, var in enumerate(storage_vars[1:]):
        series_list.append(ax1.plot(diurnal_df.index, diurnal_df[var], 
                                    color = plt.cm.cool(colour_idx[i]), 
                                    label = var_names[i + 1]))
    series_list.append(ax1.plot(diurnal_df.index, diurnal_df.Fc_storage_obs, 
                                color = '0.5', label = var_names[0]))
    series_list.append(ax2.plot(diurnal_df.index, diurnal_df.ustar, 
                                color = 'black', label = '$u_*$'))
    ax1.axhline(storage_mean, color = 'black')
#    ax1.axvline(day_ind[0], color = 'black', linestyle = ':')
#    ax1.axvline(day_ind[-1], color = 'black', linestyle = ':')
    ax2.axhline(0.42, linestyle = '--', color = 'black')    
    plt.setp(ax1.get_yticklabels()[0], visible = False)
    plt.setp(ax2.get_yticklabels()[0], visible = False)
    labs = [ser[0].get_label() for ser in series_list]
    lst = [i[0] for i in series_list]
    ax1.legend(lst, labs, fontsize = 16, loc = [0.08,0.75], 
               numpoints = 1, ncol = 2, frameon = False)
    plt.tight_layout()
    fig.savefig('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo ' \
                'basic C paper/Images/diurnal_storage.png',
                bbox_inches='tight',
                dpi = 300) 
    plt.show()
    
    return

def plot_Sc_ustar_example_time_series():

    """
    This script plots a two week example period of CO2 mixing ratio on LHS and 
    ustar on RH axis (with ustar threshold shown as dotted line)
    """

    # Set var lists    
    dates_list = ['2012-02-11 12:00:00','2012-02-19 12:00:00']    
    vars_list = ['Cc_LI840_32m', 'Cc_LI840_16m', 'Cc_LI840_8m', 'Cc_LI840_4m',
                 'Cc_LI840_2m', 'Cc_LI840_1m', 'ustar']
    new_list = ['0.5m', '2m', '4m', '8m', '16m', '36m']

    # Get data
    df = get_data(vars_list)

    # Allocate tick locations
    tick_locs = [i for i in 
                 df.loc[dates_list[0]: dates_list[1]].index
                 if i.hour == 0 and i.minute == 0]
    tick_labs = [dt.datetime.strftime(i.date(), '%Y-%m-%d') for i in 
                 df.loc[dates_list[0]: dates_list[1]].index
                 if i.hour == 0 and i.minute == 0]

    # Create plot
    fig = plt.figure(figsize = (12, 6))
    fig.patch.set_facecolor('white')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    colour_idx = np.linspace(0, 1, 6)
    ax1.set_ylim([360,520])
    ax1.set_ylabel('$CO_{2}\/(ppm)$', fontsize = 22)

    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels(tick_labs, rotation = 'vertical', fontsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.xaxis.set_ticks_position('bottom')
    ax2.set_ylabel('$u_{*}\/(ms^{-1})$', fontsize = 22)
    ax2.tick_params(axis = 'y', labelsize = 14)
    for i, var in enumerate(vars_list[:-1]):
        ax1.plot(df.loc[dates_list[0]: dates_list[1]].index, 
                 df.loc[dates_list[0]: dates_list[1], var], 
                 label = new_list[i], color = plt.cm.cool(colour_idx[i]))
    ax2.plot(df.loc[dates_list[0]: dates_list[1]].index, 
             df.loc[dates_list[0]: dates_list[1], 'ustar'], label = 'ustar',
             linestyle = ':', color = 'black')
    ax2.axhline(0.42, color = 'grey', linestyle = '--')
    ax1.legend(loc = [0.7, 0.72], ncol = 2)
    plt.tight_layout()
    plt.show()
    
    return

def plot_T_resp():
    
    num_cats = 30
    
    # Make variable lists
    use_vars = ['Fc', 'Fc_QCFlag', 'ustar', 'ustar_QCFlag', 
                'Fsd', 'Fc_storage_obs', 'Ta']

    # Get data
    df = get_data(use_vars)    

    # Remove daytime, missing or filled data where relevant
    df = df[df.Fc_QCFlag == 0]
    df = df[df.ustar_QCFlag == 0]    
    df = df[df.Fsd < 5]
    df = df[df.ustar > 0.42]
    df.drop(['Fc_QCFlag','ustar_QCFlag', 'ustar', 'Fsd'], axis = 1, inplace = True)
    df.dropna(inplace = True)

    # Get T response function
    params = dark.optimise_all({'NEE_series': df.Fc + df.Fc_storage_obs,
                                'TempC': df.Ta},
                                {'Eo_prior': 200,
                                 'rb_prior': 2})
    
    # Put into temperature categories
    df.sort('Ta')
    df['Ta_cat'] = pd.qcut(df.Ta, num_cats, 
                           labels = np.linspace(1, num_cats, num_cats))

    # Do grouping
    mean_df = df.groupby('Ta_cat').mean()
    std_df = df.groupby('Ta_cat').mean()
    count_df = df.groupby('Ta_cat').count()

    # Make NEE estimated series
    mean_df['Fc_est'] = dark.TRF({'TempC': mean_df.Ta}, params['Eo'], params['rb'])

    # Plot
    fig, ax = plt.subplots(1, 1, figsize = (12, 8))
    fig.patch.set_facecolor('white')
    ax.set_xlim([5, 30])
    ax.set_ylim([-2, 6])
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    ax.set_xlabel('$Air\/temperature\/(^oC)$', fontsize = 18)
    ax.set_ylabel(r'$NEE\/(\mu mol C\/m^{-2} s^{-1})$', fontsize = 18)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.plot(df.Ta, df.Fc, '+', color = '0.7')   
    ax.errorbar(x = mean_df.Ta, y = mean_df.Fc, yerr = std_df.Fc / np.sqrt(count_df.Fc) * 2,
                fmt = 'o', color = 'black', marker = 'o', markeredgecolor = 'black', 
                markerfacecolor = 'white', markersize = 7, markeredgewidth = 1)
    ax.plot(mean_df.Ta, mean_df.Fc_est, color = 'black')
         
    plt.show()
    
    return

def plot_T_resp_all():
    
    num_cats = 30
    
    # Make variable lists
    use_vars = ['Fc', 'Fc_QCFlag', 'ustar', 'ustar_QCFlag', 
                'Fsd', 'Fc_storage_obs', 'Ta']

    # Get data
    df = get_data(use_vars)    

    # Remove daytime, missing or filled data where relevant
    df = df[df.Fc_QCFlag == 0]
    df = df[df.ustar_QCFlag == 0]    
    df = df[df.Fsd < 5]
    df = df[df.ustar > 0.42]
    df.drop(['Fc_QCFlag','ustar_QCFlag', 'Fsd'], axis = 1, inplace = True)
    df.dropna(inplace = True)

    # Get T response function without profile measurements
    params_nostor = dark.optimise_all({'NEE_series': df.Fc,
                                       'TempC': df.Ta},
                                      {'Eo_prior': 200,
                                       'rb_prior': 2})

    # Get T response function with profile measurements
    params_stor = dark.optimise_all({'NEE_series': df.Fc + df.Fc_storage_obs,
                                     'TempC': df.Ta},
                                    {'Eo_prior': 200,
                                     'rb_prior': 2}) 

    # Make NEE estimated series without storage
    df['Fc_est'] = dark.TRF({'TempC': df.Ta}, 
                            params_nostor['Eo'], params_nostor['rb'])

    # Make NEE estimated series with storage
    df['Fc_Sc_est'] = dark.TRF({'TempC': df.Ta}, 
                               params_stor['Eo'], params_stor['rb'])

    # Put into temperature categories
    df.sort('Ta')
    df['Ta_cat'] = pd.qcut(df.Ta, num_cats, 
                           labels = np.linspace(1, num_cats, num_cats))

    # Do grouping
    mean_df = df.groupby('Ta_cat').mean()
    std_df = df.groupby('Ta_cat').mean()
    count_df = df.groupby('Ta_cat').count()


    # Plot
    fig, ax = plt.subplots(1, 1, figsize = (12, 8))
    fig.patch.set_facecolor('white')
    ax.set_xlim([5, 30])
    ax.set_ylim([-2, 6])
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    ax.set_xlabel('$Air\/temperature\/(^oC)$', fontsize = 18)
    ax.set_ylabel(r'$NEE\/(\mu mol C\/m^{-2} s^{-1})$', fontsize = 18)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
#    ax.plot(df.Ta, df.Fc, '+', color = '0.7')   
#    ax.errorbar(x = mean_df.Ta, y = mean_df.Fc, yerr = std_df.Fc / np.sqrt(count_df.Fc) * 2,
#                fmt = 'o', color = 'black', marker = 'o', markeredgecolor = 'black', 
#                markerfacecolor = 'white', markersize = 7, markeredgewidth = 1)
    ax.plot(mean_df.Ta, mean_df.Fc, color = 'red')
    ax.plot(mean_df.Ta, mean_df.Fc + mean_df.Fc_storage_obs, color = 'green')
    ax.plot(mean_df.Ta, mean_df.Fc_storage_obs, color = 'black')
         
    plt.show()
    
#def storage_and_T_by_wind_sector():
#
#    # Assign storage and met variables
#    stor_var = 'Fc_storage_obs_6'
#    T_var = 'Ta_HMP_32m'
#    ws_var = 'Ws_RMY_32m'
#    wd_var = 'Wd_RMY_32m'
#    
#    # Wind sectors
#    wind_sectors_dict = {'NNE': [0,45], 'ENE': [45,90], 'ESE': [90,135],
#                         'SSE': [135,180], 'SSW': [180,225], 'WSW': [225,270],
#                         'WNW': [270,315], 'NNW': [315,360]}    
#    
#    # Get data, then subset to exclude extraneous / bad data
#    df, attr = get_data()
#    df = df[df.Fsd < 10]
#    df = df[df[ws_var] != 0]
#    df = df[[stor_var, T_var, ws_var, wd_var]]
#    df.dropna(inplace = True)
#
#    # Separate out by wind sector
#    results_df = pd.DataFrame(index = wind_sectors_dict.keys(), 
#                              columns = ['count','stor_mean', 'T_mean'])
#    for sector in wind_sectors_dict:
#        results_df.loc[sector, 'count'] = len(df[(df[wd_var] > wind_sectors_dict[sector][0]) & 
#                                                     (df[wd_var] < wind_sectors_dict[sector][1])])
#        results_df.loc[sector, 'stor_mean'] = df[stor_var][(df[wd_var] > wind_sectors_dict[sector][0]) & 
#                                                         (df[wd_var] < wind_sectors_dict[sector][1])].mean()
#        results_df.loc[sector, 'T_mean'] = df[T_var][(df[wd_var] > wind_sectors_dict[sector][0]) & 
#                                                     (df[wd_var] < wind_sectors_dict[sector][1])].mean()                                                 
#
#    return results_df                                                 
#
    

    
def calc_annual_sum(use_storage, ustar_threshold, do_light_response):

    reload(rp_run)
    rslt_dict = rp_run.main(use_storage = use_storage,
                            ustar_threshold = ustar_threshold,
                            do_light_response = do_light_response)[0]    
    
    # Generate model estimates
    rslt_dict['NEE_mod'] = rslt_dict['Re'] + rslt_dict['GPP']
    
    n_cases = len(rslt_dict['NEE_series'][~np.isnan(rslt_dict['NEE_series'])])
    print 'The number of available data = ' + str(n_cases)
    
    # Gap fill
    rslt_dict['NEE_filled'] = rslt_dict['NEE_series']
    rslt_dict['NEE_filled'][np.isnan(rslt_dict['NEE_filled'])] = \
        rslt_dict['NEE_mod'][np.isnan(rslt_dict['NEE_filled'])]
    
    # Create dataframe
    df = pd.DataFrame(rslt_dict, index = rslt_dict['date_time'])
    df.drop('date_time', inplace = True, axis = 1)
    df['NEE_sum'] = df['NEE_filled'] * 0.0018 * 12
    
    for yr in ['2012', '2013', '2014']:
        
        print yr + ': ' + str(df.loc[yr, 'NEE_sum'].sum())
    
    