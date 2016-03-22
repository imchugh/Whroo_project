# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:25:40 2016

@author: imchugh
"""

# Python standard modules
import os
import copy as cp
import numpy as np
import pdb

# My modules
import DataIO as io
import respiration as re
import data_formatting as dt_fm
import datetime as dt

reload(re)

# Get the data and format appropriately
def get_data(configs_dict):

    # Get file extension and target
    paths_dict = configs_dict['files']
    ext = os.path.splitext(paths_dict['input_file'])[1]
    data_input_target = os.path.join(paths_dict['input_path'],
                                     paths_dict['input_file'])

    # Initialise name change dictionary with new names via common keys
    oldNames_dict = configs_dict['variables']
    newNames_dict = {'carbon_flux':'NEE_series',
                     'carbon_storage': 'Fc_storage',
                     'temperature': 'TempC',
                     'solar_radiation': 'Fsd',
                     'vapour_pressure_deficit': 'VPD',
                     'friction_velocity': 'ustar',
                     'wind_speed': 'ws'}
    names_dict = {oldNames_dict[key]: newNames_dict[key] for key in oldNames_dict}                     

    # get data (screen only the Fc data to obs only)
    if ext == '.nc':
        Fc_dict = io.OzFluxQCnc_to_data_structure(data_input_target,
                                                  var_list = [oldNames_dict
                                                              ['carbon_flux']],
                                                  QC_accept_codes = [0])
        Fc_dict.pop('date_time')
        ancillary_vars = [oldNames_dict[var] for var in oldNames_dict.keys() 
                          if not var == 'carbon_flux']
        ancillary_dict, global_attr = io.OzFluxQCnc_to_data_structure(
                                          data_input_target,
                                          var_list = ancillary_vars,
                                          return_global_attr = True)                                  
        data_dict = dict(Fc_dict, **ancillary_dict)
        T_dict = io.OzFluxQCnc_to_data_structure(data_input_target,
                                                 var_list = ['Ta_HMP_2m', 
                                                             'Ta_HMP_4m', 
                                                             'Ta_HMP_8m',
                                                             'Ta_HMP_16m', 
                                                             'Ta_HMP_36m',
                                                             'Ts'])
        data_dict = dict(data_dict, **T_dict)                                     
    elif ext == '.df':
        data_dict, global_attr = io.DINGO_df_to_data_structure(
                                     data_input_target,
                                     var_list = oldNames_dict.values(),
                                     return_global_attr = True)

    # Rename relevant variables    
    data_dict = dt_fm.rename_data_dict_vars(data_dict, names_dict)

    # Make NEE the sum of Fc + storage if specified
    if configs_dict['global_configs']['use_storage']:
        data_dict['NEE_series'] = data_dict['NEE_series'] + data_dict['Fc_storage']

    # Drop all cases where there is no storage data
    data_dict['NEE_series'][np.isnan(data_dict['Fc_storage'])] = np.nan

    # Remove low ustar values according to threshold
    ustar_threshold = configs_dict['global_configs']['ustar_threshold']
    data_dict['NEE_series'][(data_dict['ustar'] < ustar_threshold) &
                            (data_dict['Fsd'] < 5)] = np.nan    
   
    return data_dict, global_attr
    
#------------------------------------------------------------------------------    
def main(use_storage = True, ustar_threshold = False, 
         config_file = False, do_light_response = False):
    """
    No positional arguments - prompts for a configuration file
    Kwargs: use_storage - if True then algorithm looks for a variable called 
                          Fc_storage and then sums that with Fc
            ustar_threshold - set to a particular value to override the ustar
                              threshold set in the global configs root item of
                              the configuration file (this is done so that can
                              be set in function call from another script)
    """
    
    # Do the respiration fit

    # Get configurations
    if not config_file:
        configs_dict = io.config_to_dict(io.file_select_dialog())
    else:
        configs_dict = io.config_to_dict(config_file)
    configs_dict['global_configs']['use_storage'] = use_storage

    # Override default ustar_threshold if requested by user
    if not isinstance(ustar_threshold, bool):
        if isinstance(ustar_threshold, (int, float)):
            configs_dict['global_configs']['ustar_threshold'] = ustar_threshold

    # Get data
    data_dict, attr = get_data(configs_dict)
    
    # Set up respiration configs and add measurement interval, ustar and output path
    re_configs_dict = dict(configs_dict['respiration_configs'], **
                           configs_dict['global_configs'])
    re_configs_dict['measurement_interval'] = int(attr['time_step'])
    re_full_path = os.path.join(configs_dict['files']['output_path'],
                               configs_dict['respiration_configs']['output_folder'])
    if not os.path.isdir(re_full_path): os.makedirs(re_full_path)
    re_configs_dict['output_path'] = re_full_path

    time_clip_bool_array = data_dict['date_time'] < dt.datetime(2014,1,1)
    
    # Throw out 2014 data
    for key in data_dict.keys():
        data_dict[key] = data_dict[key][time_clip_bool_array]

    # Create index for cases where T is valid for all variables
    bool_list = []
    T_vars = ['Ta_HMP_2m', 'Ta_HMP_4m', 'Ta_HMP_8m', 'Ta_HMP_16m', 'Ta_HMP_36m', 'Ts']
    for var in T_vars:
        bool_list.append(~np.isnan(data_dict[var]))
    master_bool = bool_list[0] & bool_list[1] & bool_list[2] & bool_list[3] & bool_list[4]
    NEE_bool = ~np.isnan(data_dict['NEE_series'])

    # Remove data where missing, then check which temp series has lowest error
    RMSE_list = []
    for var in T_vars:
        data_dict['TempC'] = data_dict[var]
        data_dict[var][np.isnan(master_bool)] = np.nan
        
        # Calculate Re                            
        re_rslt_dict, re_params_dict, re_error_dict = re.main(cp.copy(data_dict), 
                                                              re_configs_dict)
                                                              
        re_est_array = re_rslt_dict['Re'][(data_dict['Fsd'] < 5) & master_bool & NEE_bool]
        re_obs_array  = data_dict['NEE_series'][(data_dict['Fsd'] < 5) & master_bool & NEE_bool]
        
        RMSE_list.append(((re_obs_array - re_est_array) **2).mean())
    
    return {var: RMSE_list[i] for i, var in enumerate(T_vars)}
        
        
