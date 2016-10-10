# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:26:30 2016

@author: imchugh
"""
# Standard modules
import os

# My modules
import DataIO as io

def get_data(var_list = None):
    
    reload(io)    
    
    file_in = '/home/imchugh/Ozflux/Sites/Whroo/Data/Processed/all/Whroo_2011_to_2014_L6.nc'
    
    return io.OzFluxQCnc_to_data_structure(file_in, var_list = var_list, 
                                           output_structure = 'pandas')
                                           
def set_output_path(f_name):
    
    dir_path = ('/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo/basic C paper/Images/')
#                '/media/Data/Dropbox/Work/Manuscripts in progress/Writing/Whroo/' \
#                'basic C paper/Images
    return os.path.join(dir_path, f_name)