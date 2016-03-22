# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:20:19 2015

@author: imchugh
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pdb

data_min = 364

avg_int = [1971,2000]

years_compare = [2011,2012,2013,2014]

plot_compare = ['climatol']

exclude_noQC = False

df = pd.read_csv('/home/imchugh/Analysis/Whroo/Data/External/BOM_088109_precip_all.csv')

df.index = [dt.datetime(df.loc[i, 'Year'],df.loc[i, 'Month'],df.loc[i, 'Day'])
            for i in df.index]

cols_list = df.columns

# Remove data which has not been QC'd
if exclude_noQC:
    df[cols_list[-3]] = np.where(df[cols_list[-1]]=='Y', df[cols_list[-3]], np.nan)

# Calculate amount of missing data and monthly sums for each requested year
yrs_data_list = []
for yr in years_compare:

    num_days_obs = df.loc[str(yr), cols_list[-3]].groupby(lambda x: x.year).count().loc[yr]
    num_days_yr = 365 if yr % 4 != 0 else 366
    print str(num_days_yr - num_days_obs) + ' days of observations missing from year ' + str(yr)
    temp_df = df.loc[str(yr), cols_list[-3]].groupby(lambda x: x.month).sum()
    temp_df.name = str(yr)
    yrs_data_list.append(temp_df)

monthly_df = pd.concat(yrs_data_list, axis = 1)


# Calculate standard climatology for site using specified averaging interval
climatol = df.loc[str(avg_int[0]): str(avg_int[1]), cols_list[-3]].groupby([lambda x: x.dayofyear]).mean()
climatol.index = [(dt.datetime(2010,1,1) + dt.timedelta(i - 1)).month for i in climatol.index]
monthly_df['climatol_' + str(avg_int[0]) + '-' + str(avg_int[1])] = climatol.groupby(level=0).sum().round(1)

# Calculate mean for all available years    
num_records = df[cols_list[-3]].groupby([lambda x: x.year]).count()
full_years = list(num_records[num_records >= data_min].index)
full_df = pd.concat([df.loc[str(i)] for i in full_years])
monthly_df['all_avail_data'] = (full_df[cols_list[-3]].groupby([lambda x: x.month]).sum() / len(full_years)).round(1)

print 'The following years had required minimum number of days of obs (' + str(data_min) + '):'
print full_years

# Do plotting
fig = plt.figure(figsize=(12,8))
fig.patch.set_facecolor('white')

width = 0.3
x_series = np.linspace(0.5, 11.5, 12)
var_name = [i for i in df.columns if 'climatol in i'] if plot_compare == 'climatol' else 'all_avail_data'
LT_annual_mean = monthly_df[var_name].sum().round(1)

for i, yr in enumerate(years_compare):
    
    sbplt = i + 1 + 220
    ax = fig.add_subplot(sbplt)
    
    annual_mean = monthly_df[str(yr)].sum().round(1)
    
    LT = plt.bar(x_series, monthly_df[var_name], width, color = '0.2')
    annual = plt.bar(x_series + width, monthly_df[str(yr)], width, color = '0.6')
    
    ax.set_title(str(yr), fontsize=20, y=1.03)
    
    if i == 0:
        ax.legend((LT, annual), ('1971-2000', 'year'), bbox_to_anchor=(0.5, 0.99), 
                  frameon=False, fontsize=14)
    if i == 0:
        ax.text(7, 159, 'Climatology: ' + str(LT_annual_mean) + 'mm', fontsize=14)
        ax.text(7, 143, 'Annual: ' + str(annual_mean) + 'mm', fontsize=14)
    else:
        ax.text(8, 159, 'Annual: ' + str(annual_mean) + 'mm', fontsize=14)        
            
    ax.set_xlim([0, 12.6])
    ax.set_xticks(x_series+width)
    if i > 1:    
        ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    else:
        ax.xaxis.set_visible(False)
    ax.set_ylim([0, 180])    
    if i%2 == 0:    
        ax.set_ylabel('Rainfall (mm)', fontsize=16)

    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=12)
    
#plt.figlegend((LT, annual), ('1971-2000', 'year'), 'center', ncol=2, frameon = False, fontsize=10)
plt.tight_layout()
plt.savefig('/home/imchugh/Analysis/Whroo/Images/mangalore_annual_precip_2011_2014.png', bbox_inches='tight')

    
    