#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 14:55:34 2024

@author: egom802
"""

#Load datasets
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from windrose import WindroseAxes
import matplotlib.cm as cm


rt_dir='/home/egom802/Documents/GitHub/ShoreModel_Benchmark'

os.chdir(rt_dir+'/1run_models/')

from preprocess import normal_data2,create_mat_satbin, wave_transform

os.chdir(rt_dir)

#%%

fp = 'datasets' #File path
fn_tran =  'transects_coords.csv' #File name for transects
target_trans = ['Transect2', 'Transect5', 'Transect8'] # Target transects for evaluation


# Read data
df_tran = pd.read_csv(os.path.join(fp, fn_tran), index_col='ID')

fn_obs =  'shorelines_obs.csv' # Satellite data File name for shoreline observations
#varying degrees of rmse between transects.  Transects 1,3,4 and 8 have the smallest rmse
fn_gt =  'shorelines_groundtruth.csv' #File name for groudtruth data

fn_targ_short =  'shorelines_target_short.csv' # File name for short-term shoreline prediction target
fn_targ_medium =  'shorelines_target_medium.csv' # File name for medium-term shoreline prediction target


# Read shoreline data
df_gt = pd.read_csv(os.path.join(fp, 'shorelines', fn_gt), index_col='Datetime')
df_gt.index = pd.to_datetime(df_gt.index)

df_targ_short = pd.read_csv(os.path.join(fp, 'shorelines', fn_targ_short), index_col='Datetime')
df_targ_short.index = pd.to_datetime(df_targ_short.index)

df_targ_medium = pd.read_csv(os.path.join(fp, 'shorelines', fn_targ_medium), index_col='Datetime')
df_targ_medium.index = pd.to_datetime(df_targ_medium.index)

df_obs = pd.read_csv(os.path.join(fp, 'shorelines', fn_obs), index_col='Datetime')
df_obs.index = pd.to_datetime(df_obs.index)
df_obs

#%%

# Temporal view of shoreline position
df_diff = df_obs.resample('D').interpolate('linear').reindex(df_gt.index) - df_gt
RMSE = np.sqrt((df_diff**2).mean())

fig, axes = plt.subplots(len(df_tran), 1, figsize=(9.6, 9.3))

# Iterate transects 
for i, tran_id in enumerate(df_tran.index):
    
    #ax = axes[int(i%(len(df_tran)/2)), int(i//(len(df_tran)/2))]
    ax = axes[i]
    
    # Plot time series of shoreline
    ax.plot(df_obs.index, df_obs[tran_id], color='k', label='Observation', zorder=-1, alpha=0.8)
    ax.scatter(df_gt.index, df_gt[tran_id], color='r', marker='*', 
               label='Ground Truth', zorder=1)
    ax.scatter(df_targ_medium.index, df_targ_medium[tran_id], color='k', marker='*', 
               label='Medium-term context', zorder=1)
    
    ax.fill_between(df_targ_medium.index, ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.3, 
                    color='lightcoral', label='Medium-term target')
    ax.fill_between(df_targ_short.index, ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.3, 
                    color='forestgreen', label='Short-term target')
    
    ax.set_title('{}, RMSE={:.3}'.format(tran_id, RMSE[tran_id]))
    ax.set_xlim((df_targ_medium.index.min()-pd.Timedelta(days=365), df_targ_short.index.max()))
    
    if i == 0:
        ax.set_ylabel('Shoreline\n Position (m)')
        ax.legend(ncol=5, fontsize=8, loc=2)
        
fig.subplots_adjust(hspace=1)

#%% Load waves

# Read data

# Hs: Significant wave height
# Tp: Peak wave period
# Dir: Mean wave direction

WAVE_PARAMS = ['Hs', 'Tp', 'Dir'] 

dfs_wave = {}
for wave_param in WAVE_PARAMS:
    df_wave = pd.read_csv(
        os.path.join(fp, 'hindcast_waves' ,'{}.csv'.format(wave_param)),
        index_col = 'Datetime'
    )
    df_wave.index = pd.to_datetime(df_wave.index)
    dfs_wave[wave_param] = df_wave
    
#%%

# Plot wave roses

fig, axes = plt.subplots(ncols=3, figsize=(10, 3),
                        subplot_kw={'projection': 'windrose'})

for i, tran_id in enumerate(target_trans):
    ax = axes[i]
    
    # Plot wave rose for Hs
    ax.bar(dfs_wave['Dir'][tran_id], dfs_wave['Hs'][tran_id], normed=True, opening=0.8, bins=[0, 1, 1.5, 2], 
        cmap=cm.RdYlGn_r, edgecolor='white')
    ax.set_title(tran_id)

    if i == 2:
        cbar = ax.legend(loc=(1.1, -0.1), title='Hs (m)')

plt.tight_layout()

#%%

#Put all training info together

#First, cut the wave data, 2000-2020. no need to Re-sample daily. STart with trs2
#Transform wave direction to components

mat_in=pd.DataFrame()
for wave_param in WAVE_PARAMS:
    mat_in[wave_param]= dfs_wave[wave_param]['2000':'2018-12-30'].Transect2

df_wvx,df_wvy= wave_transform(mat_in['Tp'],mat_in['Dir'])

#Need to append shoreline data to the last column
mat_in['Wvx']=df_wvx
mat_in['Wvy']=df_wvy
mat_in= mat_in.drop('Dir', axis=1)

#Resample daily. Linear interpolation

df_shore= df_obs.resample('D').interpolate(method='linear')['2000':'2018-12-30'].Transect2

mat_in['yout']= df_shore

#%%














