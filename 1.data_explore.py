#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:07:42 2024

@author: egom802
"""


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from windrose import WindroseAxes
import matplotlib.cm as cm

os.chdir('/home/egom802/Documents/GitHub/ShoreModel_Benchmark')
#%%
# Set inputs
fp = 'datasets' #File path
fn_tran =  'transects_coords.csv' #File name for transects
target_trans = ['Transect2', 'Transect5', 'Transect8'] # Target transects for evaluation


# Read data
df_tran = pd.read_csv(os.path.join(fp, fn_tran), index_col='ID')
df_tran

# Visualize transects
fig, ax = plt.subplots(1,1)

# Plot transects
ax.plot(df_tran[['Land_x', 'Sea_x']].transpose(), 
        df_tran[['Land_y', 'Sea_y']].transpose(),
        color='k')

# Plot target transects
ax.plot(df_tran.loc[target_trans, ['Land_x', 'Sea_x']].transpose(), 
        df_tran.loc[target_trans, ['Land_y', 'Sea_y']].transpose(),
        color='g')
ax.plot([], [], color='g', label='Target\n Transects')

# Highlight landward ends
df_tran.plot.scatter('Land_x', 'Land_y', c='r', 
                     label='Land end', ax=ax)


# Add transect labels
for i, row in df_tran.iterrows():
    ax.text(row['Land_x'], row['Land_y'], i)

# Set ax labels
ax.set_aspect('equal')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')

ax.legend()
plt.savefig('figures/transects.jpg', dpi=300, bbox_inches='tight')

fn_obs =  'shorelines_obs.csv' # File name for shoreline observation
fn_targ_short =  'shorelines_target_short.csv' # File name for short-term shoreline prediction target
fn_targ_medium =  'shorelines_target_medium.csv' # File name for medium-term shoreline prediction target
fn_gt =  'shorelines_groundtruth.csv' #File name for groudtruth

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

# Spatial view of shoreline position
land_x, land_y = df_tran['Land_x'], df_tran['Land_y'] # Land end coords
sea_x, sea_y = df_tran['Sea_x'], df_tran['Sea_y'] # Sea end coords
tran_len = np.sqrt((sea_x-land_x)**2+(sea_y-land_y)**2) # Length of transect


fig, ax = plt.subplots(1,1)

# Plot transects
ax.plot(df_tran[['Land_x', 'Sea_x']].transpose(), 
        df_tran[['Land_y', 'Sea_y']].transpose(),
        color='k')

# Plot land ends
df_tran.plot.scatter('Land_x', 'Land_y', c='r', 
                     label='Land end', ax=ax)

# Iterate dates
for date, row_obs in df_obs.iterrows():
    
    # Only label first shoreline
    if date == df_obs.index[0]:
        label_obs = 'Shoreline'
    else:
        label_obs = None
    
    # Calculate shoreline coords
    x_obs = land_x + row_obs/tran_len*(sea_x-land_x)
    y_obs = land_y + row_obs/tran_len*(sea_y-land_y)
    
    # Plot shoreline
    ax.plot(x_obs, y_obs, alpha=0.5, linewidth=0.5, label=label_obs)
        
    
    
ax.set_aspect('equal')
ax.set_xlabel('Easting')
ax.set_ylabel('Northing')

ax.legend()
plt.savefig('figures/shorelines_spatial.jpg', dpi=300, bbox_inches='tight')


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
# plt.tight_layout()
plt.savefig('figures/shorelines_temporal.jpg', dpi=300, bbox_inches='tight')


#%%

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
plt.savefig('figures/wave_roses.jpg', dpi=300, bbox_inches='tight')


#%%


# Plot timeseries wave parameters averaged over transects
tran_id = 'Transect1'
labels = ['Hs (m)', 'Tp (s)', 'Dir ($^\circ$)']

fig, axes = plt.subplots(3, 1, figsize=(8, 6))

for i, wave_param in enumerate(WAVE_PARAMS):
    # Calculate the average 
    df_mean = dfs_wave[wave_param].mean(1)
    ax = axes[i]
    ax.plot(df_mean.index, df_mean.values, color='k', linewidth=0.1)
    ax.fill_between(df_targ_medium.index, ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.3, 
                    color='lightcoral', label='Medium-term target')
    ax.fill_between(df_targ_short.index, ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.3, 
                    color='forestgreen', label='Short-term target')
    ax.set_ylabel(labels[i])
plt.savefig('figures/wave_ts.jpg', dpi=300, bbox_inches='tight')    


#%%

# Read data

df_SLR_obs = pd.read_csv(
    os.path.join(fp, 'sealevel', 'SLR_obs.csv'),
    index_col = 'Year')
df_SLR_proj = pd.read_csv(
    os.path.join(fp, 'sealevel', 'SLR_proj.csv'),
    index_col = 'Year')

#%%

# Plot time series
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
# ax.hlines(df_SLR_obs['SLR (mm)'].mean(), df_SLR_obs.index[0], df_SLR_obs.index[-1], 
#           color='r', linestyle='--', label='baseline (Mean of observation)')
a, b = np.polyfit(df_SLR_obs.index, df_SLR_obs['SLR (mm)'], 1)
ax.plot(df_SLR_obs.index, df_SLR_obs['SLR (mm)'], color='k', linestyle='-', label='Observation')
ax.plot(df_SLR_obs.index, a*df_SLR_obs.index+b, color='k', linestyle='--', label='Trend of observation')
ax.plot(df_SLR_proj.index, df_SLR_proj['RCP85'], color='r', linestyle='-', label='RCP8.5 Projection')
ax.plot(df_SLR_proj.index, df_SLR_proj['RCP45'], color='g', linestyle='-', label='RCP4.5 Projection')

ax.set_xlabel('Year')
ax.set_ylabel('Sea level (m)')

plt.legend()
plt.savefig('figures/SLR_ts.jpg', dpi=300, bbox_inches='tight') 


