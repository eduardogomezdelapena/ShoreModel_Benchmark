#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 16:27:14 2024

@author: egom802
"""

import os

import datetime
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import string

from sklearn.metrics import r2_score, mean_squared_error

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from matplotlib import gridspec
import matplotlib.lines as mlines
import seaborn as sns
import skill_metrics as sm
from sklearn.metrics import r2_score

import matplotlib.patches as mpatches

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors  # For RGBA to hex conversion

from src.plots import *
from src.utilities import cal_metrics, mielke_lambda, percentile_mean, fluctuation_frequency

# Set inputs
TRANSECTS = ['Transect2', 'Transect5', 'Transect8'] # List of transects for evaluation


fp_input = 'datasets/shorelines' # File path for input data
fp_sub = 'submissions/{}' # File path for submission
fp_resub = 'resubmissions/{}' # File path for resubmission

fn_obs = 'shorelines_obs.csv' # File name for observed shoreline
fn_cali = 'shorelines_calibration.csv' # File name for calibration shoreline
fn_targ_short = 'shorelines_hidden_short.csv' # File name for target shoreline (short-term)
# fn_targ_medium = 'shorelines_hidden_medium.csv' # File name for target shoreline (medium-term)
fn_pred_short = 'shorelines_prediction_short.csv' # File name for predicted shoreline (short-term)
# fn_pred_medium = 'shorelines_prediction_medium.csv' # File name for predicted shoreline (medium-term)
# fn_pred_RCP45 = 'shorelines_prediction_long_RCP45.csv' # File name for predicted shoreline (short-term)
# fn_pred_RCP85 = 'shorelines_prediction_long_RCP85.csv' # File name for predicted shoreline (medium-term)


# n_colors = 10

# # Define the colormaps to sample from
# colormap_names = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds']

# # Create an array to store the colors
# colors = []

# # Sample colors from each colormap
# for cmap_name in colormap_names:
#     cmap = plt.get_cmap(cmap_name, n_colors)
#     colors.extend(cmap(np.linspace(0, 1, n_colors))[2:9])

# # Create a new colormap from the combined colors
# cmap = ListedColormap(colors)
# cmap

# Get the colors from tab20b and tab20c
tab20b = plt.get_cmap('tab20b')
tab20c = plt.get_cmap('tab20c')

# Combine the colors
colors_tab20b = tab20b(np.linspace(0, 1, 20))
colors_tab20c = tab20c(np.linspace(0, 1, 20))

# Concatenate the two sets of colors
combined_colors = np.vstack((np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]), colors_tab20b, colors_tab20c))

# Create a new colormap from the combined colors
cmap = plt.matplotlib.colors.ListedColormap(combined_colors)
cmap


# Read model metadata
df_meta = pd.read_excel('model_summary.xlsx', header=[1])
df_meta = df_meta[~df_meta['Model Name'].str.contains('corrected_JAAA')]
df_meta['Process'] = 'CS_LS'
df_meta.loc[(df_meta['Cross-Shore'].isna())&(~df_meta['Long-Shore'].isna()), 'Process']='LS_Only'
df_meta.loc[(~df_meta['Cross-Shore'].isna())&(df_meta['Long-Shore'].isna()), 'Process']='CS_Only'


# Define groups of interest
model_groups = {
    'DDM': list(df_meta.loc[df_meta['Type']=='DDM', 'Model Name']),
    'HM': list(df_meta.loc[df_meta['Type']=='HM', 'Model Name']),
    'PBM': list(df_meta.loc[df_meta['Type']=='PBM', 'Model Name']),
    'COCOON': list(df_meta.loc[df_meta['Model Name'].str.contains('COCOON'), 'Model Name']),
    'CoSMoS': list(df_meta.loc[df_meta['Model Name'].str.contains('CoSMoS'), 'Model Name']),
    'LSTM': list(df_meta.loc[df_meta['Model Name'].str.contains('LSTM'), 'Model Name']),
    'ShoreFor': list(df_meta.loc[df_meta['Model Name'].str.contains('ShoreFor'), 'Model Name']),
    'CS_Only': list(df_meta.loc[df_meta['Process']=='CS_Only', 'Model Name']),
    'LS_Only': list(df_meta.loc[df_meta['Process']=='LS_Only', 'Model Name']),
    'CS_LS': list(df_meta.loc[df_meta['Process']=='CS_LS', 'Model Name']),
}


#%% Prediction evaluation (Short-term)
#2.1.1 Load predictions, calculate ensemble and loss

# Load prediction
MODELS = list(df_meta['Model Name'].values)
MODEL_TYPES = dict(zip(MODELS, df_meta['Type']))
MODEL_COLORS = dict(zip(df_meta['Model Name'], cmap.colors[0:len(df_meta)]))

# Read obs and calibration shoreline data
df_obs = pd.read_csv(os.path.join(fp_input, fn_obs), index_col='Datetime')
df_obs.index = pd.to_datetime(df_obs.index)
df_obs.sort_index(inplace=True)

# Read obs and calibration shoreline data
df_targ = pd.read_csv(os.path.join(fp_input, fn_targ_short), index_col='Datetime',
                      date_format='%mm/%dd/%yy')
df_targ.index = pd.to_datetime(df_targ.index)
df_targ.sort_index(inplace=True)

# Read model calibrations
dfs_pred = {}
freqs_pred = {} # This saves the frequency level of preds. High freq preds will be plotted on the bottom.
for model in MODELS:
    if df_meta.loc[df_meta['Model Name']==model, 'Submission Type'].values[0] == 'Submission':
        fp = fp_sub
    else:
        fp = fp_resub
    
    try:
        df_pred = pd.read_csv(os.path.join(fp.format(model), fn_pred_short), index_col='Datetime')
    except:
        df_pred = pd.read_csv(os.path.join(fp.format(model), fn_pred_short), index_col='datetime')
        df_pred.index.name = 'Datetime'
    df_pred.index = pd.to_datetime(df_pred.index, format='mixed')
    df_pred.sort_index(inplace=True)
    dfs_pred[model] = df_pred
    freqs_pred[model] = df_pred.apply(fluctuation_frequency).median()
    
#%%

# Calculate ensembles
ensemble_values = []
index = pd.date_range(start='2019-01-01', end='2023-12-31')

for key, df_pred in dfs_pred.items():
    ensemble_values.append(df_pred[TRANSECTS].reindex(index).values) 
ensemble_values = np.stack(ensemble_values)
#ensemble_mean = np.mean(ensemble_values, axis=0)
ensemble_mean = percentile_mean(ensemble_values, 5, 95, axis=0)
ensemble_median = np.nanmedian(ensemble_values, axis=0)
ensemble_max = np.nanmax(ensemble_values, axis=0)
ensemble_min = np.nanmin(ensemble_values, axis=0)
ensemble_std = np.nanstd(ensemble_values, axis=0)

dfs_pred['Ensemble'] = pd.DataFrame(ensemble_mean, columns=TRANSECTS, index=index)
if 'Ensemble' not in MODELS:
    MODELS.append('Ensemble')
MODEL_TYPES['Ensemble'] = 'ENS'
MODEL_COLORS['Ensemble'] = 'k'
#%%
# Calculate metrics
df_loss = pd.DataFrame(columns=TRANSECTS, index=dfs_pred.keys()) # Dataframe to save loss for model and transects
metrics_all = {}

for i, tran_id in enumerate(TRANSECTS):
    metrics = {}
    
    # Calculate metrics for the target
    metrics["Target"] = cal_metrics(df_targ[[tran_id]], df_targ[[tran_id]])
    metrics["Prediction"] = {}
    for model in dfs_pred.keys():
        # Calculate metrics for predictions
        metrics["Prediction"][model] = cal_metrics(df_targ[[tran_id]], dfs_pred[model][[tran_id]])
        df_loss.loc[model, tran_id] = metrics["Prediction"][model]['loss']
    metrics_all[tran_id] = metrics
    
# Sort zorder based on freq
sorted_freq = {k: v for k, v in sorted(freqs_pred.items(), key=lambda item: item[1], reverse=True)}
zorders = {model: i for i, model in enumerate(sorted_freq.keys(), 1)}
zorders['Ensemble'] = 41
zorders['CoSMoS-COAST-CONV_SV'] = 40
zorders['GAT-LSTM_YM'] = 39
zorders['iTransformer-KC'] = 38
#%% # Plot timeseries comparison
fig = plot_ts(TRANSECTS, df_targ=df_targ, dfs_pred=dfs_pred, task='short', zorders=zorders, colors=MODEL_COLORS)
plt.savefig('figures/Short/Timeseries_all.jpg', dpi=300, bbox_inches='tight')

fig = plot_ts_interactive(TRANSECTS, df_targ=df_targ, dfs_pred=dfs_pred, task='short', zorders=zorders, 
                         colors=MODEL_COLORS, loss=df_loss.mean(1))
fig.write_html('figures/Short/Timeseries.html')

# Save timeseries for group comparison
for group_name, group_elements in model_groups.items():
    dfs_pred_group = {key: dfs_pred[key] for key in group_elements if key in dfs_pred}
    fig = plot_ts(TRANSECTS, df_targ=df_targ, dfs_pred=dfs_pred_group, task='short', zorders=zorders, 
                  colors=MODEL_COLORS, loss=df_loss.mean(1))
    plt.savefig('figures/Short/Timeseries_{}.jpg'.format(group_name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
#%% 2.1.3 Taylor Diagram for model ranking


# MODEL_COLORS.update({"Ensemble": 'k'})

# del MODEL_COLORS['Ensemble']
# del MODEL_TYPES['Ensemble']

fig, axes = plt.subplots(1, 3, figsize=(15, 7))


for i, tran_id in enumerate(TRANSECTS):
    metrics = metrics_all[tran_id]
    ax = axes[i]
    if i != len(TRANSECTS)-1:
        ax = plot_taylor(metrics, MODEL_TYPES, MODEL_COLORS, legend=None, ax=ax, 
                         SDS_RMS=round(10/df_targ.std()[tran_id], 2))
    else:
        ax = plot_taylor(metrics, MODEL_TYPES, MODEL_COLORS, legend='Average', 
                         aver_scores=df_loss.mean(1), ax=ax, 
                         SDS_RMS=round(10/df_targ.std()[tran_id], 2))
        
    ax.set_title('Short-Term Prediction: {}'.format(tran_id), loc="left", y=1.1)

plt.subplots_adjust(wspace=0.2)
plt.savefig('figures/Short/TaylorDiagram.jpg', dpi=300, bbox_inches='tight')
df_loss['Avg'] = df_loss.mean(1)



























