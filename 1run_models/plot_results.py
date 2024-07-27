#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 17:37:17 2024

@author: egom802
"""

import os

rt_dir='/home/egom802/Documents/GitHub/ShoreModel_Benchmark'

os.chdir(rt_dir+'/1run_models/')

from plot_utils.helpers import taylor_plot, ts_plot, heat_plot, coverage_plot
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date
from scipy.stats import pearsonr

#%%
#Script's location:
abs_path= os.path.abspath(os.getcwd())
os.chdir(abs_path)

#CVisionDeficiency friendly palette from  Crameri, F. (2018). 
#Scientific colour maps. Zenodo. http://doi.org/10.5281/zenodo.1243862
cm_data = np.loadtxt("./colormaps/roma.txt")
cvd_fmap = LinearSegmentedColormap.from_list('cvd_friendly', cm_data)

#Import Hybrid ensemble time series (test period)
mat_hyb= pd.read_csv('../1run_models/output/Hybrid_ensemble.csv')  
mat_hyb['Datetime'] = pd.to_datetime(mat_hyb['Datetime'])
mat_hyb = mat_hyb.set_index(['Datetime'])

#Import CNN ensemble time series (test period)
mat_cnn= pd.read_csv('../1run_models/output/Hybrid_ensemble.csv')  
mat_cnn['Datetime'] = pd.to_datetime(mat_cnn['Datetime'])
mat_cnn = mat_cnn.set_index(['Datetime'])

#Import Observed Shoreline
inputs= pd.read_csv('../1run_models/output/inputs_target.csv')  
inputs['Datetime'] = pd.to_datetime(inputs['Datetime'])
inputs = inputs.set_index(['Datetime'])

#Get dates intersection  between  ML ensemble and inputs
idx_intsct= inputs.index.intersection(mat_hyb.index)
inputs= inputs[idx_intsct [0]: idx_intsct[-1]]


###########################PLOTS###############################################

#%%Load observations
rt_dir='/home/egom802/Documents/GitHub/ShoreModel_Benchmark'
os.chdir(rt_dir)

fp = 'datasets' #File path
fn_tran =  'transects_coords.csv' #File name for transects
target_trans = ['Transect2', 'Transect5', 'Transect8'] # Target transects for evaluation



fn_obs =  'shorelines_obs.csv' # Satellite data File name for shoreline observations
#varying degrees of rmse between transects.  Transects 1,3,4 and 8 have the smallest rmse



df_obs = pd.read_csv(os.path.join(fp, 'shorelines', fn_obs), index_col='Datetime')
df_obs.index = pd.to_datetime(df_obs.index)

# taylor_plot(mat_hyb,mat_cnn,inputs,cvd_fmap)

# ts_plot(mat_hyb,mat_cnn,inputs,cvd_fmap)

# # heat_plot(mat_hyb,mat_cnn,inputs)

# coverage_plot(mat_hyb,mat_cnn,inputs)
os.chdir(abs_path)
#%%

# mat_in[ plot_date :mat_in.index[-1]].yout.plot()
ax=mat_hyb.plot()
#Rescale target shoreline time series
df_obs['2014-08':'2018-12-30'].Transect2.plot(ax=ax)

# inputs= df_obs['2014-08':'2018-12-30'].Transect2.resample('D').interpolate(method='linear')
inputs= df_obs['2014-08':'2018-12-30'].Transect2

#%%
lower= mat_cnn.min(axis=1)
upper= mat_cnn.max(axis=1)
mean= mat_cnn.mean(axis=1)
lower_hyb= mat_hyb.min(axis=1)
upper_hyb= mat_hyb.max(axis=1)
mean_hyb= mat_hyb.mean(axis=1)
######################PLOT FIGURE 7########################################
fig,ax = plt.subplots(figsize=(11.8,6.8))
fs= 20
fs_tk= 14
fs_leg= 12
colors = cvd_fmap(np.linspace(0, 1, 4))

#CNN-LSTM ensemble
ax.plot(inputs,'--k', markersize=5)
ax.plot(inputs,'*k', markersize=5, label="Observations")
ax.fill_between(mat_hyb.index, y1=lower_hyb, y2=upper_hyb, 
             alpha=0.3, color=colors[3])
ax.plot(mat_hyb.index, mean_hyb, alpha=0.9,
              color=colors[3], label= "CNN-LSTM")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlim([date(2014, 8, 9), date(2016, 12, 30)])
ax.grid()
ax.legend(loc='lower right',fontsize=fs_leg)
ax.tick_params(axis='both', which='major', labelsize=fs_tk)

plt.rcParams.update({'font.size': 28})
fig.text(0.04, 0.5, 'Cross-shore displacement ' r'$[m]$', va='center',
     rotation='vertical',fontsize=fs)
plt.title('Shoreshop Transect 2')

# Add text at a specific point in the plot
text_date = pd.Timestamp('2015-02-10')
plt.text(text_date, 235, 'RMSE:' + str(round(np.mean(rmse_arr),2)), fontsize=12, color='red')
plt.text(text_date, 230, 'Pearson r:' + str(round(np.mean(pear_arr),2)), fontsize=12, color='red')
plt.text(text_date, 225, 'Mielke:' + str(round(np.mean(mielke_arr),2)), fontsize=12, color='red')

#Uncomment to save plot
# plt.savefig('./figures/Fig7.png',
#             bbox_inches='tight',dpi=300)
