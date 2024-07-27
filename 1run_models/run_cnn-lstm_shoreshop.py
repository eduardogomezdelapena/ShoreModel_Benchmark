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
#%%
from utils.splits import normal_data, split_sequences
from utils.loss_errors import index_mielke, mielke_loss

import os
import random
import numpy as np
import pandas as pd
import math
import scipy.stats
from datetime import  timedelta
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.layers import Flatten, RepeatVector,TimeDistributed
from keras.layers import Conv1D, MaxPooling1D

#%%



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

mat_in=mat_in.dropna()
#create mat out only with shoreline data
mat_out= pd.DataFrame(mat_in.yout.values, index=mat_in.index)

#%%
#Remember: Tairua the forecast is from July 2014 (2014-07-01), previous data 
#is training in SPADS and ShoreFor
date_forecast= '2014-07-01'
#Train until  2 years before the test set
train_date=pd.to_datetime(date_forecast) - timedelta(days=365*2)
train_date= str(train_date.strftime("%Y-%m-%d"))
_, mat_in = normal_data(mat_in,train_date)
scaler, mat_out_norm = normal_data(mat_out,train_date)
#%%##################TRAIN, DEV AND TEST SPLITS################################
#Manually split by date 
train = mat_in[mat_in.index[0]:train_date].values.astype('float32')
#Development set (2 years before the test set)
devinit_date=pd.to_datetime(train_date) + timedelta(days=1)
devinit_date= str(devinit_date.strftime("%Y-%m-%d"))
dev_date=pd.to_datetime(date_forecast) - timedelta(days=1)
dev_date= str(dev_date.strftime("%Y-%m-%d"))
dev= mat_in[devinit_date:dev_date].values.astype('float32')
#Test set, depends on study site
test = mat_in[date_forecast:mat_in.index[-1]].values.astype('float32')
#%%############################################################################
#From pandas to array, HERE WE SEPARATE THE INPUTS FROM THE Y_OUTPUT
# split a multivariate sequence into samples for LSTM
#how many time steps is the network allowed to look back
n_steps_in, n_steps_out =40,1
train_x, train_y = split_sequences(train, n_steps_in, n_steps_out)
dev_x, dev_y = split_sequences(dev, n_steps_in, n_steps_out)
test_x, test_y = split_sequences(test, n_steps_in, n_steps_out)
# # the dataset knows the number of features, e.g. 2
n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[
    2], train_y.shape[1]
#%%###################Fix random seed##########################################
def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
#%%####################CONV-LSTM Network#######################################
loss= mielke_loss
min_delta= 0.001
def cnn_custom(train_x, train_y, dev_x, dev_y, cfg):
    print("--------------------------------")
    print("Model:", cfg)
    set_seed(33)
    # define model    # create configs
    n_filters, n_kernels, n_mem,n_drop,n_epochs,n_batch = cfg    
    model = Sequential()
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernels,
                     activation='relu', input_shape=(n_steps_in, n_features)))
    model.add(Conv1D(filters=n_filters, kernel_size=n_kernels,
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2)) 
    model.add(Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(n_mem, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(Dropout(n_drop))      
    model.add(TimeDistributed(Dense(1)))    
    model.compile(optimizer='adam', loss=loss)
    # fit model
    es = EarlyStopping(patience=10, verbose=2, min_delta=min_delta, 
                       monitor='val_loss', mode='auto',
                       restore_best_weights=True)
    history= model.fit(train_x, train_y, validation_data=(dev_x, dev_y),
                         batch_size=n_batch, epochs=n_epochs, verbose=2,
                         callbacks=[es])  
    return model, history
#%%###################### Load Grid Search Hyperparameters#####################
scores = list()   
cfg_list=pd.read_csv(rt_dir+'/1run_models/'+'/hyp/10best_hyp_Wavesonly_Mielke_Hybrid.csv')
cfg_list= cfg_list[["f","k","m","D","e","b"]]
cfg_list= cfg_list.values.tolist()
for i in range(len(cfg_list)):     
    for element in range(len(cfg_list[i])):
        #Position where dropout percentage is
        if element != 3:
            cfg_list[i][element] = int(cfg_list[i][element])
#%%#Run model configurations in loop###########################################
#Predefine empty dataframe
plot_date = pd.to_datetime(date_forecast) + timedelta(days=n_steps_in)
plot_date= str(plot_date.strftime("%Y-%m-%d"))
yresults= pd.DataFrame(index=mat_in[ plot_date :mat_in.index[-1]].index,
                       columns=['ann1','ann2','ann3','ann4','ann5',
                                'ann6','ann7','ann8','ann9','ann10'])
#Rescale target shoreline time series
testY = scaler.inverse_transform(test_y)
#train the models
for (index, colname) in enumerate(yresults):
    print('Model number:' + str(index))
    #Train model with hyp config from config list
    model,_ = cnn_custom(train_x, train_y, dev_x, dev_y, cfg_list[index]) 
    testdl = model.predict(test_x)     

    yresults.iloc[:,index]= scaler.inverse_transform(testdl.reshape(
        testdl.shape[0]*testdl.shape[1],testdl.shape[2]))

    
    print('Metrics')
    print('RMSE:' )
    print(str(math.sqrt(mean_squared_error(yresults.iloc[:,index].values,
                                           testY))))
    print('Pearson:' )
    print(str(scipy.stats.pearsonr(yresults.iloc[:,index].values,
                                   testY[:,0])[0]))
    print('Mielke:' )    
    print(str(index_mielke(yresults.iloc[:,index].values,testY[:,0])))



yresults.to_csv('./1run_models/output/Hybrid_ensemble.csv')
mat_in.to_csv('./1run_models/output/inputs_target.csv')

#Metrics 
# rmse_arr=np.array([math.sqrt(mean_squared_error(yresults[colname].values,testY)) for (index, colname) in enumerate(yresults)])
# pear_arr=np.array([scipy.stats.pearsonr(yresults[colname].values,testY[:,0])[0] for (index, colname) in enumerate(yresults)])
# mielke_arr=np.array( [index_mielke(yresults[colname].values,testY[:,0]) for (index, colname)






