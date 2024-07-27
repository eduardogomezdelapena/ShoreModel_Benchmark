#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:54:05 2024

@author: egom802
"""

import scipy.io
import pandas as pd
from datetime import timedelta
from datetime import date
import datetime
import matplotlib.pyplot as plt
import os
usr='egom802'
rt_dir='/home/'+usr+'/Documents/GitHub/ANNs_4_USCoasts/OceanBeach/1run_models'
os.chdir(rt_dir)

from utils.splits import normal_data, split_sequences
from utils.loss_errors import index_mielke, mielke_loss

import os
import random
import numpy as np
import math
import scipy.stats

from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

def matnum_2_py(vec_2_convert):
    """  """
    py_dates=[]
    for i in range(len(vec_2_convert[0])):
        python_datetime = date.fromordinal(int(vec_2_convert[0][i])) + timedelta(days=int(vec_2_convert[0][i]%1)) - timedelta(days = 366)
        py_dates.append(python_datetime)
    time_cosmos = py_dates
    return time_cosmos


def matnum_2_py_mat(mat_2_convert):
    """ Transforms a 2d matrix with datenums to a python readible
    date time index. This is the context of reading a .mat struct into
    python"""
    # Create an empty pandas dataframe with rows and columns that match time_gps
    num_rows = len(mat_2_convert)
    num_columns = len(mat_2_convert[0])
    py_dates  = pd.DataFrame(index=range(num_rows), columns=range(num_columns))

    for j in range(len(mat_2_convert[0])): 
        i=0 #row inex
        while mat_2_convert[i][j] != 0:
            python_datetime = date.fromordinal(int(mat_2_convert[i][j])) + timedelta(days=int(mat_2_convert[i][j]%1)) - timedelta(days = 366)
            py_dates.iloc[i,j] = python_datetime
            i += 1
            if i == len(mat_2_convert): #if i row index reaches maximum, break
                break
        j += 1   #column index
        
    return py_dates

def normal_data2(df,date):
    """
    Normalize dataframe values to range [0,1]
    """
    #x= df.values
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)) 
    #Here we fit the scaler 
    scaler= scaler.fit(df[df.index[0]:date])
    #Here we transform the data
    x_scaled = scaler.transform(df)
    df = pd.DataFrame(x_scaled,index=df.index, columns=df.columns)
    return scaler,df
#%%
def create_mat_satbin(gps_obs,sat_obs,time_gps_obs,time_sat_obs):
    """Create binary satellite-gps matrix with shoreline data """
        
    #Create observations dataframes. Note sat and gps obs  have != timevecs
    df_gps_obs= pd.DataFrame(data= gps_obs)
    df_sat_obs= pd.DataFrame(data= sat_obs)

    #Replace zeros with nans
    df_gps_obs.replace(0, np.nan, inplace=True)
    df_sat_obs.replace(0, np.nan, inplace=True)
    
    yout_df=pd.DataFrame()
    satbin_df=pd.DataFrame()    
    # Loop over the column names in the DataFrame
    for column in df_gps_obs.columns:
        # Ensure the column exists in both DataFrames
        if column in df_sat_obs:
    
            #Create observations dataframes. Note sat and gps obs have != timevecs
            #slice dfs, only column of interest, drop nans in time vec
            time_gps_obs_col= time_gps_obs.iloc[:,column].dropna()
            time_sat_obs_col= time_sat_obs.iloc[:,column].dropna()
            
            #Transform to dataframes, drop nans in var vecs
            tmp_gps_obs= pd.DataFrame(data= gps_obs[:,column],columns=['yout_'+str(column)])
            tmp_gps_obs.dropna(inplace=True)
            tmp_sat_obs= pd.DataFrame(data= sat_obs[:,column],columns=['yout_'+str(column)])
            tmp_sat_obs.dropna(inplace=True)
            
            #Set index
            tmp_gps_obs = tmp_gps_obs.set_index(time_gps_obs_col)
            tmp_sat_obs = tmp_sat_obs.set_index(time_sat_obs_col)
            
            # Add a binary column to indicate the origin of each datum
            tmp_gps_obs['satbin_'+str(column)] = 0  # Origin 0 for data from df1
            tmp_sat_obs['satbin_'+str(column)] = 1  # Origin 1 for data from df2
            
            mat_out = pd.concat([tmp_gps_obs,tmp_sat_obs], axis=0)
            mat_out=mat_out.sort_index(axis=0)
            
            #Store results
            temp_df= pd.DataFrame({f'{column}': mat_out['yout_'+str(column)]})
            yout_df= pd.concat([yout_df, temp_df], axis=1)
    
            temp_df= pd.DataFrame({f'{column}': mat_out['satbin_'+str(column)]})                
            satbin_df= pd.concat([satbin_df, temp_df], axis=1)
            
    return(yout_df,satbin_df)

#%%

# Create matrix of inputs for DL models
def create_mat_in(mat_waves,mat_out,mat_mei):
    """ Create matrix of inputs for DL models. 
    This includes creating the input variables:
        daycount, max_Hs, e_mean
        FEx_m,  FEy_m"""
     
    #Merge dataframes based on indexes
    mat_in = mat_waves.merge(mat_out,how='left', left_index=True,
                             right_index=True)
    mat_in.index= mat_in.index.astype("datetime64[ns]")
    #Create s vector with only shoreline data
    # s = mat_in.yout.dropna()
    # #declare empty list
    # day_count=[]
    # max_Hs=[]
    # e_mean=[]
    # FEx_m=[] ; FEy_m=[] 
    # for i in range(len(s)):
    # #Slice of the input matrix that is between the
    # #entrance 0 and entrance 1 of the shoreline S series
    #     if i == 0:
    #         ini_idx= s.index[0] - pd.DateOffset(months=1)
    #         idx_list=mat_in.loc[ (mat_in['Hs'].index > ini_idx) &
    #                         (mat_in['Hs'].index <= s.index[i])]
    #     # Highest Hs in the gap
    #     else: 
    #         idx_list=mat_in.loc[ (mat_in['Hs'].index > s.index[i-1]) &
    #                     (mat_in['Hs'].index <= s.index[i])]
    #     #Highest Hs in the previous month
    #     # else: 
    #     #     ini_idx= s.index[i] - pd.DateOffset(months=1)
    #     #     idx_list=mat_in.loc[ (mat_in['Hs'].index > ini_idx) &
    #     #                 (mat_in['Hs'].index <= s.index[i])]
            
    #     #where did the maximum happened
    #     idx_max=idx_list.Hs.idxmax()
    #     #the actual maximum
    #     max_s=idx_list.Hs.max()
    #     #calculate elapsed time between maximum and observation
    #     e_t= s.index[i]- idx_max 
    #     #Calculate the wave energy of every individual entry in the gap 
    #     E_i= 1/8 * 9.81 *1025 * (idx_list.Hs ** 2)
    
    #     #Taken from Charline Dalinghaus thesis (2015)
    #     #Calculate wave direction
    #     #FE= E_i*((1.56*idx_list.Tp)/2) #Fluxo de energia 
    #     FE= E_i*((9.8 * 8)/2) #Fluxo de energia 
        
    #     #Direção do Fluxo de Energia
    
    #     FEx= FE*idx_list.Wvx # Calcula o fluxo na direção x 
    #     FEy= FE*idx_list.Wvy; # Calcula o fluxo na direção y 
    
    #     #Mean
    #     #Append days into new list n
    #     day_count.append(e_t.days)
    #     max_Hs.append(max_s)
    #     e_mean.append(E_i.mean())
    #     FEx_m.append(FEx.mean())
    #     FEy_m.append(FEy.mean())  
    
    #Here could go a plotting function for model inputs
    #Append newly generated data
    mat_in.dropna(inplace=True)
    # mat_in.insert(loc = 4, column = 'day_count',  value = day_count)
    # mat_in.insert(loc = 4, column = 'max_Hs',     value = max_Hs)
    # mat_in.insert(loc = 4, column = 'mean_waveE', value = e_mean)
    # mat_in.insert(loc = 4, column = 'FEx_m',      value = FEx_m)
    # mat_in.insert(loc = 4, column = 'FEy_m',      value = FEy_m)
    #Merge dataframes based on indexes
    # mat_mei = mat_in.merge(mat_mei,how='left', left_index=True, right_index=True)
    # mat_in.insert(loc = 4, column = 'mei',        value = mat_mei.MEI)
    
    mat_in.index.names = ['Datetime']
    
    return mat_in


# For loop and temporal merges

#Plot
# fig, axs = plt.subplots(figsize=(8, 6))

# axs.scatter(x=df_sat_obs.index, y=df_sat_obs.iloc[:,0], marker='.',
#                      color='k',label='satellite')
# axs.scatter(x=df_gps_obs.index, y=df_gps_obs.iloc[:,0], marker='*',
#                      color='r',label='gps')

# # Iterate over columns and plot each column with its name as the label
# for column in df_cosmos:
#     axs.plot(df_cosmos[column], label=column)

# # Set labels and title
# axs.set_xlabel('Time') 
# axs.set_ylabel('Shoreline displacement')
# axs.set_title('Transect number: ' + str(transect_id[t_id][0]))

# # Show the plot
# plt.legend()
# plt.show()

#%%

def wave_transform(df_tp, df_dir):
    """We need to transform Wave dir into its components x, and y in a matrix way
    #Each column represents the wave location that matches the shoreline transect """


    
    #First calculate wave length for deep water
    df_wl= (9.8 * np.power(df_tp,2)) / (2*math.pi) 
    #Wave_velocity= 1/Tp * Wave_length
    df_wv= (1/df_tp) * df_wl
     #Transform to x,y components
     # Convert to radians.
    df_rad=df_dir*np.pi / 180
     # Calculate the wind x and y components.
    df_wvx=df_wv*np.cos(df_rad)
    df_wvy=df_wv*np.sin(df_rad)
    
    return(df_wvx,df_wvy)

def in_out_mat_4_models(t_id, df_hs,df_tp,df_wvx,df_wvy,
                        yout_df,satbin_df,mei):
    """For a given shoreline/wave location, 
    create input and output matrix to be used directly for DL models run.
    We here define which input variables go into DL models. The best performance
    is achieved by dropping:
        -Tp
        -satbin
        -FEy_m, FEx_m
        -mei
    """
    # # Loop over the column names, i.e. number of transects
    #Create unified wave matrix, for transect of interest
    mat_waves= pd.concat([df_hs.iloc[:,t_id], df_tp.iloc[:,t_id],
                          df_wvx.iloc[:,t_id],df_wvy.iloc[:,t_id]],axis=1)
    mat_waves.columns=['Hs', 'Tp', 'Wvx', 'Wvy']
    
    #WITH SATBIN included
    #Create unified output matrix, for transect of interest
    # mat_out= pd.concat([yout_df.iloc[:,t_id], satbin_df.iloc[:,t_id] ],
    #                    axis=1)
    # mat_out.columns=['yout','satbin']    

    
    #Without satbin
    #Create unified output matrix, for transect of interest
    mat_out= pd.concat([yout_df.iloc[:,t_id] ],
                       axis=1)
    mat_out.columns=['yout']    


    #Creating input matrix
    mat_in= create_mat_in(mat_waves,mat_out,mei)
    #Drop rows with nan values, 
    mat_in=mat_in.dropna(axis=0)
    
    #Traditional 3
    # mat_in.drop('Hs', inplace=True, axis=1)
    mat_in.drop('Tp', inplace=True, axis=1)
    # mat_in.drop('Wvx', inplace=True, axis=1)
    # mat_in.drop('Wvy', inplace=True, axis=1)
    
    # # Condition only gps: Keep rows where 'satbin' is 0
    # condition = mat_in['satbin'] == 0
    # # Filtered DataFrame
    # mat_in = mat_in[condition]
    # #Then drop satbin binary column
    # mat_in.drop('satbin', inplace=True, axis=1)  

    
    #Wave energy related
    # mat_in.drop('mean_waveE', inplace=True, axis=1)
    # mat_in.drop('FEy_m', inplace=True, axis=1)
    # mat_in.drop('FEx_m', inplace=True, axis=1)
    
    #day count and maximum Hs
    # mat_in.drop('max_Hs', inplace=True, axis=1)
    # mat_in.drop('day_count', inplace=True, axis=1)
    
    #Slice data before 1995, waves are another source ,
    #apparently wave dir comes from a look-up table
    # mat_in= mat_in['1995-01-01' :]
    
    #Climate index
    # mat_in.drop('mei', inplace=True, axis=1)
    
    mat_out= pd.DataFrame(mat_in.yout.values, index=mat_in.index)
    
    return(mat_in,mat_out)


























