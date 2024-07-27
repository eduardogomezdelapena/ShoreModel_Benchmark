#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:18:58 2024
Script to run the CNN-LSTM models for Ocean Beach, it was originally
hosted in /home/egom802/Documents/GitHub/ANNs_4_USCoasts/OceanBeach/data
This script's ouputs are used in metrics.py
@author: egom802
"""

import scipy.io
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import os
import random
import numpy as np
import math
import scipy.stats

from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv1D, MaxPooling1D

import keras

from keras.layers import LSTM
from keras.layers import RepeatVector,TimeDistributed

rt_dir='/home/egom802/Documents/GitHub/ANNs_4_USCoasts/OceanBeach/1run_models'
os.chdir(rt_dir)
from utils.splits import normal_data, split_sequences
from utils.loss_errors import index_mielke
#from utils.loss_errors import  pearson_mse_loss, combined_loss

from utils.loss_errors import correlation_coefficient_loss

pr_dir='/home/egom802/Documents/GitHub/ANNs_4_USCoasts/OceanBeach/data'
os.chdir(pr_dir)

from preprocess import matnum_2_py, matnum_2_py_mat
from preprocess import normal_data2,create_mat_satbin, wave_transform
from preprocess import in_out_mat_4_models


from tensorflow.keras import backend as K
from tensorflow.python.ops import math_ops

#%%

alphas= np.float32(np.linspace(0,1,11))
alphas= np.float32([1])
rmse_list = []; pear_list = []
#alpha is mileke loss or pearson loss coefficient

#Year to stop training (dev set included)
# year_stop_train='2003'

years = [str(year) for year in range(2003, 2015)]
# years = ["2015"]

for year_stop_train in years:
        
    for alpha in alphas:
        
        print('alpha:'+str(alpha))
        #Working string to save files and results
        
        #Dont forget to change the loss function
        wk_str= str(alpha)+'mielke_'+str(round(1-alpha,1))+'mse_resampled_4days_'+year_stop_train+'_onlysat' ##################### Change here
        # wk_str= str(alpha)+'mielke_'+str(round(1-alpha,1))+'mse_7980' 
        
        #Set loss function for training anns
        # loss= pearson_mse_loss
        # loss= combined_loss
        
        #Steps that the ANN can look back
        # n_steps_in, n_steps_out =52,1
        # n_steps_in, n_steps_out =22,1
        n_steps_in, n_steps_out =90,1
        #Defininf train, dev, and test splits
           
        #2years dev
        # train_date='2012-12-30'
        # devinit_date= '2013-01-01'
        # dev_date= '2014-12-31'
    
        train_date=str(int(year_stop_train)-2) +'-12-31'
        devinit_date= str(int(year_stop_train)-1) +'-01-01'
        dev_date= str(int(year_stop_train)-1) + '-12-31'
    
        # test_init= '2015-01-01'
        # test_end= '2020-12-31'
        
        #create directories to save plots
        bs='/home/egom802/Documents/GitHub/ANNs_4_USCoasts/OceanBeach/2gen_plots/figures/individual_TD/'
        directory_name = bs+ wk_str
        
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)
            os.mkdir(directory_name+'/time_series')
        # Remember that from 2015 on, CoSMoS is in Forecast mode
        dir_cosmos='/home/egom802/Documents/GitHub/ANNs_4_USCoasts/OceanBeach/data/Originals/inputs/'
        #%%
         
        @tf.function()
        def combined_loss(y_true, y_pred, alpha=alpha):
            """ Mielke index 
            if pearson coefficient (r) is zero or positive use kappa=0
            otherwise see Duveiller et al. 2015
            """
            x = y_true
            y = y_pred
            mx = K.mean(x)
            my = K.mean(y)
            xm, ym = x-mx, y-my
        
            diff= math_ops.squared_difference(y_true, y_pred) 
            d1= K.sum(diff)
            d2= K.var(y_true)+K.var(y_pred)+ tf.math.square(
                (K.mean(y_true)-K.mean(y_pred)))
           
            if correlation_coefficient_loss(y_true, y_pred) < 0:
                kappa = tf.multiply(K.abs( K.sum(tf.multiply(xm,ym))),2)
                loss= 1-(  ( d1* (1/K.int_shape(y_true)[1])  ) / (d2 +kappa))
            else:
                loss= 1-(  ( d1* (1/K.int_shape(y_true)[1])  ) / d2 )
                
            
            def mse(y_true, y_pred):
                # Mean Squared Error component
                mse = tf.reduce_mean(tf.square(y_true - y_pred))
                return mse
            
            mse_loss= mse(y_true, y_pred)
            mielke_loss= 1- loss
            # return 1- loss
            return alpha * mielke_loss  + (1 - alpha) * mse_loss
        
    
    #%%
            
        loss= combined_loss
        #%%
        mat_contents = scipy.io.loadmat(dir_cosmos+'Shoreline_and_Cosmos_performance.mat')
        
        # Accessing its keys
        structure_keys = mat_contents['dataStruct'].dtype.names
        
        # Print the keys
        print("Fields in cosmos & shoreline data MATLAB structure:")
        for key in structure_keys:
            print(key)
            
        # Accessing the struct. The struct name in MATLAB becomes a key in the dictionary.
        my_struct = mat_contents['dataStruct']
        #% Transform matlab date numbers to actual dates
        time_cosmos = my_struct['time_cosmos'][0, 0]  # Accessing the first element
        time_cosmos = matnum_2_py(time_cosmos)
        
        #% #time gps and time sat are matrixes, meaning there is a different time vector for
        #each transect with gps amd sat observations
        time_gps_obs = my_struct['time_gps_obs'][0, 0] 
        time_sat_obs = my_struct['time_sat_obs'][0, 0] 
        
        time_gps_obs= matnum_2_py_mat(time_gps_obs)
        time_sat_obs= matnum_2_py_mat(time_sat_obs)
        
        #Load transects
        transect_id = my_struct['transect_id'][0, 0]  
        
        #Load and plot gps & satellite observations
        gps_obs = my_struct['gps_obs'][0, 0] # [:,t_id] 
        sat_obs = my_struct['sat_obs'][0, 0] # [:,t_id] 
        
        #Load CoSMos results
        sat_cosmos = my_struct['sat_cosmos'][0, 0] #[:,t_id] 
        gps_cosmos = my_struct['gps_cosmos'][0, 0] #[:,t_id] 
        gps_sat_cosmos = my_struct['gps_sat_cosmos'][0, 0] #[:,t_id] 
        
        # #Create CoSMoS results dataframe 
        # d = {'sat_cosmos': sat_cosmos, 'gps_cosmos': gps_cosmos,
        #       'gps_sat_cosmos':gps_sat_cosmos}
        
        # df_cosmos=pd.DataFrame(index=time_cosmos, data= d)
        # Remember that from 2015 on, CoSMoS is in Forecast mode
        #% Loop through shoreline obs vars to make satbin matrix
        
        #%% Create binary satellite-gps matrix with shoreline data           
        yout_df,satbin_df = create_mat_satbin(gps_obs,sat_obs,time_gps_obs,time_sat_obs)
        
        #Transform index to datetime
        yout_df.index=pd.to_datetime(yout_df.index)
        
        # yout_df.iloc[:,58].sort_index().plot()
        
        #Filter to only satellite observations
        yout_df= yout_df[satbin_df == 1]
        
        #lets resample
        yout_df=yout_df.resample('4D').interpolate(method='linear')
        
        
        
        # yout_df=yout_df.resample('W').interpolate(method='linear')
        
        # yout_df['0'].plot(marker='o')
        
        #%% Save satellite binary matrix
        
        # #Sort indexes in chronological time
        # satbin_df.columns=  [int(col) for col in transect_id] # Converting string to int column names
        # satbin_df.sort_index(inplace=True)
        # satbin_df.to_csv('../1run_models/output/satbin_all_transects.csv',index=True)  
        
        #%%
        # yout_df.sort_index().plot()
        
        # specific_year = '2000'
        # yout_df= yout_df[specific_year:]
        
        #%% Load waves
        
        mat_contents = scipy.io.loadmat(dir_cosmos+'Waves_Cosmos.mat')
        # Access its keys
        structure_keys = mat_contents['dataStruct'].dtype.names
        # Print the keys
        print("Fields in the waves MATLAB structure:")
        for key in structure_keys:
            print(key)
        # Accessing the struct. The struct name in MATLAB becomes a key in the dictionary.
        wave_struct = mat_contents['dataStruct']
        # Load time, transform it to python readable
        time_waves = wave_struct['time'][0, 0]  # Accessing the first element
        time_waves = matnum_2_py(np.transpose(time_waves))
        #Load wave vars
        Hs = np.transpose(wave_struct['Hs'][0, 0])  
        Dir = np.transpose(wave_struct['Dir'][0, 0])
        Tp = np.transpose(wave_struct['Tp'][0, 0])
        
        mop_id=[]
        for element in range(0,np.size(wave_struct['MOP_ID'][0, 0])):
            ind=  str(wave_struct['MOP_ID'][0, 0][element][0][0])
            mop_id.append(ind)
            
        #Transform & process wave vars
        df_hs,df_tp,df_wvx,df_wvy= wave_transform(time_waves,Hs,Tp,Dir)
        
        # df_wvy.iloc[:,1].plot()
        #%% Load MEI v2
        
        mei= pd.read_csv('../data/Originals/meiv2_processed.csv',index_col='Date', parse_dates=True)
        
        #resample a time series to a daily frequency while using the same value 
        #for all days within a given month,
        mei = mei.resample('D').ffill()
        os.chdir(rt_dir)
        
        #59 transects, from id 7958 to 8016
        
        df_rmse=pd.DataFrame()
        df_pear=pd.DataFrame()
        df_mielke=pd.DataFrame()
        df_std=pd.DataFrame()
        
        df_yresults=pd.DataFrame()
        df_obs_dl=pd.DataFrame()
        
        #Loop through all transects, create input and output matrix for DL models run
        #21 is transect 1979, 34 is 1992
        #21 is where wave data changes from SF to SM
        # for t_id in range(21,len(transect_id)):
        for t_id in range(22,len(transect_id)):
        # for t_id in range(22,23):
        # for t_id in range(0,22):
        
            #ts_number defined for plotting purposes
            ts_number=str(transect_id[t_id][0])
            print(ts_number)
        
            #For a given shoreline/wave location, 
            #create input and output matrix to be used directly for DL models run.
            #We here define which input variables go into DL models.The best performance
            #is achieved by dropping (i.e. not inputting):
            #    -Tp
            #    -satbin
            #    -FEy_m, FEx_m
            #    -mei
            mat_in,mat_out= in_out_mat_4_models(t_id, df_hs,df_tp,df_wvx,df_wvy,
                                                yout_df,satbin_df,mei)   
            
            #%%##################TRAIN, DEV AND TEST SPLITS################################
            #Manually split by date    
        
            # train_date='2009-12-31'    
            #%%##########################DATA NORMALIZATION################################
            _, mat_in = normal_data2(mat_in,train_date)
            scaler, mat_out_norm = normal_data(mat_out,train_date)
            #%%##################TRAIN, DEV AND TEST SPLITS################################
            #Manually split by date 
            train = mat_in[mat_in.index[0]:train_date].values.astype('float32')
            #Development set (2 years before the test set)
            # devinit_date=pd.to_datetime(train_date) + timedelta(days=1)
            # devinit_date= str(devinit_date.strftime("%Y-%m-%d"))
        
            # devinit_date= '2010-01-01'
            # dev_date=pd.to_datetime(date_forecast) - timedelta(days=1)
            # dev_date= str(dev_date.strftime("%Y-%m-%d"))
        
            dev= mat_in[devinit_date:dev_date].values.astype('float32')
            #%%
        
            # interesting performance
            # n_steps_in, n_steps_out =18,1
            #Test set, depends on study site
            test_init= '2015-01-01'
            test_end= '2020-12-31'
            #Calculate the actual index, taking into acount that the n_steps_in blocks 
            #need a first n_steps_in block to work
            test_init=mat_in.loc[mat_in.index <  test_init].tail(n_steps_in-1).index.min()
        
            #Calculate the actual index, the immediate available 
            test_end=mat_in.loc[mat_in.index > test_end].head(1).index.max()
            test = mat_in[test_init:test_end].values.astype('float32')
            # test = mat_in[test_init:mat_in.index[-1]].values.astype('float32')
            #%%############################################################################
            #From pandas to array, HERE WE SEPARATE THE INPUTS FROM THE Y_OUTPUT
            # split a multivariate sequence into samples 
            train_x, train_y = split_sequences(train, n_steps_in, n_steps_out)
            dev_x, dev_y = split_sequences(dev, n_steps_in, n_steps_out)
            test_x, test_y = split_sequences(test, n_steps_in, n_steps_out)
            # # the dataset knows the number of features, e.g. 2
            n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[
                2], train_y.shape[1]
            #%%###################Define Loss functions: NEURAL NETWORK##################
            def set_seed(seed):
                tf.random.set_seed(seed)
                os.environ['PYTHONHASHSEED'] = str(seed)
                np.random.seed(seed)
                random.seed(seed)
            
            #%%
            # loss='mse'
            # loss=td_distance_loss
            # loss=real_td_distance_loss
            # loss= mielke_loss
            # loss= combined_loss  
        
            min_delta= 0.001
            def cnn_custom(train_x, train_y, dev_x, dev_y, cfg):
                print("--------------------------------")
                print("Model:", cfg)
                set_seed(33)
                # define model    # create configs
                n_filters, n_kernels, n_mem,n_drop,n_epochs,n_batch = cfg    
                n_epochs=50
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
                
                # optimizer= keras.optimizers.Adam(learning_rate=1e-2)
                
                # model.compile(optimizer=optimizer, loss=loss)
                
                model.compile(optimizer='adam', loss=loss)
                # fit model
                es = EarlyStopping(patience=20, verbose=2, min_delta=min_delta, 
                                   monitor='val_loss', mode='auto',
                                   restore_best_weights=True)
                history= model.fit(train_x, train_y, validation_data=(dev_x, dev_y),
                                     batch_size=n_batch, epochs=n_epochs, verbose=2,
                                     callbacks=[es])  
                return model, history
            
            #%%############################################################################
            # Load Grid Search Hyperparameters
            scores = list()   
            cfg_list=pd.read_csv('./hyp/10best_hyp_Wavesonly_Mielke_Hybrid.csv')
            cfg_list= cfg_list[["f","k","m","D","e","b"]]
            cfg_list= cfg_list.values.tolist()
            for i in range(len(cfg_list)):     
                for element in range(len(cfg_list[i])):
                    #Position where dropout percentage is
                    if element != 3:
                        cfg_list[i][element] = int(cfg_list[i][element]) 
            #%%#Run model configurations in loop###########################################
            #Predefine empty dataframe
            plot_date = pd.to_datetime(test_init) #+ timedelta(days=n_steps_in-1)
            #To have the next value, regardless of the time gap
            tg=mat_in.loc[mat_in.index >  test_init].head(n_steps_in-1).index.max()
            ############################################
            # plot_date= str(plot_date.strftime("%Y-%m-%d"))
            
            yresults= pd.DataFrame(index=mat_in[ tg : test_end].index,
                                    columns=[ts_number])
            
            # yresults= pd.DataFrame(index=mat_in[ tg : test_end].index,
            #                         columns=['ann1','ann2','ann3','ann4','ann5',
            #                                 'ann6','ann7','ann8','ann9','ann10'])
            
            #Rescale target shoreline time series
            testY = scaler.inverse_transform(test_y)
            # train_x=train[:,0:4]
            # train_y=train[:,4]
            for (index, colname) in enumerate(yresults):
                print('Model number:' + str(index))
                #Train model with hyp config from config list
                model,_ = cnn_custom(train_x, train_y, dev_x, dev_y, cfg_list[index]) 
                testdl = model.predict(test_x)     
                # yresults.iloc[:,index]= scaler.inverse_transform(testdl)
                
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
            
            #%% EXPORT ENSEMBLE
            #This could be postmodel run, ideally
            
            df_yresults= pd.concat([df_yresults,yresults],axis=1)
            
            df_obs_dl=pd.concat([df_obs_dl,pd.Series(np.squeeze(testY),name=ts_number, index=pd.to_datetime(yresults.index))],axis=1)
        
            #Metrics 
            rmse_arr=np.array([math.sqrt(mean_squared_error(yresults[colname].values,testY)) for (index, colname) in enumerate(yresults)])
            df_rmse=pd.concat([df_rmse, pd.Series(rmse_arr, name=ts_number)], axis=1)
            
            pear_arr=np.array([scipy.stats.pearsonr(yresults[colname].values,testY[:,0])[0] for (index, colname) in enumerate(yresults)])
            df_pear=pd.concat([df_pear,pd.Series(pear_arr, name=ts_number)], axis=1)
            
            mielke_arr=np.array( [index_mielke(yresults[colname].values,testY[:,0]) for (index, colname) in enumerate(yresults)])
            df_mielke=pd.concat([df_mielke,pd.Series(mielke_arr, name=ts_number)], axis=1)
            
                
            # #Need to add standard deviation
            std_arr= np.array([np.std(yresults[colname].values) for (index, colname) in enumerate(yresults)])
            df_std=  pd.concat([df_std,pd.Series(std_arr, name=ts_number)], axis=1)
            
            #%%
            yresults2=yresults
            yresults2['Obs']=testY
            
            # #monthly resample
            # yresults2 = yresults2.resample('ME').mean()
            
            # # Fill missing values by forward filling
            # yresults2 = yresults2.interpolate(method='linear')
            
            # fs_tk= 14; 
            #Set obs colors to red
            color_dict = {'Obs': 'midnightblue'}
            
            plt.rcParams.update({'font.size': 28})
            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            fig,ax = plt.subplots(figsize=(3750*px,1875*px))
            #Set ann colors to gray
            yresults2.plot(ax=ax, marker='o', color=[color_dict.get(x, 'darksalmon') for x in yresults2.columns])
            
            plt.annotate("RMSE: "+str(round(np.mean(rmse_arr),2)), xy=('2015-06-01', 50))
            plt.annotate("r: "+str(round(np.mean(pear_arr),2)), xy=('2016-06-01', 40))
            plt.title('DL model (CNNs): Ocean Beach, transect #' +ts_number)
            # plt.annotate("pear: "+str(round(np.mean(pear_arr),2)), xy=('2019-12-01', 135))
            # plt.annotate("mielke: "+str(round(np.mean(mielke_arr),2)), xy=('2019-12-01', 140))
            ax.set_xlim([datetime.date(2015, 1, 1), datetime.date(2020, 12, 31)])
            # ax.set_ylim([-110,80])
            # ax.tick_params(axis='both', which='major', labelsize=fs_tk)
            ax.legend(loc="lower left",ncol=6, fontsize=20)
            # plt.yticks(np.arange(-100, 80, 20))
            plt.ylabel('Cross-shore displacement ' r'$[m]$')
            plt.xlabel('Year')
            plt.xticks(rotation=0)
            #%%
            #Uncomment to save plot
            plt.savefig('../2gen_plots/figures/individual_TD/'+wk_str+'/time_series/ts_transect_'+str(ts_number)+'.png')#,
                        #bbox_inches='tight',dpi=300)
                
            plt.close()
            #%%
            #Export time series
            # yresults2.to_csv('./output/metric_analysis/all_transects/hybann1_'+ts_number+ '.csv',index=True)
            
           
        df_yresults.to_csv('./output/y_ann_all_transects_'+wk_str+'.csv',index=True)  
        
        #Sort indexes in chronological time
        df_obs_dl.sort_index(inplace=True)
        df_obs_dl.to_csv('./output/dl_obs_all_transects_'+wk_str+'.csv',index=True)  
        
        # df_obs_dl.columns = [int(col) for col in df_pear.columns] # Converting string to int column names
        
        # df_obs_dl.index=pd.to_datetime(df_obs_dl.index)
        
        #Shoreline observations    
        df_obs_dl.plot()
        #Results of deep learning models
        df_yresults.plot()
        #%%
        # df_rmse.index= alpha
        df_rmse.index=['alpha_'+str(alpha)]
        rmse_list.append(df_rmse)
        
        df_pear.index=['alpha_'+str(alpha)]
        pear_list.append(df_pear)
        # #When loop finish, export metrics  
        
        
    # Concatenate all DataFrames in the list into one DataFrame
    final_rmse = pd.concat(rmse_list, ignore_index=False)
    final_pear = pd.concat(pear_list, ignore_index=False)
        
    final_rmse.to_csv('./output/final_rmse_'+wk_str+'.csv',index=True)  
    final_pear.to_csv('./output/final_pear_'+wk_str+'.csv',index=True)  
    
    
    
    final_pear.mean(axis=1)
    
    final_rmse.mean(axis=1)


    