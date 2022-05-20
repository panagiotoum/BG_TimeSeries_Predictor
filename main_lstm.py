# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import math
from matplotlib import pyplot
import scipy.io
from scipy import stats
from numpy import array
import argparse

# Path
from pathlib import Path
import os

# Sklearn
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

# Torch
import torch
import torch.nn as nn
from torch.autograd import Variable

# Keras
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.layers import Bidirectional

# Preprocessing
from preprocessing.data_preprocessing import data_pre, sliding_windows
from metrics.metrics import rmse,mard,cc


# arguments

parser = argparse.ArgumentParser(description='Keras LSTMs architectures')
parser.add_argument('--data_type', type=str,
                    default='new',
                    help='Ohio dataset, new: OhioData2020, old:OhioData2018')
parser.add_argument('--model_name', type=str,
                    default='lstm',
                    help='lstm or bilstm')
parser.add_argument('--ph', type=int, default = 6,
                    help='Prediction Horizon (ph), ph = 6 (30 min) or 12 (60 min)')
parser.add_argument('--data_path', type=str,
                    default='/media/maria/5d4f6b8a-2a39-41cf-957e-6a790108f04a/Ohio Data/Ohio2020/',
                    help='Path to load the data')
parser.add_argument('--save_path', type=str,
                    default='/home/maria/anaconda3/envs/thesis2/diabetes/',
                    help='Path to save the models and plots')
args = parser.parse_args()

# the models parameters are default from the code (lack of time :/)
model_name = args.model_name
data_type = args.data_type
save_path = args.save_path
data_path = args.data_path


# Read the data and save into a dictionary
if data_type == 'new':
    patients = [540,544,552,567,584,596]
    dict_train={}
    dict_test={}
 
    for i,j in enumerate(patients):
        train = pd.read_csv(data_path+'train/'+str(j) +'-ws-training_processed.csv')
        dict_train[str(j)]=train
        test = pd.read_csv(data_path+'test/'+str(j) +'-ws-testing_processed.csv')
        dict_test[str(j)]=test
elif data_type == 'old':
    patients = [559,563,570,575,588,591]

    dict_train={}
    dict_test={}

    for i,j in enumerate(patients):
        train = pd.read_csv(data_path+'train/'+str(j) +'-ws-training_processed.csv')
        dict_train[str(j)]=train
        test = pd.read_csv(data_path+'test/'+str(j) +'-ws-testing_processed.csv')
        dict_test[str(j)]=test

else:
    print("False data type")



rmse_l = []
mard_l = []
cc_l = []
p_l = []
for i in patients:
    training_data_x, training_data_y, _ = data_pre(dict_train[str(i)])
    test_data_x, test_data_y, min_max_scaler = data_pre(dict_test[str(i)])
    
    seq_length = 24
    n_predictions = args.ph
    x_train, y_train = sliding_windows(training_data_x, training_data_y, seq_length, n_predictions)
    x_test, y_test = sliding_windows(test_data_x, test_data_y, seq_length, n_predictions)

    trainX = Variable(torch.Tensor(np.array(x_train)))
    trainY = Variable(torch.Tensor(np.array(y_train)))
    #print(trainX.shape)
    #print(trainY.shape)

    testX = Variable(torch.Tensor(np.array(x_test)))
    testY = Variable(torch.Tensor(np.array(y_test)))
    
    ###
    # Calling `save('my_model')` creates a SavedModel folder `my_model`.
    model_dir = Path(save_path+str(i)+"/"+model_name+"/ph"+str(n_predictions))
    model_dir.mkdir(parents=True,exist_ok=True)
    #model.save(model_name)
    checkpoint_path = save_path+str(i)+"/"+model_name+"/ph"+str(n_predictions)+"/"+model_name+".ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    opt = keras.optimizers.Adam(learning_rate=0.001)
    
    if model_name == 'lstm':
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(24, 4),return_sequences = False))
        model.add(Dense(20, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(np.shape(trainY)[1], activation = 'relu'))
        model.compile(optimizer="adam", loss="mse")
        model.summary()
        
    elif model_name == 'bi-lstm':
        model = Sequential()
        model.add(LSTM(4, activation='relu', input_shape=(24, 4),return_sequences = True))
        model.add(Bidirectional(LSTM(8, return_sequences=False), input_shape=(24,40)))
        model.add(Dense(8, activation = 'relu'))
        model.add(Dense(4, activation = 'relu'))
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(4, activation = 'relu'))
        model.add(Dense(np.shape(trainY)[1], activation = 'relu'))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
    
    model.fit(trainX.numpy(), trainY.numpy(), epochs=100, batch_size = 1,callbacks=[callback,cp_callback])
  
    ###############################################################################################################

    # make predictions
    testPredict_scaled = model.predict(testX.numpy(), verbose = 0)
    
    # de-normalized data
    testY = min_max_scaler.inverse_transform(testY)
    testPredict = min_max_scaler.inverse_transform(testPredict_scaled)

    ## Evaluation Performance
    rmse_test = rmse(testPredict[:,0],testY[:,0],testY.shape[0])
    rmse_l.append(rmse_test)
    print("RMSE for PersonID "+ str(i) + ":",rmse_test)

    mard_test = mard(testPredict[:,0],testY[:,0],testY.shape[0])
    mard_l.append(mard_test)
    print("MARD for PersonID "+ str(i) + ":",mard_test)

    cc_test, p_value = scipy.stats.pearsonr(testPredict[:,0],testY[:,0])
    cc_l.append(cc_test)
    p_l.append(p_value)
    print("CC for PersonID "+ str(i) + ":",cc_test*100)
    
    ### plot
    plt.figure(figsize=((12,6)))
    plt.plot(testPredict[:,0])
    plt.plot(testY[:,0])
    plt.suptitle('Time-Series Prediction')
    plt.legend()
    plt.savefig(save_path+str(i)+"/"+model_name+"/ph"+str(n_predictions)+"/"+"plot.png")
    plt.show()

    ###############################################################################################################
# save results in csv file.

# dictionary of lists  
dict = {'rmse': rmse_l, 'mard': mard_l, 'cc': cc_l, 'p-value':p_l}  
       
df = pd.DataFrame(dict) 
#display(df)

df.to_csv('metrics_'+model_name+'_ph'+str(n_predictions)+'_'+data_type+'.csv') 
