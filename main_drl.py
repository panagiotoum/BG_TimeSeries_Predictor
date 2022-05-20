# imports
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import scipy.io
from scipy import stats
import argparse

# open AI gym env
import gym as gym
from gym import Env
from gym.spaces import Discrete, Box 
import gym.spaces as spaces

# DDPG Algorithm
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.ddpg import DDPG

# Preprocessing
from preprocessing.data_preprocessing import data_pre, sliding_windows
# metrics
from metrics.metrics import rmse,mard,cc

# Path
from pathlib import Path
import os

import warnings
# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# arguments

parser = argparse.ArgumentParser(description='Keras LSTMs architectures')
parser.add_argument('--data_type', type=str,
                    default='new',
                    help='Ohio dataset, new: OhioData2020, old:OhioData2018')
parser.add_argument('--model_name', type=str,
                    default='Noise',
                    help='Noise or NoNoise')
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
n_predictions = args.ph


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



class bg_env(gym.Env):
    def __init__(self, data_x,data_y):
        '''
        '''
        self.reward = 0
        self.data_x= data_x
        self.data_y= data_y
        self.x_dim = data_x.shape
        self.total_data = self.x_dim[0]
        self.data_pointer = 0
        self.state = self.prepare_state(self.data_x[self.data_pointer])

        low = np.zeros((self.x_dim[1]*self.x_dim[2]), dtype=np.float32)
        high = np.ones((self.x_dim[1]*self.x_dim[2]), dtype=np.float32)
        
        #self.observation_space = (level,expert,defect_id)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(np.zeros(1, dtype=np.float32),np.ones(1, dtype=np.float32),dtype=np.float32)
                      
    def observation_space(self):
        '''
        Return the Observation space
        '''
        return self.observation_space

    def action_space(self):
        '''
        Return the Action space
        '''
        return self.action_space

    def prepare_state(self,data):
        reshaped = np.reshape(data,(data.shape[0]*data.shape[1]))
        return reshaped
    
    def step(self,action):
        done = False
        
        if self.data_pointer>= self.total_data-1:
            done = True
        
        reward = -(abs(action - self.data_y[self.data_pointer]))
        self.reward += reward
        self.state = self.prepare_state(self.data_x[self.data_pointer])
        self.data_pointer +=1
              
        return self.state, reward, done, {}
    
    def reset(self):
        '''
        Resets the env by reseting the messages and the level of the state/observation
        '''
        self.reward = 0 
        self.data_pointer = 0
        self.state = self.prepare_state(self.data_x[self.data_pointer])
        return self.state

def prepare_my_state(data):
    reshaped = np.reshape(data,(data.shape[0]*data.shape[1]))
    return reshaped


## list for results
rmse_l = []
mard_l = []
cc_l = []
p_l = []

# training and evaluation for each patient seperately
for i in patients:
    training_data_x, training_data_y, _ = data_pre(dict_train[str(i)])
    test_data_x, test_data_y, min_max_scaler = data_pre(dict_test[str(i)])
    
    # time window
    seq_length = 24
    # prediction horizon
    
    # split sequences
    x_train, y_train = sliding_windows(training_data_x, training_data_y, seq_length, n_predictions)
    x_test, y_test = sliding_windows(test_data_x, test_data_y, seq_length, n_predictions)

    # create_new_dr
    model_dir = Path(save_path+str(i)+"/"+model_name+"/ph"+str(n_predictions))
    model_dir.mkdir(parents=True,exist_ok=True)

    # create env
    env = bg_env(x_train,y_train)
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    if model_name == 'Noise':
    # train model with noise
      model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
    elif model_name == 'NoNoise':
    # train moden without noise
      model = DDPG(MlpPolicy, env, verbose=1)
    
    # training
    model.learn(total_timesteps=20)
  
    ######################################################
    # predict in test set
    predictions = []
    for obs in x_test:
        my_obs = prepare_my_state(obs)
        action, _states = model.predict(my_obs)
        predictions.append([action[0]])
    
    # denormalized the data
    y_test = min_max_scaler.inverse_transform(y_test)
    predictions = min_max_scaler.inverse_transform(predictions)

    ########################################################

 ########################################################
    # metrics
    rmse_test = rmse(predictions.flatten(),y_test.flatten(),len(predictions))
    rmse_l.append(rmse_test)
    print("RMSE for PersonID "+ str(i) + ":",rmse_test)

    mard_test = mard(predictions.flatten(),y_test.flatten(),len(predictions))
    mard_l.append(mard_test)
    print("MARD for PersonID "+ str(i) + ":",mard_test)

    cc_test, p_value = scipy.stats.pearsonr(predictions.flatten(),y_test.flatten())
    cc_l.append(cc_test)
    p_l.append(p_value)
    print("CC for PersonID "+ str(i) + ":",cc_test*100)
    
    # plot
    plt.figure(figsize=((12,6)))
    plt.plot(predictions)
    plt.plot(y_test)
    plt.suptitle('Time-Series Prediction')
    plt.legend()
    plt.savefig(save_path+str(i)+"/"+model_name+"/ph"+str(n_predictions)+"/"+"plot.png")
    #plt.show()   
    ###############################################################################################################
# save results in csv file.

# dictionary of lists  
dict = {'rmse': rmse_l, 'mard': mard_l, 'cc': cc_l, 'p-value':p_l}  
       
df = pd.DataFrame(dict) 
#display(df)

df.to_csv('metrics_'+model_name+'_ph'+str(n_predictions)+'_'+data_type+'.csv') 
