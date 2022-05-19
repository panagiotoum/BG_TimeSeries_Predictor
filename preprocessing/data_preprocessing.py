import pandas as pd
from sklearn import preprocessing
import numpy as np

# Data pre-processing
def data_pre(df):
    # fill missing values for basal
    # fill with the previous value
    df['basal'] = df['basal'].fillna(method='ffill')
    # fill with the next value
    df['basal'] = df['basal'].fillna(method='bfill')

    ## replace nan values of bolus and carbs with zero
    df['bolus'] = df['bolus'].fillna(0)
    df['carbInput'] = df['carbInput'].fillna(0)

    ##cbg
    ## replace missing cbg values with finger measure if exist
    if df['cbg'].isnull().values.any():
        index_cbg = df['cbg'].index[df['cbg'].apply(np.isnan)]
        index_finger = df['finger'].index[df['finger'].notnull()]
        common_index = np.intersect1d(index_cbg, index_finger)
        for i in common_index:
            df['cbg'].loc[i] = df['finger'].loc[i]

    ## replace other nan values with median method
    df[['cbg']] = df[['cbg']].fillna(df[['cbg']].median())

    ## keep the 'cbg','basal','carbInput','bolus' features.
    df = df[['cbg','basal','carbInput','bolus']]
    df = df.dropna()
    
    
    df_x = df
    df_y = df[["cbg"]]

    # normalized data with minmax
    x = df_x.values #returns a numpy array
    min_max_scaler_x = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler_x.fit_transform(x)
    df_x = x_scaled
    
    
    y = df_y.values #returns a numpy array
    min_max_scaler_y = preprocessing.MinMaxScaler()
    y_scaled = min_max_scaler_y.fit_transform(y)
    df_y = y_scaled

    #df.isnull().values.any()
    
    return df_x, df_y, min_max_scaler_y


# dataloading
#seq_lenght: number of timesteps/timewindow
#n_predictions: number of timesteps for prediction

# create sliding windows for training
def sliding_windows1(data_x,data_y, seq_length, n_predictions):
    x = []
    y = []

    for i in range(len(data_x)-seq_length-1):
        if (i+seq_length+n_predictions > len(data_x)-1) :
            break
        _x = data_x[i:(i+seq_length)]
        #_y = data[i+seq_length]
        _y = data_y[i+seq_length+n_predictions-1]
        #print(_y.shape)
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)
