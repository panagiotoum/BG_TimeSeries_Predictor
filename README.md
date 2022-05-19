# Blood Glucose (BG) Timeseries Forecasting with Deep Neural Networks and Deep Reinforcement Learning

## Dataset: Ohio Dataset
In the Ohio data, we provide you with 8 features that could be used to predict future blood glucoe values. The data is extracted from the xml files to which the Ohio-publication refers.
5minute_intervals_timestamp: it is a standardized to timestamps (starting from the beginning of the UNIX timestamps)
Features:
- missing_cbg: an indicater if the cbg measure is missing for a certain timestamp
- cbg: blood glucose measurement
- finger: finger-stick blood glucose measurement for reference
- basal: basal insuline delivery rate
- bolus: bolus insuline doses for meal correction etc
- hr: heartrate
- gsr: galvanic skin response
- carbInput: self-reported carbohydrate input

### Data Preprocessing 


## Set-up for LSTM with Keras
- Tensorflow version 2.9.0
- Keras-Applications version 1.0.8
- Keras-Preprocessing version 1.1.2

## Installations for Deep Reinforcement Learning
```
pip install tensorflow==1.14.0
pip install stable-baselines
pip install stable-baselines[mpi]
pip install mpi4py
```
