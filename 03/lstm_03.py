"""
LSTM with 128:64 Neurons, 2 time steps, 1000 epochs
"""
import os
import time
import datetime as dt
import psycopg2
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from sqlalchemy import create_engine
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

conn = psycopg2.connect(dbname="ventris", host="127.0.0.1", user="ventris_admin", password="X4NAdu")
engine = create_engine('postgresql://ventris_admin:X4NAdu@localhost:5432/ventris')

ITERATION = "03"
TIME_STEPS = 2
TRAIN_TABLENAME = "lstm_madness_train_%s" % ITERATION

def df_dataset(query, conn):
  df = pd.read_sql_query(query, con=conn)
  df_wide = df.pivot(index="dt", columns="ticker", values="ret")
  return df, df_wide.values.astype('float32')


def generate_train_test_dataset(dataset, train_size):
  train = dataset[0:train_size]
  test = dataset[train_size:len(dataset_x)]
  return train, test


def reshape_train_dataset(dataset, time_steps, number_of_features):
  """
  The data set should have the shape: 
  (3063, 1)
  (sample, features) or (rows, columns)
  This function adds the time_steps in the tuple which the lstm mode will use to
  when it does its fit. 
  """
  return numpy.reshape(dataset, (dataset.shape[0], dataset.shape[1], number_of_features))


def create_dataset_with_timesteps(dataset, time_steps=1):
    # dataset has to be of type numbpy.array
    # convert an array of values into a dataset matrix
    end_index = time_steps
    offset = time_steps - 1 # This offset handles the fact that python includes the first index and excludes the last
    data = []
    for i in range(len(dataset) - offset):
        a = dataset[i:end_index, 0]
        end_index = end_index + 1
        data.append(a)
    return numpy.array(data)

numpy.random.seed(7)

# get x data

x_query = """
SELECT
  dt
, ticker
, ret
FROM xlf_returns
WHERE dt > '1998-12-22'
AND dt < '2017-02-28'
ORDER BY dt
"""

x_df, dataset_x = df_dataset(x_query, conn)

# get y data

y_query = """
SELECT
  dt
, ticker
, ret
FROM xlf_next_day_returns
WHERE dt > '1998-12-22'
AND dt < '2017-02-28'
ORDER BY dt;
"""

y_df, dataset_y = df_dataset(y_query, conn)

# add time steps to 
# Let's use five to represent a week. 

START_INDEX = TIME_STEPS - 1
dataset_x_with_ts = create_dataset_with_timesteps(dataset_x, TIME_STEPS)
dataset_y_with_ts_offset = dataset_y[START_INDEX::]
# >>> dataset_x_with_time_steps
# array([ array([ 0.01474535,  0.00660498, -0.01312336,  0.0106383 , -0.00394733], dtype=float32),
#        array([ 0.00660498, -0.01312336,  0.0106383 , -0.00394733, -0.00924707], dtype=float32),
#        array([-0.01312336,  0.0106383 , -0.00394733, -0.00924707,  0.        ,
#         0.00933338], dtype=float32),
#        ...,
#        array([ 0.00780929, -0.00244702,  0.00040883,  0.00490401,  0.00081338,
#         0.        , -0.00772048,  0.00532346], dtype=float32),
#        array([-0.00244702,  0.00040883,  0.00490401,  0.00081338,  0.        ,
#        -0.00772048,  0.00532346], dtype=float32),
#        array([ 0.00040883,  0.00490401,  0.00081338,  0.        , -0.00772048,
#         0.00532346], dtype=float32)], dtype=object)

# >>> len(dataset_x_with_time_steps)
# 4568
# test = dataset_y[4:4573]
# >>> test
# array([[-0.00924707],
#        [ 0.        ],
#        [ 0.00933338],
#        ..., 
#        [-0.00772048],
#        [ 0.00532346],
#        [-0.00040725]], dtype=float32)
# >>> len(test)
# 4569
"""
Because added the time steps, we have shift our dataset_y so that the predict dates
sync up.
"""



# Reshape df_wide into numpy arrays
train_size = int(len(dataset_x) * 0.67)
stage_train_x, stage_test_x = generate_train_test_dataset(dataset_x_with_ts, train_size)
train_y,  test_y =  generate_train_test_dataset(dataset_y_with_ts_offset, train_size)

# >>> stage_train_x.shape
# (3063, 1)
# (sample, features) or (rows, columna)

# reshape x input to be [samples, time steps, features]
# the lstm requires this tuple as a signature of the data structure
train_x = reshape_train_dataset(stage_train_x, TIME_STEPS, 1)
test_x = reshape_train_dataset(stage_test_x, TIME_STEPS, 1)
# >>> train_x.shape       
# (2387, 1, 55)
# the middle value is the lookback 


# input_dim = the number of features
number_of_features = train_x.shape[2]

d = 0.2
model = Sequential()
model.add(LSTM(2, input_dim=number_of_features))
model.add(Dropout(d))
model.add(Dense(1,init='uniform',activation='linear'))
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

model.fit(
    train_x,
    train_y,
    batch_size=32,
    nb_epoch=1000,
    validation_split=0.1,
    verbose=1)

# make predictions
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

# Add prediction to y_df_dataframe
y_df_train = y_df[0:train_x.shape[0]]
y_df_train["prediction"] = train_predict
y_df_train.to_sql(TRAIN_TABLENAME, engine)
