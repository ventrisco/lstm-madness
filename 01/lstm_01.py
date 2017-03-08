import psycopg2
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from sqlalchemy import create_engine
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

conn = psycopg2.connect(dbname="ventris", host="127.0.0.1", user="ventris_admin", password="X4NAdu")
engine = create_engine('postgresql://ventris_admin:X4NAdu@localhost:5432/ventris')

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

x_df = pd.read_sql_query(x_query, con=conn)

# x_df
# 4568  2017-02-21    xlf  0.004904
# 4569  2017-02-22    xlf  0.000813
# 4570  2017-02-23    xlf  0.000000
# 4571  2017-02-24    xlf -0.007720
# 4572  2017-02-27    xlf  0.005323

# [4573 rows x 3 columns]

x_df_wide = x_df.pivot(index="dt", columns="ticker", values="ret")

# >>> x_df_wide
# ticker           xlf
# dt                  
# 1998-12-23  0.014745
# 1998-12-24  0.006605
# 1998-12-28 -0.013123



# Get y data

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

y_df = pd.read_sql_query(y_query, con=conn)

# >>> y_df
# 4569  2017-02-22    xlf  0.000000
# 4570  2017-02-23    xlf -0.007720
# 4571  2017-02-24    xlf  0.005323
# 4572  2017-02-27    xlf -0.000407

# [4573 rows x 3 columns]

y_df_wide = y_df.pivot(index="dt", columns="ticker", values="ret")

# >>> y_df_wide
# ticker           xlf
# dt                  
# 1998-12-23  0.006605
# 1998-12-24 -0.013123
# 1998-12-28  0.010638
# 1998-12-29 -0.003947
# 1998-12-30 -0.009247

# Create dataset from dataframe. Basically extract the array and includes the
# astype flag that's used by tensorflow
dataset_x = x_df_wide.values.astype('float32')
dataset_y = y_df_wide.values.astype('float32')


train_size = int(len(dataset_x) * 0.67)
test_size = len(dataset_x) - train_size
stage_train_x, stage_test_x = dataset_x[0:train_size], dataset_x[train_size:len(dataset_x)]
train_y,  test_y = dataset_y[0:train_size], dataset_y[train_size:len(dataset_y)]

# >>> stage_train_x.shape
# (3063, 1)
# (sample, features) or (rows, columna)

# reshape x input to be [samples, time steps, features]
# the lstm requires this tuple as a signature of the data structure
train_x = numpy.reshape(stage_train_x, (stage_train_x.shape[0], 1, stage_train_x.shape[1]))
test_x = numpy.reshape(stage_test_x, (stage_test_x.shape[0], 1, stage_test_x.shape[1]))
# >>> train_x.shape       
# (2387, 1, 55)
# the middle value is the lookback 


# input_dim = the number of features
number_of_features = train_x.shape[2]

model = Sequential()
model.add(LSTM(1, input_dim=number_of_features))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_y, nb_epoch=100, batch_size=1, verbose=2)


# make predictions
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

# Add prediction to y_df_dataframe
y_df_train = y_df[0:train_x.shape[0]]
y_df_train["prediction"] = train_predict
y_df_train.to_sql('lstm_madness_xlf_train_01', engine)

# I run the epochs on another machine so I need to download data to my local 
# machine to plot.
import psycopg2
import pandas as pd
from ggplot import *

conn = psycopg2.connect(dbname="ventris", host="192.168.1.119", user="ventris_admin", password="X4NAdu")

plot_title = "XLF LSTM Train lookback: 1, layers: 1"

# Plot scatterplot 
plot_query = """
SELECT * 
FROM lstm_madness_xlf_train_01
ORDER BY index
"""

plot_df = pd.read_sql_query(plot_query, con=conn)
p = ggplot(aes(x='ret', y='prediction'), data=plot_df)
p + geom_point() + ggtitle(plot_title)


plot_cum_query = """
SELECT dt
, SUM(ret) OVER (ORDER BY dt) AS cum_actual
, SUM(prediction) OVER (ORDER BY dt) AS cum_predict
FROM lstm_madness_xlf_train_01
ORDER BY index
"""

plot_cum_df = pd.read_sql_query(plot_cum_query, con=conn)

ggplot(pd.melt(plot_cum_df, id_vars=['dt']), aes(x='dt', y='value', color='variable')) +\
    geom_line() +\
    ggtitle(plot_title) +\
    labs(y = "Cumulative Returns")