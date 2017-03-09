import psycopg2
import pandas as pd
from ggplot import *
conn = psycopg2.connect(dbname="ventris", host="192.168.1.119", user="ventris_admin", password="X4NAdu")

plot_title = "XLF LSTM Train 03, N: 128:64, TS: 5"

# Plot scatterplot 
plot_query = """
SELECT * 
FROM lstm_madness_train_03
ORDER BY index
"""

plot_df = pd.read_sql_query(plot_query, con=conn)

p = ggplot(aes(x='ret', y='prediction'), data=plot_df)
p + geom_point() + ggtitle(plot_title)

# Plot cumulative returns
plot_cum_query = """
SELECT dt
, SUM(ret) OVER (ORDER BY dt) AS cum_actual
, SUM(prediction) OVER (ORDER BY dt) AS cum_predict
FROM lstm_madness_train_03
ORDER BY index
"""

plot_cum_df = pd.read_sql_query(plot_cum_query, con=conn)

ggplot(pd.melt(plot_cum_df, id_vars=['dt']), aes(x='dt', y='value', color='variable')) +\
    geom_line() +\
    ggtitle(plot_title) +\
    labs(y = "Cumulative Returns")