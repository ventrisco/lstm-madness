-- Init
CREATE USER ventris_admin WITH PASSWORD 'X4NAdu';
ALTER ROLE ventris_admin WITH SUPERUSER;
CREATE DATABASE ventris WITH OWNER ventris_admin;

psql -d ventris -U ventris_admin -h 127.0.0.1

-- Create table for XLF components

DROP TABLE IF EXISTS xlf_components_data;
CREATE TABLE xlf_components_data (
  ticker text
, dt date
, open numeric
, high numeric
, low numeric
, close numeric
, volume numeric
, adj_close numeric
);

COPY xlf_components_data FROM '/var/tmp/xlf_buffer.csv' WITH DELIMITER ',';

-- Create table for xlf_etf_data

DROP TABLE IF EXISTS xlf_etf_data;
CREATE TABLE xlf_etf_data (
  ticker text
, dt date
, open numeric
, high numeric
, low numeric
, close numeric
, volume numeric
, adj_close numeric
);

COPY xlf_etf_data FROM '/var/tmp/xlf_etf_buffer.csv' WITH DELIMITER ',';

-- Create returns based on the days's close
DROP TABLE IF EXISTS xlf_components_returns;
CREATE TABLE xlf_components_returns AS
SELECT
  dt
, ticker
, CASE 
    WHEN LAG(close, 1) OVER (PARTITION BY ticker ORDER BY dt) > 0
      THEN (close - LAG(close, 1) OVER (PARTITION BY ticker ORDER BY dt)) / LAG(close, 1) OVER (PARTITION BY ticker ORDER BY dt)
    ELSE NULL END AS ret
FROM xlf_components_data;

-- Create returns of xlf 
DROP TABLE IF EXISTS xlf_returns;
CREATE TABLE xlf_returns AS
SELECT
  dt
, ticker
, CASE 
    WHEN LAG(close, 1) OVER (PARTITION BY ticker ORDER BY dt) > 0
      THEN (close - LAG(close, 1) OVER (PARTITION BY ticker ORDER BY dt)) / LAG(close, 1) OVER (PARTITION BY ticker ORDER BY dt)
    ELSE NULL END AS ret
FROM xlf_etf_data;

-- Create xlf next day return
-- Probably going to predict this the most often
DROP TABLE IF EXISTS xlf_next_day_returns;
CREATE TABLE xlf_next_day_returns AS
SELECT
  dt
, ticker
, CASE
    WHEN LEAD(close, 1) OVER (PARTITION BY ticker ORDER BY dt) > 0
      THEN (LEAD(close, 1) OVER (PARTITION BY ticker ORDER BY dt) - close) / close
    ELSE NULL END AS ret
FROM xlf_etf_data;

# double check offset


-- Need to get the max of the earliest of the ticker symbolvs
-- If you don't do this it you'll need have NAAN values
SELECT * FROM xlf_components_returns
WHERE ticker IN (
  SELECT ticker
  FROM
  (
  SELECT 
    MIN(dt) as min_dt, 
    ticker 
  FROM xlf_components_data 
  GROUP BY ticker 
  ) AS a
  WHERE min_dt <= '2003-01-01'
) AND dt > '2003-01-01';

