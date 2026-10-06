CREATE DATABASE IF NOT EXISTS xtad;
DROP TABLE IF EXISTS xtad.transactions;
CREATE EXTERNAL TABLE xtad.transactions (
  TransactionID BIGINT, TransactionDT BIGINT, isFraud INT, TransactionAmt DOUBLE,
  ProductCD STRING, card1 BIGINT, card4 STRING, card6 STRING, P_emaildomain STRING, DeviceType STRING,
  tx_hour INT, tx_weekday INT, card1_hist_count DOUBLE, card1_amount_z DOUBLE)
STORED AS PARQUET
LOCATION 'hdfs://namenode:8020/xtad/hive/transactions';
