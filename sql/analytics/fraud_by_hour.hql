SELECT tx_hour, COUNT(*) AS txns, SUM(isFraud) AS fraud, ROUND(AVG(isFraud)*100,3) AS fraud_rate_pct
FROM xtad.transactions GROUP BY tx_hour ORDER BY tx_hour
