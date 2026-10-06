SELECT ProductCD, COUNT(*) AS txns, SUM(isFraud) AS fraud, ROUND(AVG(isFraud)*100,3) AS fraud_rate_pct, ROUND(AVG(TransactionAmt),2) AS avg_amount
FROM xtad.transactions GROUP BY ProductCD ORDER BY fraud_rate_pct DESC
