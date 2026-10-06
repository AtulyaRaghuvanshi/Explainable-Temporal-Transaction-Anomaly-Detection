SELECT bucket AS amount_bucket, COUNT(*) AS txns, ROUND(AVG(isFraud)*100,3) AS fraud_rate_pct
FROM (SELECT isFraud, CASE WHEN TransactionAmt < 25 THEN '1: <25' WHEN TransactionAmt < 100 THEN '2: 25-100'
                           WHEN TransactionAmt < 500 THEN '3: 100-500' ELSE '4: 500+' END AS bucket
      FROM xtad.transactions) t
GROUP BY bucket ORDER BY amount_bucket
