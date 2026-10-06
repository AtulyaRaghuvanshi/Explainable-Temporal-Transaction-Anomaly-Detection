SELECT bucket AS card_history, COUNT(*) AS txns, ROUND(AVG(isFraud)*100,3) AS fraud_rate_pct
FROM (SELECT isFraud, CASE WHEN card1_hist_count = 0 THEN '1: first seen' WHEN card1_hist_count < 5 THEN '2: 1-4 prior'
                           WHEN card1_hist_count < 20 THEN '3: 5-19 prior' ELSE '4: 20+ prior' END AS bucket
      FROM xtad.transactions) t
GROUP BY bucket ORDER BY card_history
