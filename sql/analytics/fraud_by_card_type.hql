SELECT COALESCE(card4,'unknown') AS network, COALESCE(card6,'unknown') AS card_type, COUNT(*) AS txns, ROUND(AVG(isFraud)*100,3) AS fraud_rate_pct
FROM xtad.transactions GROUP BY COALESCE(card4,'unknown'), COALESCE(card6,'unknown') ORDER BY txns DESC
