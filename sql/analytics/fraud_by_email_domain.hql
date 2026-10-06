SELECT COALESCE(P_emaildomain,'unknown') AS email_domain, COUNT(*) AS txns, SUM(isFraud) AS fraud, ROUND(AVG(isFraud)*100,3) AS fraud_rate_pct
FROM xtad.transactions GROUP BY COALESCE(P_emaildomain,'unknown') HAVING COUNT(*) >= 20 ORDER BY fraud_rate_pct DESC LIMIT 20
