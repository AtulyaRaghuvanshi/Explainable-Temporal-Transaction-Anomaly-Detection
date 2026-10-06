# X-TAD results — full IEEE-CIS dataset on the HDFS + Spark + Hive pipeline

Run date: 2026-10-06. Data: Kaggle `ieee-fraud-detection`, `train_transaction.csv` + `train_identity.csv` (all rows, no sampling).
Raw tables for everything below are in [reports/results/](results/).

## 1. Pipeline run

| Stage | Tool | Result | Time |
|---|---|---|---|
| Load raw CSVs | HDFS (`/xtad/raw`) | 651.7 MB transactions + 25.3 MB identity | ~1 min |
| Feature engineering | Spark 3.5.3 (1 master, 1 worker, local Docker) | 590,540 rows, windowed per-entity history features, Parquet back to HDFS | ~2.5 min |
| Analytics | Hive 4.0 external table over Spark output, 6 HiveQL queries | 6 result tables (below) | ~1 min |
| Export + prepare | HDFS → pandas | 413,378 train / 88,581 valid / 88,581 test rows, 455 features (chronological split) | ~1 min |
| LightGBM | — | trained, early stopping | 1 m 45 s |
| Autoencoder (PyTorch) | — | 25 epochs on up to 250k rows | 4 m 41 s |
| Hybrid, SHAP (20 txns), TECS (100 txns), ablation | — | see below | ~1 min |

Fraud base rate in the data: 3.5% (14,538 of 413,378 in train).

## 2. Model results (test set, chronological hold-out)

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| **LightGBM** | **0.513** | **0.897** |
| Hybrid (0.6 fraud + 0.2 anomaly + 0.2 context) | 0.469 | 0.871 |
| Autoencoder (anomaly score alone) | 0.128 | 0.775 |
| Context score alone | 0.025 | 0.329 |

At threshold 0.5: LightGBM precision 0.32 / recall 0.59 / F1 0.42 / MCC 0.41. Hybrid precision 0.45 / recall 0.50 / F1 0.47 / MCC 0.45.

Precision when reviewing only the top-scored transactions (hybrid): 86% in the top 0.1% (88 transactions), 88% in the top 0.5%, 80% in the top 1%, 66% in the top 2%. LightGBM alone: 100% / 92% / 89% / 69%.

TECS (Temporal Explanation Consensus Score, 100 test transactions): mean adjacent cosine similarity 0.995 (min 0.985); mean top-k Jaccard overlap 0.90 (min 0.60, median 0.91; at least a quarter of transactions have a perfect overlap of 1.0).

## 3. Hive analytics (SQL-on-Hadoop over the full 590k rows)

- **Product:** `ProductCD = C` has an 11.7% fraud rate versus 2.0% for `W` (the most common product, 440k rows). `S` 5.9%, `H` 4.8%, `R` 3.8%.
- **Card type:** credit cards are riskier than debit: visa credit 6.8%, mastercard credit 6.9%, discover 7.9%, against 2.2–2.5% for visa and mastercard debit.
- **Email domain** (domains with at least 20 rows): `protonmail.com` 40.8% (31 of 76), `mail.com` 19.0%, `outlook.es` 13.0%, `outlook.com` 9.5%, `hotmail.com` 5.3%, `gmail.com` 4.4%. Missing email domains sit near the average (3.0%).
- **Time of day:** fraud rate peaks in the early-morning band, hours 5–9 at 7–10.6%, while volume is lowest there (2.5k–9.7k per hour against 40k+ in the evening). Midday to evening runs 2.3–3.7%. Note that `TransactionDT` is a time offset with an unknown start, so "hour 0" is not necessarily midnight local time. Treat this as a cyclical pattern, not a clock-hour claim.
- **Amount:** smallest (<25) and largest (500+) transactions are the riskiest: 7.0% and 4.7%, against 3.0% for 25–100 and 3.4% for 100–500.
- **Card history:** cards with 20+ prior transactions have a higher fraud rate (3.7%) than newer cards (about 2.5%). So "card seen for the first time" is not a strong fraud signal on this dataset.

## 4. Findings

1. **The big data pipeline works at full scale.** Spark on HDFS processed all 590,540 rows × 434 raw columns in about 2.5 minutes on a single-worker Docker cluster capped at about 3.9 GB. Narrow window projections joined back to the wide table were what made it fit in memory.
2. **Spark features match the pandas features.** On a 60,000-row test, all 23 compared feature columns (history counts, amount z-scores, seconds-since-previous, basic features) matched the original pandas implementation to within 1e-3. This check found and fixed one bug (first transactions per card became 0 instead of null).
3. **LightGBM alone beats the hybrid on ranking metrics** (PR-AUC 0.513 vs 0.469, ROC-AUC 0.897 vs 0.871). The hybrid does give a better F1 and MCC at the fixed 0.5 threshold (0.47 / 0.45 vs 0.42 / 0.41), but the ranking metrics (PR-AUC, ROC-AUC) are what compare the scores independent of threshold, and those favour LightGBM.
4. **The context score hurts the hybrid.** Its ROC-AUC is 0.33, below 0.5, meaning it is anti-correlated with fraud on this data. It still gets a 20% weight in the hybrid. Dropping it or re-deriving it is the clearest way to improve the hybrid. The autoencoder score is weak but above chance (ROC-AUC 0.775).
5. **Explanations are stable under temporal reference changes.** TECS cosine is about 0.995 and top-k overlap about 0.90, so SHAP attributions for a transaction barely change across the temporal backgrounds tested. This is a supportive result for the proposed metric, but it does not show TECS can detect unstable explanations, because no unstable explanations were tested.
6. **SHAP highlights Vesta-engineered columns** (V258, C1, V156, C13, C14, V149, C11), plus the project's own `ProductCD_amount_z` feature at rank 10. This is the Spark-built historical feature, so the behavioural features do carry signal. Caveat: these importances come from only 20 explained test transactions, so they are indicative, not a stable global ranking.

## 5. Caveats

- LightGBM metrics come from a single chronological split and a single run, with no confidence intervals or repeated seeds.
- SHAP global importance uses 20 transactions; TECS uses 100. Both are small samples.
- The Spark cluster is one worker on one machine, so this shows the pipeline works with the Hadoop tools, not horizontal scaling.
- The Kaggle test set has no labels, so all evaluation is on a hold-out from the training file, as the README describes.
