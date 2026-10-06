# X-TAD — Explainable Temporal Transaction Anomaly Detection

Research-oriented IEEE-CIS fraud/anomaly detection project.

Big data stack: **HDFS** (storage) + **Apache Spark** (distributed feature engineering) + **Apache Hive** (SQL analytics), run as Docker containers.

Pipeline: leakage-safe temporal behavioral features -> LightGBM fraud model + PyTorch autoencoder -> hybrid risk score -> SHAP -> counterfactual-ready explanations -> Temporal Explanation Consensus Score (TECS).

Dataset: https://www.kaggle.com/competitions/ieee-fraud-detection/data

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Put `train_transaction.csv` and `train_identity.csv` under `data/raw/`, or run `python scripts/download_data.py` after configuring Kaggle credentials.

## Big data layer (HDFS + Spark + Hive)
```
raw CSVs --load--> HDFS /xtad/raw --Spark (window functions)--> HDFS /xtad/features  --export--> Python models (LightGBM, AE, SHAP, TECS)
                                                        \-> HDFS /xtad/hive/transactions <-- Hive external table <-- sql/analytics/*.hql --> dashboard
```
Requires Docker Desktop (about 4 GB RAM). Spark and Hive run inside the containers, so no local Java/Hadoop is needed.
```bash
python scripts/make_synthetic_data.py          # only if you do not have the Kaggle files in data/raw/
python scripts/00_bigdata_pipeline.py          # up -> load -> spark -> hive -> export  (or pass individual steps)
python scripts/01_prepare.py --from-hdfs       # use the Spark-built features instead of pandas
```
UIs: HDFS http://localhost:9870, Spark http://localhost:8080, HiveServer2 http://localhost:10002. Stop with `docker compose down`.
Hive result tables are written to `data/artifacts/hive/` and shown in the dashboard.

## Run
```bash
python scripts/01_prepare.py --sample-frac 0.20
python scripts/02_train_lightgbm.py
python scripts/03_train_autoencoder.py
python scripts/04_hybrid.py
python scripts/05_explain.py --n 20
python scripts/06_tecs.py --n 100
python scripts/07_ablation.py
streamlit run dashboard/app.py
```
Use `--sample-frac 1.0` for the full labeled training set.

## Research contribution
TECS is a proposed metric that compares local SHAP explanations for the same transaction under multiple temporal/reference backgrounds. It reports adjacent cosine similarity and top-k Jaccard overlap. It should be presented as a proposed framework/metric, not as the first explanation-stability method without a formal literature review.

## Leakage prevention
Data are chronologically sorted. Historical counts, means and previous-transaction features use only prior observations. Train/validation/test are chronological. The public Kaggle competition test labels are not used for supervised evaluation.
