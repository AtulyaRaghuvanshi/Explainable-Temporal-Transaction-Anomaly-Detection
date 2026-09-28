# X-TAD — Explainable Temporal Transaction Anomaly Detection

Research-oriented IEEE-CIS fraud/anomaly detection project.

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
