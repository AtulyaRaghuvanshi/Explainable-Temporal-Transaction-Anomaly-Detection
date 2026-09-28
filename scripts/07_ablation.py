import sys,pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score,roc_auc_score
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));from src.config import load_config
cfg=load_config();a=root/cfg['paths']['artifacts'];d=pd.read_parquet(a/'hybrid_test_scores.parquet');rows=[]
for n,s in [('LightGBM',d.fraud_probability),('Autoencoder',d.anomaly_score),('Context',d.context_score),('Hybrid',d.hybrid_score)]:rows.append({'model':n,'pr_auc':average_precision_score(d.isFraud,s),'roc_auc':roc_auc_score(d.isFraud,s)})
o=pd.DataFrame(rows);print(o.to_string(index=False));o.to_csv(a/'ablation.csv',index=False)
