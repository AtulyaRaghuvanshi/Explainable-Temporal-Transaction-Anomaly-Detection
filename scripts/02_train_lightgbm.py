import sys,json,joblib
from pathlib import Path
import pandas as pd,lightgbm as lgb
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.config import load_config,ensure_dirs
from src.metrics import binary_metrics,at_k
cfg=load_config();ensure_dirs(cfg);p=root/cfg['paths']['processed'];a=root/cfg['paths']['artifacts'];tr=pd.read_parquet(p/'train.parquet');va=pd.read_parquet(p/'valid.parquet');te=pd.read_parquet(p/'test.parquet');cols=pd.read_csv(p/'feature_columns.csv',header=None)[0].tolist()
m=lgb.LGBMClassifier(objective='binary',n_estimators=cfg['model']['n_estimators'],learning_rate=cfg['model']['learning_rate'],num_leaves=cfg['model']['num_leaves'],max_depth=cfg['model']['max_depth'],colsample_bytree=cfg['model']['colsample_bytree'],reg_alpha=cfg['model']['reg_alpha'],reg_lambda=cfg['model']['reg_lambda'],class_weight='balanced',random_state=42,n_jobs=-1)
m.fit(tr[cols],tr.isFraud,eval_set=[(va[cols],va.isFraud)],eval_metric='average_precision',callbacks=[lgb.early_stopping(cfg['model']['early_stopping_rounds']),lgb.log_evaluation(50)])
pred=m.predict_proba(te[cols])[:,1];met=binary_metrics(te.isFraud,pred,cfg['hybrid']['threshold']);met['at_k']=at_k(te.isFraud,pred);print(json.dumps(met,indent=2));joblib.dump(m,a/'lightgbm.joblib');pd.DataFrame({'TransactionID':te.TransactionID,'fraud_probability':pred}).to_parquet(a/'lightgbm_test_scores.parquet',index=False);json.dump(met,open(a/'lightgbm_metrics.json','w'),indent=2)
