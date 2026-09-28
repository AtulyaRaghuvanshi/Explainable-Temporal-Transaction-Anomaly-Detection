import argparse,sys,json
from pathlib import Path
import pandas as pd,numpy as np,joblib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.config import load_config
from src.explainability import explainer,values,top
ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=20);args=ap.parse_args();cfg=load_config();p=root/cfg['paths']['processed'];a=root/cfg['paths']['artifacts'];tr=pd.read_parquet(p/'train.parquet');te=pd.read_parquet(p/'test.parquet');cols=pd.read_csv(p/'feature_columns.csv',header=None)[0].tolist();m=joblib.load(a/'lightgbm.joblib');bg=tr[cols].sample(min(1000,len(tr)),random_state=42);exp=explainer(m,bg);scores=pd.read_parquet(a/'hybrid_test_scores.parquet');ids=scores.nlargest(args.n,'hybrid_score').TransactionID;X=te.set_index('TransactionID').loc[ids,cols];sv=values(exp,X);rows=[{'TransactionID':int(t),'top_features':json.dumps(top(sv[i],cols,10))} for i,t in enumerate(X.index)];json.dump(rows,open(a/'explanations.json','w'),indent=2);np.save(a/'shap_values_global.npy',sv)
