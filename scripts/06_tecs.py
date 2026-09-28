import argparse,sys
from pathlib import Path
import pandas as pd
import joblib
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.config import load_config
from src.explainability import explainer,values,per_tx
ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=100);args=ap.parse_args();cfg=load_config();p=root/cfg['paths']['processed'];a=root/cfg['paths']['artifacts'];tr=pd.read_parquet(p/'train.parquet');te=pd.read_parquet(p/'test.parquet');cols=pd.read_csv(p/'feature_columns.csv',header=None)[0].tolist();m=joblib.load(a/'lightgbm.joblib');scores=pd.read_parquet(a/'hybrid_test_scores.parquet');ids=scores.nlargest(args.n,'hybrid_score').TransactionID;X=te.set_index('TransactionID').loc[ids,cols];bg1=tr[cols].sample(min(500,len(tr)),random_state=1);recent=tr[tr.TransactionDT>=tr.TransactionDT.quantile(.8)];bg2=recent[cols].sample(min(500,len(recent)),random_state=2);bg3=tr[cols].sample(min(500,len(tr)),random_state=3);sv=[values(explainer(m,bg),X) for bg in [bg1,bg2,bg3]];out=per_tx(sv,10);out['TransactionID']=X.index.values;out.to_parquet(a/'tecs.parquet',index=False);print(out[['tecs_cosine','tecs_topk']].describe())
