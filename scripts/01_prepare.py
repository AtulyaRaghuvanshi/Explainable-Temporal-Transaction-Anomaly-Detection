import argparse,sys
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.config import load_config,ensure_dirs
from src.data import load_raw,chronological_split
from src.features import prepare
ap=argparse.ArgumentParser();ap.add_argument('--sample-frac',type=float,default=None);args=ap.parse_args();cfg=load_config();ensure_dirs(cfg)
df=load_raw(cfg['paths']['raw_transaction'],cfg['paths']['raw_identity']);f=args.sample_frac if args.sample_frac is not None else cfg['data']['sample_frac']
if f<1:df=df.sample(frac=f,random_state=cfg['data']['random_state'])
tr,va,te=chronological_split(df,cfg['data']['train_frac'],cfg['data']['valid_frac']);tr,va,te,cols=prepare(tr,va,te);out=root/cfg['paths']['processed'];out.mkdir(parents=True,exist_ok=True)
for n,x in [('train',tr),('valid',va),('test',te)]:x.to_parquet(out/f'{n}.parquet',index=False)
pd.Series(cols).to_csv(out/'feature_columns.csv',index=False,header=False);print('saved',len(tr),len(va),len(te),'rows and',len(cols),'features')
