from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def load_raw(transaction_path,identity_path=None):
    df=pd.read_csv(ROOT/transaction_path,low_memory=False)
    if identity_path and (ROOT/identity_path).exists():
        ident=pd.read_csv(ROOT/identity_path,low_memory=False)
        df=df.merge(ident,on='TransactionID',how='left',suffixes=('','_identity'))
    return df
def chronological_split(df,train_frac=.70,valid_frac=.15):
    df=df.sort_values('TransactionDT').reset_index(drop=True);n=len(df);a=int(n*train_frac);b=int(n*(train_frac+valid_frac))
    return df.iloc[:a].copy(),df.iloc[a:b].copy(),df.iloc[b:].copy()
