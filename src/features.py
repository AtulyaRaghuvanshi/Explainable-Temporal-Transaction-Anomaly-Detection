import numpy as np,pandas as pd

def basic(df):
    df=df.copy()
    df['tx_hour']=((df.TransactionDT//3600)%24).astype('int16')
    df['tx_day']=(df.TransactionDT//86400).astype('int32')
    df['tx_weekday']=((df.TransactionDT//86400)%7).astype('int8')
    df['amount_log']=np.log1p(df.TransactionAmt.clip(lower=0))
    df['missing_count']=df.drop(columns=['isFraud'],errors='ignore').isna().sum(axis=1)
    return df

def historical(df):
    df=df.sort_values('TransactionDT').copy()
    entities=[c for c in ['card1','addr1','DeviceInfo','P_emaildomain','ProductCD'] if c in df]
    for c in entities:
        g=df.groupby(c,dropna=False,sort=False)
        df[f'{c}_hist_count']=g.cumcount().astype('float32')
        prev_mean=g.TransactionAmt.transform(lambda s:s.shift(1).expanding(min_periods=1).mean())
        prev_std=g.TransactionAmt.transform(lambda s:s.shift(1).expanding(min_periods=2).std())
        df[f'{c}_amount_z']=((df.TransactionAmt-prev_mean)/(prev_std.fillna(0)+1e-6)).clip(-20,20).astype('float32')
        df[f'{c}_seconds_since_prev']=(df.TransactionDT-g.TransactionDT.shift(1)).clip(lower=0).astype('float32')
    for a,b in [('card1','addr1'),('card1','DeviceInfo'),('card1','P_emaildomain')]:
        if a in df and b in df:
            key=df[a].astype('string').fillna('__NA__')+'|'+df[b].astype('string').fillna('__NA__')
            df[f'{a}_{b}_hist_count']=key.groupby(key,sort=False).cumcount().astype('float32')
    return df

def prepare(train,valid,test,precomputed=False):
    # precomputed=True: basic + historical features were already built by the Spark job (HDFS)
    if not precomputed:
        train,valid,test=basic(train),basic(valid),basic(test)
        all_df=pd.concat([train.assign(_split=0),valid.assign(_split=1),test.assign(_split=2)],ignore_index=True).sort_values('TransactionDT')
        all_df=historical(all_df)
        train=all_df[all_df._split==0].drop(columns='_split');valid=all_df[all_df._split==1].drop(columns='_split');test=all_df[all_df._split==2].drop(columns='_split')
    cats=train.select_dtypes(include=['object','string','category']).columns
    for c in cats:
        s=train[c].astype('string').fillna('__NA__');mp=s.value_counts(normalize=True)
        for frame in (train,valid,test):
            frame[c+'_freq']=frame[c].astype('string').fillna('__NA__').map(mp).fillna(0).astype('float32')
            frame.drop(columns=c,inplace=True)
    cols=train.select_dtypes(include=[np.number]).columns.tolist();cols=[c for c in cols if c not in ['isFraud','TransactionID']]
    for c in cols:
        med=train[c].replace([np.inf,-np.inf],np.nan).median()
        for frame in (train,valid,test):frame[c]=frame[c].replace([np.inf,-np.inf],np.nan).fillna(med)
    return train,valid,test,cols
