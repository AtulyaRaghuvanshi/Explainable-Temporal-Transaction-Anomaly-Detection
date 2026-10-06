import sys,json
from pathlib import Path
import pandas as pd,streamlit as st
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));from src.config import load_config
cfg=load_config();p=root/cfg['paths']['processed'];a=root/cfg['paths']['artifacts'];st.set_page_config(page_title='X-TAD',layout='wide');st.title('X-TAD — Explainable Transaction Anomaly Detection')
try:d=pd.read_parquet(a/'hybrid_test_scores.parquet');t=pd.read_parquet(a/'tecs.parquet') if (a/'tecs.parquet').exists() else None
except Exception as e:st.error('Run the pipeline first.');st.exception(e);st.stop()
c1,c2,c3,c4=st.columns(4);c1.metric('Transactions',f'{len(d):,}');c2.metric('Fraud rate',f'{d.isFraud.mean()*100:.2f}%');c3.metric('High risk',f'{(d.hybrid_score>=cfg["hybrid"]["threshold"]).sum():,}');c4.metric('Mean risk',f'{d.hybrid_score.mean():.3f}');st.subheader('Highest-risk transactions');top=d.nlargest(50,'hybrid_score');st.dataframe(top.merge(t,on='TransactionID',how='left') if t is not None else top,use_container_width=True);tid=st.number_input('TransactionID',int(d.TransactionID.min()),int(d.TransactionID.max()),int(top.iloc[0].TransactionID));r=d[d.TransactionID==tid]
if len(r):
 r=r.iloc[0];cs=st.columns(4);cs[0].metric('Fraud probability',f'{r.fraud_probability:.3f}');cs[1].metric('Anomaly',f'{r.anomaly_score:.3f}');cs[2].metric('Context',f'{r.context_score:.3f}');cs[3].metric('Hybrid risk',f'{r.hybrid_score:.3f}');
 if t is not None:st.write('### Explanation stability');st.dataframe(t[t.TransactionID==tid],use_container_width=True)
 ep=a/'explanations.json'
 if ep.exists():
  ex=pd.DataFrame(json.loads(ep.read_text()));e=ex[ex.TransactionID==tid]
  if len(e):st.write('### Top SHAP features');st.json(json.loads(e.iloc[0].top_features))
hv=a/'hive'
if hv.exists() and list(hv.glob('*.csv')):
 st.divider();st.subheader('Hive analytics (SQL-on-Hadoop over Spark output on HDFS)')
 for f in sorted(hv.glob('*.csv')):
  q=pd.read_csv(f)
  with st.expander(f.stem.replace('_',' '),expanded=False):
   st.dataframe(q,use_container_width=True)
   if 'fraud_rate_pct' in q:st.bar_chart(q.set_index(q.columns[0])['fraud_rate_pct'])
