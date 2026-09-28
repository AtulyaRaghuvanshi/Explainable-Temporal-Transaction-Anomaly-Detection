import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,precision_score,recall_score,f1_score,matthews_corrcoef
def binary_metrics(y,p,t=.5):
    pred=(np.asarray(p)>=t).astype(int)
    return {'pr_auc':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'mcc':float(matthews_corrcoef(y,pred))}
def at_k(y,scores,fracs=(.001,.005,.01,.02)):
    y=np.asarray(y);o=np.argsort(-np.asarray(scores));out=[]
    for f in fracs:
        k=max(1,int(len(y)*f));sel=o[:k];tp=y[sel].sum();out.append({'fraction':f,'k':k,'precision':float(tp/k),'recall':float(tp/max(1,y.sum()))})
    return out
