import numpy as np,shap,pandas as pd
def explainer(model,bg):return shap.TreeExplainer(model,data=bg,feature_perturbation='interventional')
def values(exp,X):
    v=exp.shap_values(X)
    if isinstance(v,list):v=v[1]
    if getattr(v,'ndim',0)==3:v=v[:,:,1]
    return np.asarray(v)
def top(v,names,k=10):
    ix=np.argsort(np.abs(v))[::-1][:k];return [(names[i],float(v[i])) for i in ix]
def cosine(a,b):
    d=np.linalg.norm(a)*np.linalg.norm(b);return float(np.dot(a,b)/d) if d else 0
def jaccard(a,b,k=10):
    A=set(np.argsort(np.abs(a))[-k:]);B=set(np.argsort(np.abs(b))[-k:]);return len(A&B)/max(1,len(A|B))
def tecs(vectors,k=10):
    return {'tecs_cosine':float(np.mean([cosine(a,b) for a,b in zip(vectors[:-1],vectors[1:])])),'tecs_topk':float(np.mean([jaccard(a,b,k) for a,b in zip(vectors[:-1],vectors[1:])]))} if len(vectors)>1 else {'tecs_cosine':1.,'tecs_topk':1.}
def per_tx(background_values,k=10):
    n=background_values[0].shape[0];rows=[]
    for i in range(n):rows.append(tecs([x[i] for x in background_values],k))
    return pd.DataFrame(rows)
