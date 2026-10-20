from pathlib import Path
import json,sys
import numpy as np
import pandas as pd

def summarize(root):
 root=Path(root);rows=[];pairs=[];makes=[]
 for seed in (0,1,2):
  d=root/f'seed{seed}'
  for name in ('flat','hierarchy','hierarchy_make_weighted'):
   f=d/name/'metrics.json'
   if not f.exists(): continue
   m=json.loads(f.read_text());assert not m['smoke'] and m['split']=='validation'
   row={k:v for k,v in m.items() if isinstance(v,(int,float,str,bool))}
   h=pd.read_csv(d/name/'history.csv');row['best_epoch']=int(h.loc[h.macro_f1.idxmax(),'epoch']);rows.append(row)
   for r in m['per_make']: makes.append(dict(seed=seed,experiment=name,**r))
   ci=pd.read_csv(d/'splits/class_info.csv').sort_values('model_idx');labels=ci.make_model.tolist()
   cm=np.load(d/name/'validation_confusion.npy');np.fill_diagonal(cm,0)
   for i,j in np.argwhere(cm>0):
    pairs.append(dict(seed=seed,experiment=name,true_class=labels[i],predicted_class=labels[j],count=int(cm[i,j]),same_make=bool(ci.make_idx.iloc[i]==ci.make_idx.iloc[j])))
 if not rows: return
 df=pd.DataFrame(rows);df.to_csv(root/'per_run.csv',index=False)
 keys=['accuracy','macro_f1','balanced_accuracy','top3_accuracy','minority_make_fine_accuracy','make_accuracy','cross_make_error_rate','within_make_error_rate']
 summary=df.groupby('experiment')[keys].agg(['count','mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(root/'mean_std.csv')
 deltas=[]
 for a,b in [('flat','hierarchy'),('hierarchy','hierarchy_make_weighted'),('flat','hierarchy_make_weighted')]:
  for seed in (0,1,2):
   x=df[(df.seed==seed)&(df.experiment==a)];y=df[(df.seed==seed)&(df.experiment==b)]
   if len(x)==len(y)==1: deltas.append(dict(seed=seed,comparison=b+' minus '+a,**{k+'_pp':100*float(y[k].iloc[0]-x[k].iloc[0]) for k in keys}))
 if deltas:
  dd=pd.DataFrame(deltas);dd.to_csv(root/'paired_deltas_pp.csv',index=False)
  ag=dd.groupby('comparison')[[k+'_pp' for k in keys]].agg(['count','mean','std']);ag.columns=['_'.join(c) for c in ag.columns];ag.to_csv(root/'paired_summary_pp.csv')
  dd.assign(macro_f1_win=dd.macro_f1_pp>0).groupby('comparison').macro_f1_win.agg(['count','sum']).to_csv(root/'macro_f1_win_counts.csv')
 pd.DataFrame(makes).to_csv(root/'per_make.csv',index=False)
 pd.DataFrame(pairs).sort_values('count',ascending=False).to_csv(root/'confusion_pairs.csv',index=False)
 print('SUMMARY_UPDATED',len(rows),'of 9',flush=True)
if __name__=='__main__': summarize(sys.argv[1])
