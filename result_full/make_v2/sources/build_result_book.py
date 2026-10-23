"""Verified four-model V2 result book, including reused V1 evidence; no test read."""
from pathlib import Path
import json,hashlib,shutil,itertools
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,recall_score,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'result_full/make_v2'
NAMES=['flat','hierarchy','hierarchy_make_weighted','flat_make_weighted'];SHORT=['A Flat','B Hierarchy','C Hier+weight','D Flat+weight']
KEYS=['accuracy','macro_f1','balanced_accuracy','top3_accuracy','minority_make_fine_accuracy','cross_make_error_rate','within_make_error_rate']
PAIRS=[('chevrolet/silverado','gmc/sierra_1500'),('chevrolet/tahoe','gmc/yukon_1500')]
WITHIN_PAIRS=[('dodge/ram_1500','dodge/ram_2500'),('bmw/325i','bmw/328i'),('honda/civic','honda/accord'),('volkswagen/golf','volkswagen/gti')]
COMPARISONS=[('hierarchy','flat','B-A hierarchy'),('flat_make_weighted','flat','D-A weight in flat'),('hierarchy_make_weighted','hierarchy','C-B weight in hierarchy'),('hierarchy_make_weighted','flat_make_weighted','C-D hierarchy with weights')]
def verify(folder,ci,va,minority):
 m=json.loads((folder/'metrics.json').read_text());hist=pd.read_csv(folder/'history.csv');p=pd.read_csv(folder/'validation_predictions.csv');y=p.target.to_numpy();pred=p.prediction.to_numpy();mapping=ci.make_idx.to_numpy()
 assert len(y)==len(va) and np.array_equal(y,va.model_idx.to_numpy()) and not m['smoke']
 values=dict(accuracy=float((y==pred).mean()),macro_f1=float(f1_score(y,pred,labels=np.arange(50),average='macro',zero_division=0)),balanced_accuracy=float(recall_score(y,pred,labels=np.arange(50),average='macro',zero_division=0)),top3_accuracy=float(p[['top1','top2','top3']].eq(p.target,axis=0).any(axis=1).mean()),minority_make_fine_accuracy=float(np.mean([(pred[mapping[y]==g]==y[mapping[y]==g]).mean() for g in minority])),cross_make_error_rate=float((mapping[y]!=mapping[pred]).mean()),within_make_error_rate=float(((y!=pred)&(mapping[y]==mapping[pred])).mean()))
 for k,v in values.items():assert abs(m[k]-v)<1e-9,(folder,k,m[k],v)
 cm=confusion_matrix(y,pred,labels=np.arange(50));assert np.array_equal(cm,np.load(folder/'validation_confusion.npy'));assert abs(hist.macro_f1.max()-m['macro_f1'])<1e-9
 return m,hist,p,cm
def pair_counts(cm,ci,pairs):
 labels=ci.make_model.tolist();return [{'target':a,'prediction':b,'errors':int(cm[labels.index(a),labels.index(b)])} for pair in pairs for a,b in [pair,pair[::-1]]]
def selection_report(selection,ci,tr,va):
 minor=np.argsort(np.bincount(tr.make_idx,minlength=10),kind='stable')[:3];rows=[];pairs=[]
 for label,folder in [('v1_center',ROOT/'runs_full/make_v1/seed0/flat'),('v2_letterbox',ROOT/selection['path'])]:
  m,h,p,cm=verify(folder,ci,va,minor);rows.append(dict(condition=label,**{k:m[k] for k in KEYS}))
  pairs.extend([dict(condition=label,**r) for r in pair_counts(cm,ci,PAIRS+WITHIN_PAIRS)])
 pd.DataFrame(rows).to_csv(OUT/'selection_comparison.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'selection_pair_errors.csv',index=False)
def build(matrix,ci,tr,va,choice,selection):
 assert len(matrix)==12;minor=np.argsort(np.bincount(tr.make_idx,minlength=10),kind='stable')[:3]
 rows=[];makes=[];classes=[];pairrows=[];checked=[];preds={};histories={};cms={}
 for r in matrix:
  seed=r['seed'];name=r['experiment'];src=ROOT/r['path'];m,h,p,cm=verify(src,ci,va,minor)
  snap=src.parent/'splits'
  for f in ['train.csv','val.csv','class_info.csv']:assert hashlib.sha256((snap/f).read_bytes()).hexdigest()==hashlib.sha256((ROOT/'splits'/f).read_bytes()).hexdigest()
  if not r['reused_v1']:
   cfg=json.loads((src/'run_config.json').read_text())
   for f,expected in cfg['source_sha256'].items():assert hashlib.sha256((src/'sources'/Path(f).name).read_bytes()).hexdigest()==expected
   assert cfg['hierarchical']==(name in ['hierarchy','hierarchy_make_weighted'])
   assert cfg['weighted']==(name in ['hierarchy_make_weighted','flat_make_weighted'])
  best=int(h.loc[h.macro_f1.idxmax(),'epoch']);rows.append(dict(seed=seed,experiment=name,preprocessing=r['preprocessing'],reused_v1=r['reused_v1'],best_epoch=best,epochs_completed=len(h),minutes=float(h.minutes.sum()),**{k:m[k] for k in KEYS}))
  makes.extend([dict(seed=seed,experiment=name,**x) for x in m['per_make']]);classes.extend([dict(seed=seed,experiment=name,**x) for x in m['per_class']])
  pairrows.extend([dict(seed=seed,experiment=name,kind='sibling' if x['target'].split('/')[0]!=x['prediction'].split('/')[0] else 'within_make',**x) for x in pair_counts(cm,ci,PAIRS+WITHIN_PAIRS)])
  dest=OUT/'evidence'/f'seed{seed}'/name;dest.mkdir(parents=True,exist_ok=True)
  for f in ['metrics.json','history.csv','validation_predictions.csv','validation_confusion.npy']:shutil.copy2(src/f,dest/f)
  if (src/'run_config.json').exists():shutil.copy2(src/'run_config.json',dest/'run_config.json')
  checked.append(dict(seed=seed,experiment=name,n=len(p),best_epoch=best,source=r['path'],checks='PASS: metrics, confusion, best epoch, split identity, model flags'))
  preds[seed,name]=p;histories[seed,name]=h;cms[seed,name]=cm
  idx=OUT/'evidence'/f'seed{seed}';ci.to_csv(idx/'class_info.csv',index=False);va[['rel_path','model_idx','make_idx','make_model']].to_csv(idx/'validation_index.csv',index=False)
 df=pd.DataFrame(rows);df.to_csv(OUT/'per_run.csv',index=False);pd.DataFrame(makes).to_csv(OUT/'per_make.csv',index=False);pd.DataFrame(classes).to_csv(OUT/'per_class.csv',index=False);pd.DataFrame(pairrows).to_csv(OUT/'confusion_pairs.csv',index=False);pd.DataFrame(checked).to_csv(OUT/'verification.csv',index=False)
 stats=df.groupby('experiment')[KEYS].agg(['mean','std']);stats.columns=['_'.join(c) for c in stats.columns];stats.to_csv(OUT/'mean_std.csv')
 deltas=[];trans=[]
 for seed in range(3):
  for a,b,label in COMPARISONS:
   x=df[(df.seed==seed)&(df.experiment==a)].iloc[0];z=df[(df.seed==seed)&(df.experiment==b)].iloc[0]
   deltas.append(dict(seed=seed,comparison=label,**{k:100*(x[k]-z[k]) for k in KEYS}))
   pa=preds[seed,a];pb=preds[seed,b];ca=pa.prediction.to_numpy()==pa.target.to_numpy();cb=pb.prediction.to_numpy()==pb.target.to_numpy();trans.append(dict(seed=seed,comparison=label,rescued=int((ca&~cb).sum()),harmed=int((~ca&cb).sum()),both_correct=int((ca&cb).sum()),both_wrong=int((~ca&~cb).sum())))
  old=json.loads((ROOT/f'runs_full/make_v1/seed{seed}/flat/metrics.json').read_text());new=df[(df.seed==seed)&(df.experiment=='flat')].iloc[0]
  deltas.append(dict(seed=seed,comparison='V2 A-V1 A preprocessing',**{k:100*(new[k]-old[k]) for k in KEYS}))
 dd=pd.DataFrame(deltas);dd.to_csv(OUT/'paired_deltas_pp.csv',index=False);pd.DataFrame(trans).to_csv(OUT/'prediction_transitions.csv',index=False)
 ds=dd.groupby('comparison')[KEYS].agg(['mean','std']);ds.columns=['_'.join(c) for c in ds.columns];ds.to_csv(OUT/'paired_mean_std_pp.csv')
 winrows=[]
 for comp,g in dd.groupby('comparison'):
  for k in KEYS:
   sign=-1 if 'error_rate' in k else 1;arr=g[k]*sign
   winrows.append(dict(comparison=comp,metric=k,mean_delta_pp=g[k].mean(),paired_sd_pp=g[k].std(ddof=1),improved_seeds=int((arr>0).sum()),n=3))
 pd.DataFrame(winrows).to_csv(OUT/'positive_delta_counts.csv',index=False)
 # Operational rule frozen before training: 3/3 favorable direction and mean >0.2pp.
 decisions=[]
 for comp,g in dd.groupby('comparison'):
  vals=g.macro_f1;decisions.append(dict(comparison=comp,mean_macro_f1_delta_pp=vals.mean(),paired_sd_pp=vals.std(ddof=1),improved_seeds=int((vals>0).sum()),operational_effect=bool((vals>0).all() and vals.mean()>.2),threshold_pp=.2,note='operational consistency rule, not statistical significance'))
 pd.DataFrame(decisions).to_csv(OUT/'hypothesis_decisions.csv',index=False)
 fig,axes=plt.subplots(3,4,figsize=(16,10),layout='constrained')
 for seed in range(3):
  for col,name in enumerate(NAMES):
   ax=axes[seed,col];h=histories[seed,name];ax.plot(h.epoch,h.train_accuracy*100,label='train augmented');ax.plot(h.epoch,h.accuracy*100,label='val accuracy');ax.plot(h.epoch,h.macro_f1*100,label='val Macro-F1');ax.set_title(f'{SHORT[col]} seed {seed}');ax.set_xlabel('Epoch');ax.set_ylabel('%')
   if seed==0 and col==0:ax.legend(fontsize=7)
 fig.savefig(OUT/'learning_curves.png',dpi=150);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 for ax,k in zip(axes,['macro_f1','minority_make_fine_accuracy','within_make_error_rate']):
  mean=[100*stats.loc[n,k+'_mean'] for n in NAMES];std=[100*stats.loc[n,k+'_std'] for n in NAMES];ax.bar(SHORT,mean,yerr=std,capsize=3);ax.tick_params(axis='x',rotation=25);ax.set_title(k);ax.set_ylabel('%')
 fig.savefig(OUT/'results_overview.png',dpi=150);plt.close(fig)
 sources=OUT/'sources';sources.mkdir(exist_ok=True)
 for f in [ROOT/'full_train.py']+list((ROOT/'experiments/make_v2').glob('*.py')):shutil.copy2(f,sources/f.name)
 selection_report(selection,ci,tr,va)
 report=['# make_v2 final validation result book','',json.dumps(choice,ensure_ascii=False,indent=2),'','Validation only; test untouched. Same fixed split, 3 seeds. CSV metrics are fractions 0..1; paired deltas are percentage points.','',df.to_string(index=False),'',pd.DataFrame(decisions).to_string(index=False),'','B-A: hierarchy; D-A: weighting in Flat; C-B: weighting in hierarchy; C-D: hierarchy under matched weights. V2 A-V1 A is the preprocessing bundle comparison. If center selected, A/B/C are reused V1 and V2 A-V1 A is zero by design.','The letterbox intervention changes both training augmentation and validation geometry, so it does not isolate center cropping alone. Cache speed is reported separately and is not evidence for classification improvement.','The 0.2pp threshold with 3/3 favorable seeds was fixed before V2 training; this is an operational rule, not a statistical significance test.','Learning train scores use augmented data and changing weights during the epoch; they are not directly equivalent to evaluation-mode train scores. Weighted train loss differs from unweighted validation NLL.','Observed differences do not establish causes such as label noise, shared platform, pretraining redundancy or gradient conflict. See Stage0 visual screening limitations. No validation labels or images were removed.','Original data, cache, smoke and checkpoints stay under ignored runs_full. Evidence includes predictions/metrics only. No test performance is reported.']
 (OUT/'README.md').write_text(chr(10).join(report),encoding='utf-8')
 manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'};(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
 print('BOOK_VERIFIED',len(rows),'runs',len(manifest),'files',flush=True)
