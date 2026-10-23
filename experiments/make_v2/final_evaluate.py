"""One-time final test evaluation of frozen validation-selected checkpoints; no training."""
from pathlib import Path
import json,hashlib,time,fcntl,subprocess,traceback,shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
import full_train as ft
from experiments.make_v2.build_result_book import KEYS,COMPARISONS,PAIRS,WITHIN_PAIRS,pair_counts
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'runs_full/make_v2/final_test';OUT=ROOT/'result_full/make_v2/test'
NAMES=['flat','hierarchy','hierarchy_make_weighted','flat_make_weighted']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):ft.atomic_json(p,x)
def recompute(pred,ci,minority):
 y=pred.target.to_numpy();p=pred.prediction.to_numpy();mapping=ci.make_idx.to_numpy()
 precision,recall,f1,support=precision_recall_fscore_support(y,p,labels=np.arange(50),zero_division=0)
 return dict(accuracy=float((y==p).mean()),macro_f1=float(f1.mean()),balanced_accuracy=float(recall.mean()),top3_accuracy=float(pred[['top1','top2','top3']].eq(pred.target,axis=0).any(axis=1).mean()),minority_make_fine_accuracy=float(np.mean([(p[mapping[y]==g]==y[mapping[y]==g]).mean() for g in minority])),cross_make_error_rate=float((mapping[p]!=mapping[y]).mean()),within_make_error_rate=float(((p!=y)&(mapping[p]==mapping[y])).mean()))
def main():
 RAW.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 lock=open(ROOT/'runs_pilot/.gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=False
 assert torch.cuda.is_available(),'CUDA required'
 matrix=json.loads((ROOT/'result_full/make_v2/run_matrix.json').read_text())
 assert len(matrix)==12 and {(r['seed'],r['experiment']) for r in matrix}=={(s,n) for s in range(3) for n in NAMES}
 ci=pd.read_csv(ROOT/'splits/class_info.csv').sort_values('model_idx').reset_index(drop=True)
 tr=pd.read_csv(ROOT/'splits/train.csv');va=pd.read_csv(ROOT/'splits/val.csv')
 assert len(tr)==82929 and len(va)==17771
 mapping=ci.make_idx.to_numpy(dtype=np.int64);minority=np.argsort(np.bincount(tr.make_idx,minlength=10),kind='stable')[:3]
 assert sorted(ci[ci.make_idx.isin(minority)].make.unique().tolist())==['bmw','gmc','volkswagen']
 specs=[]
 for r in matrix:
  src=ROOT/r['path'];hist=pd.read_csv(src/'history.csv');m=json.loads((src/'metrics.json').read_text())
  assert r['preprocessing']=='center' and abs(hist.macro_f1.max()-m['macro_f1'])<1e-9
  for name in ['train.csv','val.csv','class_info.csv']:
   assert sha(src.parent/'splits'/name)==sha(ROOT/'splits'/name)
  specs.append(dict(seed=r['seed'],experiment=r['experiment'],checkpoint=str((src/'best.pt').relative_to(ROOT)),checkpoint_sha256=sha(src/'best.pt'),best_epoch=int(hist.loc[hist.macro_f1.idxmax(),'epoch']),validation_macro_f1=m['macro_f1'],hierarchical=ft.experiment_flags(r['experiment'])[0],weighted=ft.experiment_flags(r['experiment'])[1]))
 selected=max([s for s in specs if s['experiment']=='flat'],key=lambda s:s['validation_macro_f1'])
 config=dict(version=1,selection='A Flat fixed by validation model mean; representative A seed fixed by its validation Macro-F1 only',selected_model='flat',representative_seed=selected['seed'],selected_checkpoint=selected['checkpoint'],preprocessing='Resize(256), CenterCrop(224), RGB, ToTensor, ImageNet normalize; no augmentation',batch_size=64,workers=2,precision='bf16 autocast if supported, same as validation',primary_metric='macro_f1',minority_metric='unweighted mean of three manufacturer fine accuracies, NOT pooled image accuracy',minority_makes=['bmw','gmc','volkswagen'],decision_rule='paired Macro-F1 >0 for 3/3 seeds and mean >0.2 percentage points; operational rule, not significance',reselection_after_test=False,extra_training=False,checkpoints=specs,split_sha256={n:sha(ROOT/'splits'/n) for n in ['train.csv','val.csv','test.csv','class_info.csv']},source_sha256={n:sha(ROOT/n) for n in ['full_train.py','experiments/make_v2/final_evaluate.py']})
 freeze=RAW/'frozen_protocol.json'
 if freeze.exists():
  expected=json.loads(freeze.read_text())['config']
  if expected!=config:
   amendment=json.loads((OUT/'evaluation_code_amendment.json').read_text())
   normalized=json.loads(json.dumps(config));key='experiments/make_v2/final_evaluate.py'
   assert amendment['old_sha256']==expected['source_sha256'][key] and amendment['new_sha256']==config['source_sha256'][key]
   normalized['source_sha256'][key]=expected['source_sha256'][key]
   assert normalized==expected,'Model/data/metric configuration changed: refuse evaluation'
 else:
  with freeze.open('x') as f:json.dump(dict(frozen_before_test_rows_or_images_read=True,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),config=config),f,indent=2)
 shutil.copy2(freeze,OUT/freeze.name)
 # First access to test rows/images occurs only after model/config/checkpoint freeze above.
 te=pd.read_csv(ROOT/'splits/test.csv');assert len(te)==17771 and te.rel_path.is_unique
 assert ci.model_idx.tolist()==list(range(50));assert np.array_equal(mapping[te.model_idx],te.make_idx)
 assert np.array_equal(ci.make_model.to_numpy()[te.model_idx],te.make_model)
 for other in [tr,va]:
  for col in ['rel_path','md5']:assert not (set(te[col])&set(other[col])),f'exact leakage {col}'
 assert te.rel_path.map(lambda s:not Path(s).is_absolute() and '..' not in Path(s).parts).all()
 te[['rel_path','model_idx','make_idx','make_model']].to_csv(OUT/'test_index.csv',index=False);ci.to_csv(OUT/'class_info.csv',index=False)
 save(OUT/'data_audit.json',dict(train_n=len(tr),val_n=len(va),test_n=len(te),path_md5_disjoint=True,labels_consistent=True,near_duplicate_test_audit=False,labels_modified=False))
 loader=DataLoader(ft.Cars(te,ROOT/'dataset/VMMRdb',False),batch_size=64,shuffle=False,num_workers=2,pin_memory=True,prefetch_factor=2,worker_init_fn=ft.worker_seed,generator=torch.Generator().manual_seed(0))
 rows=[];makes=[];classes=[];pairs=[];verified=[];predictions={}
 for i,s in enumerate(specs):
  seed=s['seed'];name=s['experiment'];dest=OUT/'evidence'/f'seed{seed}'/name;dest.mkdir(parents=True,exist_ok=True)
  if (dest/'metrics.json').exists():
   m=json.loads((dest/'metrics.json').read_text());pred=pd.read_csv(dest/'test_predictions.csv');cm=np.load(dest/'test_confusion.npy');assert m['checkpoint_sha256']==s['checkpoint_sha256'];print('REUSE_SAVED_TEST',seed,name,flush=True)
  else:
   marker=RAW/f'seed{seed}_{name}.started'
   assert not marker.exists(),'Incomplete previous inference: refuse an unrecorded second test pass'
   marker.write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
   start=time.monotonic();ft.seed_all(seed);model=ft.CarModel(mapping,pretrained=False).cuda()
   state=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);del state
   assert np.array_equal(model.mapping.cpu().numpy(),mapping)
   save(RAW/'status.json',dict(state='running',completed=i,total=12,seed=seed,experiment=name,updated=time.time()))
   print('TEST_START',i+1,seed,name,flush=True)
   m,cm,pred=ft.evaluate(model,loader,s['hierarchical'],mapping,ci)
   m['minority_make_fine_accuracy']=float(np.mean([m['per_make'][int(g)]['fine_accuracy'] for g in minority]))
   m.update(seed=seed,experiment=name,split='test',checkpoint_sha256=s['checkpoint_sha256'],validation_best_epoch=s['best_epoch'],minutes=(time.monotonic()-start)/60)
   tmp=dest/'predictions.tmp';pred.to_csv(tmp,index=False);tmp.replace(dest/'test_predictions.csv');np.save(dest/'test_confusion.npy',cm);save(dest/'metrics.json',m)
   del model;torch.cuda.empty_cache();print('TEST_DONE',seed,name,'F1',m['macro_f1'],'minutes',m['minutes'],flush=True)
  assert len(pred)==17771 and np.array_equal(pred.target,te.model_idx)
  assert np.array_equal(cm,confusion_matrix(pred.target,pred.prediction,labels=np.arange(50)))
  computed=recompute(pred,ci,minority)
  for k,v in computed.items():assert abs(m[k]-v)<1e-9,(seed,name,k)
  assert abs(m['accuracy']+m['cross_make_error_rate']+m['within_make_error_rate']-1)<1e-12
  rows.append(dict(seed=seed,experiment=name,best_epoch=s['best_epoch'],**{k:m[k] for k in KEYS}))
  makes.extend([dict(seed=seed,experiment=name,**v) for v in m['per_make']]);classes.extend([dict(seed=seed,experiment=name,**v) for v in m['per_class']])
  pairs.extend([dict(seed=seed,experiment=name,**v) for v in pair_counts(cm,ci,PAIRS+WITHIN_PAIRS)])
  verified.append(dict(seed=seed,experiment=name,n=len(pred),checkpoint_sha256=s['checkpoint_sha256'],topk_first_vs_argmax_disagreements=int((pred.top1!=pred.prediction).sum()),verification='PASS independent metrics/confusion/labels/decomposition; topk tie order not forced'))
  predictions[seed,name]=pred;pd.DataFrame(rows).to_csv(OUT/'per_run.csv',index=False)
 df=pd.DataFrame(rows);assert len(df)==12
 stats=df.groupby('experiment')[KEYS].agg(['mean','std']);stats.columns=['_'.join(c) for c in stats.columns];stats.to_csv(OUT/'mean_std.csv')
 pd.DataFrame(makes).to_csv(OUT/'per_make.csv',index=False);pd.DataFrame(classes).to_csv(OUT/'per_class.csv',index=False);pd.DataFrame(pairs).to_csv(OUT/'confusion_pairs.csv',index=False);pd.DataFrame(verified).to_csv(OUT/'verification.csv',index=False)
 deltas=[];trans=[]
 for seed in range(3):
  for a,b,label in COMPARISONS:
   x=df[(df.seed==seed)&(df.experiment==a)].iloc[0];z=df[(df.seed==seed)&(df.experiment==b)].iloc[0]
   deltas.append(dict(seed=seed,comparison=label,**{k:100*(x[k]-z[k]) for k in KEYS}))
   pa=predictions[seed,a];pb=predictions[seed,b];ca=pa.target==pa.prediction;cb=pb.target==pb.prediction;trans.append(dict(seed=seed,comparison=label,rescued=int((ca&~cb).sum()),harmed=int((~ca&cb).sum()),both_wrong=int((~ca&~cb).sum()),both_correct=int((ca&cb).sum())))
 dd=pd.DataFrame(deltas);dd.to_csv(OUT/'paired_deltas_pp.csv',index=False);pd.DataFrame(trans).to_csv(OUT/'prediction_transitions.csv',index=False)
 ds=dd.groupby('comparison')[KEYS].agg(['mean','std']);ds.columns=['_'.join(c) for c in ds.columns];ds.to_csv(OUT/'paired_mean_std_pp.csv')
 wins=[];decisions=[]
 for label,g in dd.groupby('comparison'):
  for k in KEYS:wins.append(dict(comparison=label,metric=k,mean_delta_pp=float(g[k].mean()),paired_sd_pp=float(g[k].std(ddof=1)),improved_seeds=int(((g[k]*(-1 if 'error_rate' in k else 1))>0).sum()),n=3))
  v=g.macro_f1;decisions.append(dict(comparison=label,mean_macro_f1_delta_pp=float(v.mean()),paired_sd_pp=float(v.std(ddof=1)),improved_seeds=int((v>0).sum()),operational_effect=bool((v>0).all() and v.mean()>.2),threshold_pp=.2))
 pd.DataFrame(wins).to_csv(OUT/'positive_delta_counts.csv',index=False);pd.DataFrame(decisions).to_csv(OUT/'hypothesis_decisions.csv',index=False)
 val=pd.read_csv(ROOT/'result_full/make_v2/per_run.csv');gap=df.merge(val[['seed','experiment']+KEYS],on=['seed','experiment'],suffixes=('_test','_validation'))
 for k in KEYS:gap[k+'_test_minus_val_pp']=100*(gap[k+'_test']-gap[k+'_validation'])
 gap.to_csv(OUT/'validation_test_comparison.csv',index=False)
 save(OUT/'completion.json',dict(state='complete',evaluated_runs=12,test_n=len(te),no_training=True,selection_changed=False,selected_model='flat',representative_seed=selected['seed'],completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
 save(RAW/'status.json',dict(state='complete',completed=12,total=12,updated=time.time()))
 print('FINAL_TEST_COMPLETE',stats[[k+'_mean' for k in ['accuracy','macro_f1','minority_make_fine_accuracy']]].mul(100).to_string(),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  RAW.mkdir(parents=True,exist_ok=True);save(RAW/'status.json',dict(state='failed',error=repr(e),updated=time.time()));raise
