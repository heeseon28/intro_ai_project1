"""Authorized stages 0.5 through 2; never reads the test split."""
from pathlib import Path
from argparse import Namespace
import json,time,fcntl,hashlib,shutil,traceback
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import full_train as ft
from experiments.make_v2.tensor_cache import ROOT,BASE,REPORT,load_frames,build_cache,benchmark,sha
from experiments.make_v2.preprocessing import V2Cars,Letterbox
NAMES=['flat','hierarchy','hierarchy_make_weighted','flat_make_weighted']
BOOK=ROOT/'result_full/make_v2'
def status(**kw):ft.atomic_json(BASE/'queue_status.json',dict(updated=time.time(),**kw))
def setup_seed(seed,ci,tr,va):
 out=BASE/f'seed{seed}';out.mkdir(parents=True,exist_ok=True);snap=out/'splits';snap.mkdir(exist_ok=True)
 for name in ['train.csv','val.csv','class_info.csv']:
  src=ROOT/'splits'/name;dst=snap/name
  if dst.exists():assert sha(src)==sha(dst)
  else:shutil.copy2(src,dst)
 return out
def args_for(seed,smoke=False):return Namespace(seed=seed,epochs=1 if smoke else 30,batch_size=64,workers=2,patience=5,smoke=smoke,data_root=ROOT/'dataset/VMMRdb')
def smoke(ci,tr,va,mapping,counts):
 path=BASE/'smoke';path.mkdir(exist_ok=True);checks=[];fingerprints=[]
 for name in NAMES:
  h,w=ft.experiment_flags(name);ft.seed_all(0);m=ft.CarModel(mapping,pretrained=False)
  digest=hashlib.sha256()
  for key,v in m.state_dict().items():digest.update(key.encode());digest.update(v.numpy().tobytes())
  fingerprints.append(digest.hexdigest());lp,_=m(torch.zeros(2,3,224,224),h);F.nll_loss(lp,torch.tensor([0,1])).backward()
  assert (m.make.weight.grad is not None)==h
  if name=='flat_make_weighted':assert not h and w
  checks.append(dict(model=name,hierarchical=h,weighted=w,make_head_received_gradient=m.make.weight.grad is not None,initial_state_sha256=fingerprints[-1]))
  del m
 assert len(set(fingerprints))==1
 for shape in [(600,450),(450,600),(1530,560),(50,50),(1,80)]:
  from PIL import Image
  im=Image.new('RGB',shape,(120,80,40))
  for i in range(20):
   ft.seed_all(i);assert Letterbox(True)(im).shape==(3,224,224)
 smframes={'train':tr.groupby('model_idx',group_keys=False).head(3).reset_index(drop=True),'val':va.groupby('model_idx',group_keys=False).head(3).reset_index(drop=True)}
 ft.Cars=lambda df,root,train:V2Cars(df,root,train,'letterbox',None)
 for name in NAMES:ft.run_experiment(name,args_for(0,True),ci,mapping,smframes,counts,path)
 # Verify equal shuffled inputs/augmentation stream across model labels, independent of loss/heads.
 from torch.utils.data import DataLoader
 digests=[]
 for name in NAMES:
  ft.seed_all(0);ds=V2Cars(smframes['train'],ROOT/'dataset/VMMRdb',True,'letterbox',None)
  loader=DataLoader(ds,batch_size=64,shuffle=True,num_workers=2,pin_memory=True,prefetch_factor=2,worker_init_fn=ft.worker_seed,generator=torch.Generator().manual_seed(0))
  d=hashlib.sha256()
  for i,(x,y) in enumerate(loader):
   d.update(x.numpy().tobytes());d.update(y.numpy().tobytes())
   if i==1:break
  digests.append(d.hexdigest())
 assert len(set(digests))==1
 pd.DataFrame(checks).to_csv(BOOK/'implementation_checks.csv',index=False)
 ft.atomic_json(BOOK/'smoke_verification.json',dict(all_four_smoke_runs_pass=True,identical_initial_state=True,identical_first_two_augmented_batches=True,batch_sha256=digests[0],D_hierarchical=False,D_weighted=True,geometry_cases=100,test_read=False))
 print('ALL_SMOKE_CHECKS_PASS',flush=True)
def run_one(seed,name,preprocessing,ci,tr,va,mapping,counts,store):
 out=setup_seed(seed,ci,tr,va);folder=out/name;folder.mkdir(exist_ok=True)
 h,w=ft.experiment_flags(name)
 cfg=dict(seed=seed,experiment=name,hierarchical=h,weighted=w,preprocessing=preprocessing,cache=store is not None,epochs=30,batch_size=64,workers=2,patience=5,optimizer='AdamW',lr=3e-4,weight_decay=1e-4,schedule='cosine',selection='best validation macro_f1',precision='bf16',backbone='ImageNet1K_V1 ResNet18',split_sha256={n:sha(ROOT/'splits'/n) for n in ['train.csv','val.csv','class_info.csv']},source_sha256={n:sha(ROOT/n) for n in ['full_train.py','experiments/make_v2/preprocessing.py','experiments/make_v2/tensor_cache.py','experiments/make_v2/run_workflow.py']},geometry='letterbox; scale request .9..1.1 and translation <=10%, clipped to preserve original content rectangle' if preprocessing=='letterbox' else 'V1 PIL RandomResizedCrop(.6,1)/Resize256 CenterCrop224',test_read=False)
 cp=folder/'run_config.json'
 if cp.exists():assert json.loads(cp.read_text())==cfg,'Source/config changed on resume'
 else:ft.atomic_json(cp,cfg)
 src=folder/'sources';src.mkdir(exist_ok=True)
 for namefile in cfg['source_sha256']:shutil.copy2(ROOT/namefile,src/Path(namefile).name)
 ft.Cars=lambda df,root,train:V2Cars(df,root,train,preprocessing,store)
 status(state='running',stage=1 if name=='flat' and seed==0 else 2,seed=seed,experiment=name,preprocessing=preprocessing)
 result=ft.run_experiment(name,args_for(seed),ci,mapping,{'train':tr,'val':va},counts,out)
 return dict(seed=seed,experiment=name,path=str(folder.relative_to(ROOT)),preprocessing=preprocessing,reused_v1=False,metrics=result)
def main():
 BASE.mkdir(parents=True,exist_ok=True);BOOK.mkdir(parents=True,exist_ok=True)
 lock=open(ROOT/'runs_pilot/.gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(2);torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=False
 assert torch.cuda.is_available();ci,tr,va=load_frames();mapping=ci.make_idx.to_numpy(int);counts=np.bincount(tr.make_idx,minlength=10)
 for seed in range(3):
  old=json.loads((ROOT/f'runs_full/make_v1/seed{seed}/config.json').read_text())
  assert old['epochs']==30 and old['batch_size']==64 and old['workers']==2 and old['patience']==5
 status(state='running',stage=.5)
 bench=json.loads((REPORT/'benchmark.json').read_text()) if (REPORT/'benchmark.json').exists() else benchmark()
 store=None
 if bench['cache_selected']:
  total_epochs=30*4
  if bench['estimated_train_epoch_saving_seconds']*total_epochs>bench['estimated_build_seconds']:
   store=build_cache(pd.concat([tr,va],ignore_index=True),BASE/'cache/full')
 print('CACHE_DECISION',store is not None,flush=True)
 smoke(ci,tr,va,mapping,counts)
 selection=run_one(0,'flat','letterbox',ci,tr,va,mapping,counts,store)
 oldm=json.loads((ROOT/'runs_full/make_v1/seed0/flat/metrics.json').read_text())
 win=selection['metrics']['macro_f1']>oldm['macro_f1']
 choice=dict(state='complete',selected_preprocessing='letterbox' if win else 'center',rule='strictly greater validation Macro-F1 on seed 0; exploratory screening, no multi-seed effect claim',old_macro_f1=oldm['macro_f1'],new_macro_f1=selection['metrics']['macro_f1'],delta_pp=100*(selection['metrics']['macro_f1']-oldm['macro_f1']),source=selection['path'],cache_used=store is not None)
 ft.atomic_json(BOOK/'selection.json',choice);print('PREPROCESSING_SELECTION',json.dumps(choice),flush=True)
 from experiments.make_v2.build_result_book import selection_report
 selection_report(selection,ci,tr,va)
 matrix=[]
 for seed in range(3):
  for name in NAMES:
   if not win and name!='flat_make_weighted':
    folder=ROOT/f'runs_full/make_v1/seed{seed}'/name
    matrix.append(dict(seed=seed,experiment=name,path=str(folder.relative_to(ROOT)),preprocessing='center',reused_v1=True,metrics=json.loads((folder/'metrics.json').read_text())))
   else:matrix.append(run_one(seed,name,choice['selected_preprocessing'],ci,tr,va,mapping,counts,store))
   ft.atomic_json(BOOK/'run_matrix.json',matrix)
 # Check source split identities once more before assembly. No test read.
 load_frames()
 from experiments.make_v2.build_result_book import build
 build(matrix,ci,tr,va,choice,selection)
 status(state='complete',stage=2,runs=12,new_preprocessing=choice['selected_preprocessing'],book=str(BOOK),test_read=False)
 print('STAGES_05_TO_2_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  BASE.mkdir(parents=True,exist_ok=True);status(state='failed',error=repr(exc));traceback.print_exc();raise
