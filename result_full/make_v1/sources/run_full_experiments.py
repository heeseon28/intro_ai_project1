from pathlib import Path
import os,sys,json,subprocess,time,hashlib,fcntl,shutil
from summarize_full import summarize
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs_full'/'make_v1'
OUT.mkdir(parents=True,exist_ok=True)
def status(**d):
 d.update(updated=time.time(),pid=os.getpid());p=OUT/'queue_status.tmp';p.write_text(json.dumps(d,indent=2));p.replace(OUT/'queue_status.json')
def main():
 lock=open(OUT/'.queue.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 os.chdir(ROOT)
 (OUT/'queue.pid').write_text(str(os.getpid()))
 hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('full_train.py','run_full_experiments.py','summarize_full.py')}
 protocol=dict(seeds=[0,1,2],models=['flat','hierarchy','hierarchy_make_weighted'],epochs=30,patience=5,batch_size=64,workers=2,input_size=224,pretraining='ImageNet1K_V1',weight='inverse sqrt TRAIN make frequency',selection='validation macro_f1',test='reserved, not evaluated',source_sha256=hashes)
 manifest=OUT/'protocol.json'
 if manifest.exists(): assert json.loads(manifest.read_text())==protocol,'Protocol changed; do not mix runs'
 else: manifest.write_text(json.dumps(protocol,indent=2))
 for f in hashes:
  dst=OUT/f
  if dst.exists(): assert hashlib.sha256(dst.read_bytes()).hexdigest()==hashes[f]
  else: shutil.copy2(ROOT/f,dst)
 (OUT/'README.md').write_text('Manufacturer hierarchy experiment: 3 models x seeds 0,1,2. Maximum 30 epochs, patience 5 on validation Macro-F1. Fixed pilot split, RGB224, ImageNet ResNet18, same augmentation and initialization per seed. Weighted model uses training manufacturer inverse-square-root weights. Test reserved. Smoke results are NOT scientific results. Resume by rerunning run_full_experiments.py after session restart. Completed epochs/checkpoints preserved; partial epoch restarts. Training accuracy is online on augmented batches. Validation metrics select checkpoints and are not final test estimates. No significance claim from 3 seeds. All files local, no Git upload.'+chr(10))
 try:
  status(state='smoke_check')
  with (OUT/'smoke.log').open('a') as log:
   subprocess.run([sys.executable,'-u','full_train.py','--out-dir',str(OUT/'smoke'),'--smoke'],stdout=log,stderr=subprocess.STDOUT,check=True)
  import pandas as pd
  for name in protocol['models']:
   d=OUT/'smoke'/name;m=json.loads((d/'metrics.json').read_text());pr=pd.read_csv(d/'validation_predictions.csv');h=pd.read_csv(d/'history.csv')
   assert m['smoke'] and 0<=m['accuracy']<=m['top3_accuracy']<=1
   assert abs(m['top3_accuracy']-pr[['top1','top2','top3']].eq(pr.target,axis=0).any(axis=1).mean())<1e-8
   assert 'train_accuracy' in h and h.train_loss.notna().all()
  print('SMOKE_ALL_THREE_OK',flush=True)
  for seed in protocol['seeds']:
   for f,h in hashes.items(): assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,'Source changed during queue'
   status(state='training',seed=seed)
   cmd=[sys.executable,'-u','full_train.py','--out-dir',str(OUT/f'seed{seed}'),'--seed',str(seed),'--epochs','30','--patience','5','--batch-size','64','--workers','2']
   print('SEED_START',seed,flush=True)
   with (OUT/f'seed{seed}.log').open('a') as log: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
   summarize(OUT)
   print('SEED_COMPLETE',seed,flush=True)
  status(state='complete');print('ALL_9_RUNS_COMPLETE',flush=True)
 except BaseException as e:
  status(state='failed',error=repr(e));raise
if __name__=='__main__': main()
