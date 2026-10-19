"""Task 2 baseline: random ResNet-18, deterministic preprocessing, plain CE.
No pretrained weights, augmentation, balancing, hierarchy, special loss or LR schedule.
Test split is checked for metadata overlap, never loaded for model evaluation.
"""
import argparse, hashlib, json, math, os, random, re, shutil, subprocess, sys, time, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT=Path(__file__).resolve().parent.parent
PACKAGE=Path(__file__).resolve().parent

def write_json(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def worker_seed(_):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed);torch.set_num_threads(1)

class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,s):
        for f in self.streams: f.write(s);f.flush()
    def flush(self):
        for f in self.streams: f.flush()

class Cars(Dataset):
    def __init__(self,df,root,size):
        self.df=df.reset_index(drop=True);self.root=root
        self.tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),
                                   transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i];p=self.root/r.rel_path
        try:
            with Image.open(p) as im: x=self.tf(im.convert('RGB'))
        except Exception as e: raise RuntimeError(f'Image decode failed: {p}') from e
        return x,int(r.model_idx)

def read_splits(folder,data):
    ci=pd.read_csv(folder/'class_info.csv').sort_values('model_idx').reset_index(drop=True)
    assert ci.model_idx.tolist()==list(range(len(ci)))
    assert len(ci)==50,'Task 2 requires the existing 50 classes'
    parts={k:pd.read_csv(folder/f'{k}.csv') for k in ('train','val','test')}
    for k,d in parts.items():
        assert not d[['model_idx','make_model','rel_path','md5']].isna().any().any(),k
        assert d.rel_path.is_unique,k
        assert set(d.model_idx)==set(ci.model_idx),f'Missing class in {k}'
        assert np.array_equal(ci.make_model.to_numpy()[d.model_idx.to_numpy(dtype=int)],d.make_model.to_numpy()),k
        assert d.rel_path.map(lambda p:not Path(p).is_absolute() and '..' not in Path(p).parts).all()
        assert d.groupby('model_idx').size().reindex(ci.model_idx).tolist()==ci[f'n_{k}'].tolist(),k
    for a,b in (('train','val'),('train','test'),('val','test')):
        for col in ('rel_path','md5'):
            assert not(set(parts[a][col])&set(parts[b][col])),f'Cross-split {col} leakage: {a}/{b}'
    # Test image content is not opened, and no test performance is measured.
    available={}
    for parent in sorted({str(Path(p).parent) for k in ('train','val') for p in parts[k].rel_path}):
        try:
            with os.scandir(data/parent) as entries:
                available[parent]={entry.name for entry in entries if entry.is_file()}
        except FileNotFoundError: available[parent]=set()
    for k in ('train','val'):
        missing=[p for p in parts[k].rel_path if Path(p).name not in available[str(Path(p).parent)]]
        assert not missing,f'Missing {k} images: {missing[:5]}'
    for r in parts['train'].sample(n=min(16,len(parts['train'])),random_state=0).itertuples():
        assert hashlib.md5((data/r.rel_path).read_bytes()).hexdigest()==r.md5,'Dataset differs from split manifest'
    return ci,parts

def make_loader(ds,args,train,epoch=0):
    gen=torch.Generator().manual_seed(args.seed+epoch)
    return DataLoader(ds,batch_size=args.batch_size,shuffle=train,num_workers=args.workers,
        pin_memory=args.device=='cuda',worker_init_fn=worker_seed,generator=gen,
        prefetch_factor=2 if args.workers else None)

def metrics(y,p,loss,nclasses):
    pr,rc,f1,sup=precision_recall_fscore_support(y,p,labels=np.arange(nclasses),zero_division=0)
    return {'loss':float(loss),'accuracy':float(np.mean(np.asarray(y)==p)),
            'macro_f1':float(f1.mean()),'balanced_accuracy':float(rc.mean())},(pr,rc,f1,sup)

def epoch_pass(model,loader,criterion,device,nclasses,opt=None,limit=0):
    model.train(opt is not None);total_loss=0.;seen=0;ys=[];ps=[]
    started=time.monotonic()
    with torch.set_grad_enabled(opt is not None):
        for i,(x,y) in enumerate(loader):
            if limit and i>=limit: break
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            if opt is not None: opt.zero_grad(set_to_none=True)
            z=model(x);loss=criterion(z,y)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            if opt is not None: loss.backward();opt.step()
            total_loss+=float(loss.detach().cpu())*len(y);seen+=len(y)
            ys.extend(y.detach().cpu().tolist());ps.extend(z.detach().argmax(1).cpu().tolist())
            if i%25==0:
                elapsed=time.monotonic()-started
                phase='train' if opt is not None else 'eval'
                print(f'  {phase} batch {i+1}/{len(loader)}, loss={float(loss.detach().cpu()):.4f}, '
                      f'images/s={seen/max(elapsed,1e-9):.2f}, elapsed_min={elapsed/60:.1f}',flush=True)
    m,per=metrics(ys,np.array(ps),total_loss/seen,nclasses)
    return m,per,np.array(ys),np.array(ps)

def plot_curves(history,out):
    df=pd.DataFrame(history);df.to_csv(out/'history.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(13,3.3))
    for ax,key in zip(axes,('loss','accuracy','macro_f1')):
        ax.plot(df.epoch,df['train_'+key],label='Train (during updates)')
        ax.plot(df.epoch,df['val_'+key],label='Validation')
        ax.set(xlabel='Epoch',ylabel=key.title());ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)

def final_artifacts(model,train,val,args,ci,out,history):
    criterion=nn.CrossEntropyLoss()
    final_train,*_=epoch_pass(model,make_loader(train,args,False),criterion,args.device,len(ci),limit=2 if args.smoke else 0)
    vm,per,y,p=epoch_pass(model,make_loader(val,args,False),criterion,args.device,len(ci))
    cm=confusion_matrix(y,p,labels=np.arange(len(ci)))
    pd.DataFrame(cm,index=ci.make_model,columns=ci.make_model).to_csv(out/'confusion_matrix_counts.csv')
    norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(12,10));im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
    labels=ci.make_model.tolist();ax.set_xticks(range(len(ci)),labels,rotation=90,fontsize=6)
    ax.set_yticks(range(len(ci)),labels,fontsize=6);ax.set(xlabel='Predicted class',ylabel='True class',title='Validation confusion matrix (row-normalized)')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=180);plt.close(fig)
    pr,rc,f1,sup=per
    table=ci[['model_idx','make_model']].copy();table['precision']=pr;table['recall']=rc;table['f1']=f1;table['support']=sup
    table.to_csv(out/'per_class_metrics.csv',index=False)
    table.sort_values('recall').head(10).to_csv(out/'difficult_classes.csv',index=False)
    predictions=val.df[['rel_path','model_idx','make_model']].copy()
    predictions['prediction_idx']=p;predictions['prediction_label']=ci.make_model.to_numpy()[p]
    predictions['correct']=y==p;predictions.to_csv(out/'validation_predictions.csv',index=False)
    # Deterministic examples, no test images and no cherry-picked claim of improvement.
    chosen=list(np.flatnonzero(y!=p)[:6])+list(np.flatnonzero(y==p)[:2])
    if chosen:
        fig,axes=plt.subplots(2,4,figsize=(12,6));axes=axes.ravel()
        for ax in axes: ax.axis('off')
        for ax,i in zip(axes,chosen):
            with Image.open(val.root/val.df.iloc[i].rel_path) as im: ax.imshow(im.convert('RGB'))
            ax.set_title(f'True: {labels[y[i]]}\nPred: {labels[p[i]]}',fontsize=8)
        fig.tight_layout();fig.savefig(out/'validation_examples.png',dpi=160);plt.close(fig)
    best=max(history,key=lambda r:r['val_macro_f1'])
    summary={'complete':True,'smoke_test_only':args.smoke,'selected_by':'maximum validation macro-F1',
        'best_epoch':int(best['epoch']),'epochs_completed':len(history),'train_at_best_checkpoint':final_train,
        'validation_at_best_checkpoint':vm,'test_evaluated':False}
    write_json(out/'metrics.json',summary)
    lines=['# Task 2 result facts','',f'Smoke test only: {args.smoke}',
           f'Best epoch: {best["epoch"]}; completed epochs: {len(history)}',
           f'Validation accuracy: {vm["accuracy"]:.4f}; macro-F1: {vm["macro_f1"]:.4f}',
           f'Train accuracy at the same checkpoint: {final_train["accuracy"]:.4f}',
           '', 'Interpret learning_curves.png, difficult_classes.csv, confusion_matrix.png, and validation_examples.png.',
           'Training curves use online training metrics; final train metrics use the selected frozen checkpoint.',
           'A single seed does not establish statistical significance. Test set is reserved.',
           'Exact duplicates checked from stored MD5; near-duplicates and all manifest hashes were not re-audited.',
           'Future improvements must use these same data splits and control the baseline settings.']
    (out/'report_facts.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FINAL_METRICS',json.dumps(summary),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=ROOT.parent/'VMMRdb')
    ap.add_argument('--split-dir',type=Path,default=ROOT/'splits')
    ap.add_argument('--run-name',default='baseline_seed0')
    ap.add_argument('--epochs',type=int,default=12);ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--image-size',type=int,default=224);ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--output-root',type=Path,default=PACKAGE/'results')
    ap.add_argument('--device',choices=('auto','cuda','mps','cpu'),default='auto')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();assert re.fullmatch(r'[A-Za-z0-9_-]+',args.run_name),'Invalid run name'
    assert args.epochs>0 and args.batch_size>1 and args.workers>=0 and args.threads>0
    if not args.smoke and (args.epochs,args.image_size,args.lr,args.seed)!=(12,224,3e-4,0):
        ap.error('Task 2 requires exactly 12 epochs, image size 224, Adam lr=0.0003 and seed 0.')
    if not args.smoke and args.batch_size not in (32,64):
        ap.error('Use batch size 64 (or documented memory fallback 32).')
    if args.device=='auto': args.device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if args.smoke: args.epochs=1;args.batch_size=8;args.workers=0
    out=args.output_root.expanduser().resolve()/(args.run_name+('_smoke' if args.smoke else ''))
    if (out/'config.json').exists() and not args.resume: raise RuntimeError(f'Results already exist: {out}. Use --resume or a new --run-name.')
    out.mkdir(parents=True,exist_ok=True)
    original_stdout,original_stderr=sys.stdout,sys.stderr
    log=(out/'run.log').open('a',buffering=1,encoding='utf-8');sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    try:
        torch.set_num_threads(args.threads);seed_all(args.seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        args.data_root=args.data_root.expanduser().resolve();args.split_dir=args.split_dir.expanduser().resolve()
        ci,parts=read_splits(args.split_dir,args.data_root)
        hashes={n:hashlib.sha256((args.split_dir/n).read_bytes()).hexdigest() for n in ('train.csv','val.csv','test.csv','class_info.csv')}
        cfg={'architecture':'torchvision ResNet-18','initialization':'random, weights=None','image_size':args.image_size,
             'batch_size':args.batch_size,'learning_rate':args.lr,'optimizer':'Adam','loss':'unweighted CrossEntropyLoss',
             'epochs':args.epochs,'seed':args.seed,'normalization':{'mean':[.5]*3,'std':[.5]*3},
             'resize':'deterministic square resize','augmentation':False,'transfer_learning':False,
             'rebalancing':False,'hierarchy':False,'scheduler':False,'precision':'float32',
             'smoke':args.smoke,'split_sha256':hashes,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if args.batch_size==32: cfg['batch_size_reason']='README memory fallback: batch size 32 selected explicitly'
        if (out/'config.json').exists(): assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg,'Resume configuration or source changed'
        else: write_json(out/'config.json',cfg)
        ci.to_csv(out/'class_info.csv',index=False);shutil.copy2(__file__,out/'train_source.py')
        sizes={k:len(d) for k,d in parts.items()}
        write_json(out/'data_audit.json',{'split_sizes':sizes,'nclasses':len(ci),'cross_split_stored_md5_overlap':False,
                   'train_hashes_verified':16,'test_images_opened':False,'near_duplicates_checked':False})
        try: revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        except Exception: revision='unavailable'
        write_json(out/'environment.json',{'python':sys.version,'torch':str(torch.__version__),
                   'device':args.device,'gpu':torch.cuda.get_device_name(0) if args.device=='cuda' else None,
                   'workers':args.workers,'cpu_threads':args.threads,'data_root':str(args.data_root),'git_revision':revision,
                   'started_utc':datetime.now(timezone.utc).isoformat()})
        if args.smoke:
            parts={k:d.groupby('model_idx',group_keys=False).sample(n=2 if k=='train' else 1,random_state=0).reset_index(drop=True)
                   for k,d in parts.items() if k!='test'}
        train=Cars(parts['train'],args.data_root,args.image_size);val=Cars(parts['val'],args.data_root,args.image_size)
        # No pretrained model or external dataset is downloaded.
        model=resnet18(weights=None,num_classes=len(ci)).to(args.device)
        opt=torch.optim.Adam(model.parameters(),lr=args.lr);criterion=nn.CrossEntropyLoss()
        ckdir=out/'checkpoints';ckdir.mkdir(exist_ok=True)
        history=[];first=0;best=-1.
        if args.resume and (ckdir/'last.pt').exists():
            ck=torch.load(ckdir/'last.pt',map_location=args.device,weights_only=False)
            model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
            history=ck['history'];first=ck['epoch']+1;best=ck['best']
            print('RESUME at epoch',first+1,flush=True)
        write_json(out/'status.json',{'state':'running','smoke':args.smoke})
        print('TASK2_BASELINE_START',cfg,'sizes',sizes,flush=True)
        for epoch in range(first,args.epochs):
            seed_all(args.seed+epoch);start=time.monotonic()
            tm,*_=epoch_pass(model,make_loader(train,args,True,epoch),criterion,args.device,len(ci),opt,limit=2 if args.smoke else 0)
            vm,*_=epoch_pass(model,make_loader(val,args,False),criterion,args.device,len(ci))
            row={'epoch':epoch+1,'minutes':(time.monotonic()-start)/60,
                 **{'train_'+k:v for k,v in tm.items()},**{'val_'+k:v for k,v in vm.items()}}
            history.append(row);plot_curves(history,out)
            if vm['macro_f1']>best:
                best=vm['macro_f1'];tmp=ckdir/'best.tmp';torch.save(model.state_dict(),tmp);tmp.replace(ckdir/'best.pt')
            tmp=ckdir/'last.tmp';torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),
                          'history':history,'epoch':epoch,'best':best},tmp);tmp.replace(ckdir/'last.pt')
            write_json(out/'status.json',{'state':'running','epoch_completed':epoch+1,'best_macro_f1':best})
            print('EPOCH_RESULT',json.dumps(row),flush=True)
        model.load_state_dict(torch.load(ckdir/'best.pt',map_location=args.device,weights_only=True))
        final_artifacts(model,train,val,args,ci,out,history)
        from report import generate_report
        generate_report(out)
        write_json(out/'status.json',{'state':'complete','smoke':args.smoke,'updated_utc':datetime.now(timezone.utc).isoformat()})
        from verify_results import verify_run
        verification=verify_run(out)
        if not verification['passed']: raise RuntimeError(f"Result verification failed: {verification['errors']}")
        print('TASK2_COMPLETE',out,flush=True)
    except BaseException as e:
        write_json(out/'status.json',{'state':'failed','error':repr(e)})
        traceback.print_exc();raise
    finally:
        sys.stdout,sys.stderr=original_stdout,original_stderr
        log.close()

if __name__=='__main__':main()
