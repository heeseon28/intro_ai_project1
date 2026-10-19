"""Single-seed pilot: flat vs soft hierarchy vs make-weighted hierarchy.
Only VMMRdb train/validation are used for fitting/model selection. Test is reserved.
Resume restarts an interrupted epoch from its last completed checkpoint.
"""
import argparse, hashlib, json, math, os, random, shutil, time, traceback, fcntl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ('flat', 'hierarchy', 'hierarchy_make_weighted')

def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def worker_seed(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed); random.seed(seed); torch.set_num_threads(1)

class Cars(Dataset):
    def __init__(self, df, root, train):
        self.paths = [root / p for p in df.rel_path]
        self.labels = df.model_idx.to_numpy(dtype=np.int64)
        ops = ([transforms.RandomResizedCrop(224, scale=(.6, 1.)), transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(.2,.2,.2,.05)] if train else
               [transforms.Resize(256), transforms.CenterCrop(224)])
        self.tf = transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(
            (.485,.456,.406), (.229,.224,.225))])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im: x = self.tf(im.convert('RGB'))
        except Exception as e:
            raise RuntimeError(f'Cannot decode {self.paths[i]}') from e
        return x, int(self.labels[i])

class CarModel(nn.Module):
    def __init__(self, mapping, pretrained=True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        width = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.fine = nn.Linear(width, len(mapping))
        self.make = nn.Linear(width, int(mapping.max())+1)
        self.register_buffer('mapping', torch.tensor(mapping, dtype=torch.long))
        self.groups = [np.flatnonzero(mapping == g).tolist() for g in range(int(mapping.max())+1)]
    def forward(self, x, hierarchical):
        h = self.backbone(x); logits = self.fine(h).float()
        if not hierarchical:
            logp = F.log_softmax(logits, dim=1)
            makes = torch.stack([logp[:, ids].exp().sum(1) for ids in self.groups], dim=1)
            return logp, makes
        make_logp = F.log_softmax(self.make(h).float(), dim=1)
        # p(model|image) = p(make|image) * p(model|make,image).
        # All makes retain nonzero probability; no ground-truth make enters inference.
        within = torch.empty_like(logits)
        for ids in self.groups:
            within[:, ids] = F.log_softmax(logits[:, ids], dim=1)
        return within + make_logp[:, self.mapping], make_logp.exp()

def validate(data, split_dir):
    ci = pd.read_csv(split_dir/'class_info.csv').sort_values('model_idx').reset_index(drop=True)
    assert ci.model_idx.tolist() == list(range(len(ci)))
    mapping = ci.make_idx.to_numpy(dtype=np.int64)
    assert sorted(set(mapping)) == list(range(int(mapping.max())+1))
    assert ci.groupby('make_idx').make.nunique().max() == 1
    assert ci.groupby('make').make_idx.nunique().max() == 1
    frames = {k: pd.read_csv(split_dir/f'{k}.csv') for k in ('train','val','test')}
    for k, df in frames.items():
        assert not df[['rel_path','md5','model_idx','make_idx','make_model']].isna().any().any(), k
        assert df.rel_path.is_unique, f'duplicate paths in {k}'
        assert set(df.model_idx) == set(range(len(ci))), f'missing class in {k}'
        ids = df.model_idx.to_numpy(dtype=int)
        assert np.array_equal(mapping[ids], df.make_idx.to_numpy()), f'make mismatch: {k}'
        assert np.array_equal(ci.make_model.to_numpy()[ids],df.make_model.to_numpy()), f'label mismatch: {k}'
        assert (df.rel_path.map(lambda p: not Path(p).is_absolute() and '..' not in Path(p).parts)).all()
        with ThreadPoolExecutor(max_workers=4) as pool:
            exists = list(pool.map(lambda p: (data/p).is_file(), df.rel_path))
        assert all(exists), f'missing image in {k}'
    for a,b in (('train','val'),('train','test'),('val','test')):
        for col in ('rel_path','md5'):
            assert not (set(frames[a][col]) & set(frames[b][col])), f'{col} leakage {a}/{b}'
    for row in frames['train'].sample(n=min(32,len(frames['train'])),random_state=9).itertuples():
        p = data / row.rel_path
        assert hashlib.md5(p.read_bytes()).hexdigest() == row.md5, f'stale MD5: {p}'
        with Image.open(p) as im: im.verify()
    counts = np.bincount(frames['train'].make_idx, minlength=int(mapping.max())+1)
    assert (counts>0).all()
    print('PREFLIGHT_OK', {k:len(v) for k,v in frames.items()}, 'classes', len(ci), flush=True)
    print('TRAIN make counts:', counts.tolist(), flush=True)
    return ci, mapping, frames, counts

def evaluate(model, loader, hierarchical, mapping, ci, limit=0):
    model.eval(); ys=[]; preds=[]; make_preds=[]; nll=0.; total=0
    with torch.inference_mode():
        for step,(x,y) in enumerate(loader):
            if limit and step>=limit: break
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported()):
                lp, mp = model(x,hierarchical)
            nll += F.nll_loss(lp,y,reduction='sum').item(); total += len(y)
            ys.extend(y.cpu().tolist()); preds.extend(lp.argmax(1).cpu().tolist()); make_preds.extend(mp.argmax(1).cpu().tolist())
    y=np.array(ys); p=np.array(preds); gm=np.array(make_preds)
    prec,rec,f1,support=precision_recall_fscore_support(y,p,labels=np.arange(len(ci)),zero_division=0)
    true_make=mapping[y]; pred_make=mapping[p]
    make_rows=[]
    for g in range(int(mapping.max())+1):
        mask=true_make==g
        make_rows.append({'make_idx':g,'make':str(ci.loc[ci.make_idx==g,'make'].iloc[0]),
                          'n':int(mask.sum()),'fine_accuracy':float((p[mask]==y[mask]).mean()) if mask.any() else 0.,
                          'make_accuracy':float((gm[mask]==g).mean()) if mask.any() else 0.})
    result={'accuracy':float((y==p).mean()),'macro_f1':float(f1.mean()),'balanced_accuracy':float(rec.mean()),
            'make_accuracy':float((gm==true_make).mean()),'cross_make_error_rate':float((pred_make!=true_make).mean()),
            'within_make_error_rate':float(((p!=y)&(pred_make==true_make)).mean()),'loss':nll/total,
            'per_class':[{'model_idx':i,'label':str(ci.make_model.iloc[i]),'precision':float(prec[i]),
                          'recall':float(rec[i]),'f1':float(f1[i]),'support':int(support[i])} for i in range(len(ci))],
            'per_make':make_rows}
    return result,confusion_matrix(y,p,labels=np.arange(len(ci))),pd.DataFrame({'target':y,'prediction':p,'make_prediction':gm})

def run_experiment(name, args, ci, mapping, frames, make_counts, out):
    folder=out/name; folder.mkdir(exist_ok=True)
    if (folder/'metrics.json').exists():
        print('SKIP completed',name,flush=True); return json.loads((folder/'metrics.json').read_text())
    seed_all(args.seed)
    model=CarModel(mapping,pretrained=not args.smoke).cuda()
    hierarchical=name!='flat'
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    # Mean per-image weight = 1 over training data. No validation/test counts are used.
    mw=1/np.sqrt(make_counts.astype(float)); mw/=np.average(mw,weights=make_counts)
    class_weights=torch.tensor(mw[mapping],dtype=torch.float32,device='cuda')
    tr=Cars(frames['train'],args.data_root,True); va=Cars(frames['val'],args.data_root,False)
    def loader(ds,train,epoch):
        gen=torch.Generator().manual_seed(args.seed+epoch)
        return DataLoader(ds,batch_size=args.batch_size,shuffle=train,num_workers=args.workers,
                          pin_memory=True,prefetch_factor=2 if args.workers else None,
                          worker_init_fn=worker_seed,generator=gen)
    best=-1.; bad=0; first=0; history=[]
    last=folder/'last.pt'
    if last.exists():
        ck=torch.load(last,map_location='cuda',weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        best=ck['best'];bad=ck['bad'];first=ck['epoch']+1;history=ck['history']
        print('RESUME',name,'epoch',first+1,flush=True)
    model.eval()
    with torch.no_grad():
        lp,_=model(torch.zeros(2,3,224,224,device='cuda'),hierarchical)
        assert torch.allclose(lp.exp().sum(1),torch.ones(2,device='cuda'),atol=1e-5)
        if name.endswith('weighted'): assert class_weights.std()>0, 'no manufacturer imbalance'
    print('START',name,'parameters',sum(p.numel() for p in model.parameters()),flush=True)
    for epoch in range(first,args.epochs):
        if bad>=args.patience: break
        seed_all(args.seed+epoch); model.train(); start=time.monotonic(); loss_sum=0.; seen=0
        for step,(x,y) in enumerate(loader(tr,True,epoch)):
            if args.smoke and step>=2: break
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
            lr=3e-4*.5*(1+math.cos(math.pi*epoch/args.epochs))
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
                lp,_=model(x,hierarchical)
                losses=F.nll_loss(lp,y,reduction='none')
                w=class_weights[y] if name.endswith('weighted') else torch.ones_like(losses)
                loss=(losses*w).sum()/w.sum()
            assert torch.isfinite(loss), 'non-finite loss'
            loss.backward(); opt.step();loss_sum+=loss.item()*len(y);seen+=len(y)
            if step%100==0:
                print(name,f'epoch={epoch+1}/{args.epochs} batch={step+1}/{math.ceil(len(tr)/args.batch_size)} loss={loss.item():.4f}',flush=True)
        metrics,cm,pred=evaluate(model,loader(va,False,epoch),hierarchical,mapping,ci,limit=2 if args.smoke else 0)
        # Minority makes are the lowest third by TRAIN count, fixed before evaluation.
        minority=np.argsort(make_counts,kind='stable')[:max(1,len(make_counts)//3)]
        metrics['minority_make_fine_accuracy']=float(np.mean([metrics['per_make'][int(g)]['fine_accuracy'] for g in minority]))
        row={'epoch':epoch+1,'train_loss':loss_sum/seen,'minutes':(time.monotonic()-start)/60,
             **{k:v for k,v in metrics.items() if isinstance(v,(int,float))}}
        history.append(row); pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        print('EPOCH_RESULT',name,json.dumps(row),flush=True)
        if metrics['macro_f1']>best:
            best=metrics['macro_f1'];bad=0
            temp=folder/'best.tmp';torch.save(model.state_dict(),temp);temp.replace(folder/'best.pt')
            atomic_json(folder/'best_validation.json',metrics)
            np.save(folder/'validation_confusion.npy',cm);pred.to_csv(folder/'validation_predictions.csv',index=False)
        else: bad+=1
        temp=folder/'last.tmp'
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,
                    'bad':bad,'history':history},temp);temp.replace(last)
        atomic_json(out/'status.json',{'state':'running','experiment':name,'completed_epoch':epoch+1,'best_macro_f1':best,'updated':time.time()})
    metrics=json.loads((folder/'best_validation.json').read_text())
    metrics.update(experiment=name,seed=args.seed,split='validation',smoke=args.smoke,epochs_completed=len(history))
    atomic_json(folder/'metrics.json',metrics)
    del model,opt; torch.cuda.empty_cache()
    return metrics

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,default=ROOT/'dataset/VMMRdb')
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--patience',type=int,default=4);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    lock_root=ROOT/'runs_pilot';lock_root.mkdir(exist_ok=True)
    gpu_lock=open(lock_root/'.gpu.lock','a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.data_root=args.data_root.resolve(); torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'CUDA required'
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark=False
    if args.smoke: args.epochs=1
    source=ROOT/'splits'
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob('*.csv'))}
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    cfg.update(split_sha256=hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               backbone='resnet18',pretraining='random for smoke; ImageNet1K_V1 for pilot',
               experiments=list(EXPERIMENTS),weight='inverse sqrt TRAIN manufacturer frequency',
               device=torch.cuda.get_device_name(0),torch_version=torch.__version__,test_policy='reserved; no test metrics')
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text())==cfg, 'Resume config/source/splits changed; use a new output directory'
    else: atomic_json(out/'config.json',cfg)
    snap=out/'splits';snap.mkdir(exist_ok=True)
    for p in source.glob('*.csv'): shutil.copy2(p,snap/p.name)
    shutil.copy2(Path(__file__),out/'pilot_train_source.py')
    try:
        atomic_json(out/'status.json',{'state':'validating','updated':time.time()})
        ci,mapping,frames,counts=validate(args.data_root,snap)
        atomic_json(out/'data_audit.json',{'sizes':{k:len(v) for k,v in frames.items()},'classes':len(ci),
                    'make_train_counts':counts.tolist(),'make_weights':(1/np.sqrt(counts)).tolist(),
                    'duplicate_check':'cross-split paths and stored MD5; 32 TRAIN hashes verified; near-duplicates not audited'})
        if args.check_only: print('CHECK_COMPLETE',flush=True);return
        if args.smoke:
            frames={k:d.groupby('model_idx',group_keys=False).sample(n=3,random_state=args.seed).reset_index(drop=True)
                    for k,d in frames.items() if k!='test'}
        results=[]
        for name in EXPERIMENTS:
            r=run_experiment(name,args,ci,mapping,frames,counts,out);results.append(r)
            pd.DataFrame([{k:v for k,v in r.items() if isinstance(v,(int,float,str,bool))} for r in results]).to_csv(out/'summary.csv',index=False)
        atomic_json(out/'status.json',{'state':'complete','updated':time.time(),'smoke':args.smoke})
        print('ALL_COMPLETE',out,flush=True)
    except BaseException as e:
        atomic_json(out/'status.json',{'state':'failed','error':repr(e),'updated':time.time()})
        raise

if __name__=='__main__': main()
