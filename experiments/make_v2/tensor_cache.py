"""Lossless decoded RGB uint8 cache and bounded Stage 0.5 benchmark; no test access."""
from pathlib import Path
import os,json,time,hashlib,shutil,statistics
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader
import full_train as ft
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'dataset/VMMRdb';BASE=ROOT/'runs_full/make_v2';REPORT=ROOT/'result_full/make_v2/cache_benchmark'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_frames():
 ci=pd.read_csv(ROOT/'splits/class_info.csv').sort_values('model_idx');tr=pd.read_csv(ROOT/'splits/train.csv');va=pd.read_csv(ROOT/'splits/val.csv')
 assert len(tr)==82929 and len(va)==17771 and len(ci)==50
 expected=json.loads((ROOT/'result_full/make_v2/stage0/protocol.json').read_text())['split_sha256']
 for name,h in expected.items():assert sha(ROOT/'splits'/name)==h
 return ci,tr,va
class RGBStore:
 def __init__(self,path):
  self.path=Path(path);self.index=json.loads((self.path/'index.json').read_text());self.mm=None
 def __getstate__(self):return dict(self.__dict__,mm=None)
 def get(self,rel):
  if self.mm is None:self.mm=np.memmap(self.path/'pixels.bin',mode='r',dtype=np.uint8)
  r=self.index[rel];start,n,h,w=r
  # Copy prevents transforms from modifying the readonly cache.
  return torch.from_numpy(np.array(self.mm[start:start+n],copy=True).reshape(3,h,w))
 def pil(self,rel):return Image.fromarray(self.get(rel).permute(1,2,0).numpy(),'RGB')
def build_cache(df,path):
 path=Path(path);path.mkdir(parents=True,exist_ok=True);idxp=path/'index.json';index=json.loads(idxp.read_text()) if idxp.exists() else {}
 desired=df.rel_path.tolist();assert len(set(desired))==len(desired)
 if (path/'complete.json').exists():
  cfg=json.loads((path/'complete.json').read_text());assert cfg['paths_sha256']==hashlib.sha256(chr(10).join(desired).encode()).hexdigest();return RGBStore(path)
 offset=max((r[0]+r[1] for r in index.values()),default=0);binary=path/'pixels.bin';started=time.monotonic()
 with binary.open('r+b' if binary.exists() else 'w+b') as f:
  f.truncate(offset);f.seek(offset)
  for i,rel in enumerate(desired):
   if rel in index:continue
   with Image.open(DATA/rel) as src:rgb=np.asarray(src.convert('RGB'))
   h,w,_=rgb.shape;flat=np.ascontiguousarray(rgb.transpose(2,0,1)).reshape(-1);f.write(flat.tobytes());index[rel]=[offset,len(flat),h,w];offset+=len(flat)
   if (i+1)%256==0:
    f.flush();os.fsync(f.fileno());ft.atomic_json(idxp,index);print('CACHE_BUILD',path.name,i+1,len(desired),round(offset/1e9,3),'GB',flush=True)
  f.flush();os.fsync(f.fileno());ft.atomic_json(idxp,index)
 store=RGBStore(path)
 for i,rel in enumerate(desired):
  with Image.open(DATA/rel) as src:rgb=np.array(src.convert('RGB'))
  assert np.array_equal(store.get(rel).permute(1,2,0).numpy(),rgb),rel
  if (i+1)%5000==0:print('CACHE_VERIFY',i+1,len(desired),flush=True)
 ft.atomic_json(path/'complete.json',dict(n=len(desired),bytes=offset,seconds=time.monotonic()-started,dtype='uint8',layout='CHW',original_resolution=True,pixel_verified=len(desired),paths_sha256=hashlib.sha256(chr(10).join(desired).encode()).hexdigest(),decoder='PIL '+Image.__version__))
 return store
class CachedCars(Dataset):
 def __init__(self,df,train,store=None):
  self.df=df.reset_index(drop=True);self.store=store;self.reference=ft.Cars(self.df,DATA,train);self.tf=self.reference.tf
 def __len__(self):return len(self.df)
 def __getitem__(self,i):
  if self.store is None:return self.reference[i]
  row=self.df.iloc[i];return self.tf(self.store.pil(row.rel_path)),int(row.model_idx)
def benchmark():
 REPORT.mkdir(parents=True,exist_ok=True);ci,tr,va=load_frames();torch.set_num_threads(2)
 sample=tr.sample(512,random_state=20261022).sort_index().reset_index(drop=True)
 v=va.sample(32,random_state=20261022).sort_index().reset_index(drop=True)
 sample.to_csv(REPORT/'train_sample.csv',index=False);v.to_csv(REPORT/'validation_sample.csv',index=False)
 start=time.monotonic();store=build_cache(pd.concat([sample,v],ignore_index=True),BASE/'cache/sample');buildsecs=time.monotonic()-start
 # Compare all augmented tensors under identical RNG, keeping original PIL transforms.
 a=CachedCars(sample,True,None);b=CachedCars(sample,True,store)
 for i in range(32):
  ft.seed_all(20261022+i);x,y=a[i];ft.seed_all(20261022+i);z,q=b[i];assert y==q and torch.equal(x,z)
 aa=CachedCars(v,False,None);bb=CachedCars(v,False,store)
 x=torch.stack([aa[i][0] for i in range(len(v))]);z=torch.stack([bb[i][0] for i in range(len(v))]);assert torch.equal(x,z)
 model=ft.CarModel(ci.make_idx.to_numpy(),pretrained=False).cuda().eval();model.load_state_dict(torch.load(ROOT/'runs_full/make_v1/seed0/flat/best.pt',map_location='cuda',weights_only=True))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):
  p=model(x.cuda(),False)[0].argmax(1);q=model(z.cuda(),False)[0].argmax(1);assert torch.equal(p,q)
 del x,z
 # One warm-up + three timed runs, alternate path ordering.
 rows=[]
 for repeat in range(4):
  for mode in (['jpeg','cache'] if repeat%2==0 else ['cache','jpeg']):
   ft.seed_all(20261022);ds=CachedCars(sample,True,store if mode=='cache' else None)
   loader=DataLoader(ds,batch_size=64,shuffle=True,num_workers=2,pin_memory=True,prefetch_factor=2,worker_init_fn=ft.worker_seed,generator=torch.Generator().manual_seed(20261022))
   torch.cuda.synchronize();t0=time.monotonic();waiting=0.;seen=0;previous=t0
   for x,y in loader:
    now=time.monotonic();waiting+=now-previous;x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True)
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):loss=torch.nn.functional.nll_loss(model(x,False)[0],y)
    loss.backward();torch.cuda.synchronize();previous=time.monotonic();seen+=len(y)
   seconds=time.monotonic()-t0;rows.append(dict(repeat=repeat,warmup=repeat==0,path=mode,n=seen,seconds=seconds,images_per_second=seen/seconds,loader_wait_seconds=waiting))
   print('BENCH',rows[-1],flush=True)
 del model;torch.cuda.empty_cache();pd.DataFrame(rows).to_csv(REPORT/'timings.csv',index=False)
 d=pd.DataFrame(rows);timed=d[~d.warmup];p=timed.pivot(index='repeat',columns='path',values='seconds');speed=p.jpeg/p.cache
 # Measure original decoded byte size for all train/val, headers only; no test.
 dims=[]
 for rel in pd.concat([tr,va]).rel_path:
  with Image.open(DATA/rel) as im:dims.append(im.width*im.height*3)
 estimate=int(sum(dims));free=shutil.disk_usage(BASE).free
 meanjpeg=timed[timed.path=='jpeg'].seconds.mean();meancache=timed[timed.path=='cache'].seconds.mean()
 # Conservative: require >=5% total batch-loop gain in every paired repeat.
 adopt=bool((speed>1.05).all() and free>estimate*2)
 result=dict(state='complete',cache_selected=adopt,paired_speedups=speed.tolist(),median_speedup=float(speed.median()),sample_build_seconds=buildsecs,estimated_full_bytes=estimate,filesystem_free_bytes=free,estimated_build_seconds=buildsecs*(len(tr)+len(va))/(len(sample)+len(v)),estimated_train_epoch_saving_seconds=(meanjpeg-meancache)*len(tr)/len(sample),augmented_input_exact_matches=32,validation_input_exact_matches=32,validation_prediction_exact_matches=32,optimizer_steps=0,test_read=False,workers=2,batch_size=64,torch_version=torch.__version__,decoder='PIL '+Image.__version__,note='End-to-end forward/backward benchmark, no optimizer step; warm OS cache after warmup. Filesystem free space is not a per-user quota guarantee. Full build stops safely on quota/write failure.')
 ft.atomic_json(REPORT/'benchmark.json',result)
 (REPORT/'README.md').write_text('# Tensor cache benchmark'+chr(10)+json.dumps(result,indent=2,ensure_ascii=False)+chr(10)+'PIL transforms and augmentation RNG preserved. Cache stores lossless original RGB uint8, not augmented samples. GPU model remains in eval mode and receives no optimizer step. Cache selected only if every paired timing is >5% faster. No classification improvement is claimed.')
 print('BENCHMARK_COMPLETE',json.dumps(result),flush=True)
 return result
