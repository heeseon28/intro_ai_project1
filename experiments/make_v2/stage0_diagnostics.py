"""Stage 0 only: validation diagnostics, no training, no test access.
Run from repository root. Originals/contact sheets/cache stay under ignored runs_full.
"""
from pathlib import Path
import os,sys,json,time,hashlib,fcntl,argparse
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageOps,ImageFont
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
RAW=ROOT/'runs_full/make_v2/stage0'
OUT=ROOT/'result_full/make_v2/stage0'
DATA=ROOT/'dataset/VMMRdb'
NAMES=['flat','hierarchy','hierarchy_make_weighted']

def savejson(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));tmp.replace(path)
def frames():
 va=pd.read_csv(ROOT/'splits/val.csv');tr=pd.read_csv(ROOT/'splits/train.csv');ci=pd.read_csv(ROOT/'splits/class_info.csv').sort_values('model_idx')
 assert len(va)==17771 and len(tr)==82929 and len(ci)==50
 assert np.array_equal(ci.model_idx,np.arange(50))
 return tr,va,ci,ci.make_idx.to_numpy(int)
def predictions(va):
 result={}
 for seed in range(3):
  for name in NAMES:
   p=ROOT/f'runs_full/make_v1/seed{seed}'/name/'validation_predictions.csv'
   d=pd.read_csv(p);assert np.array_equal(d.target,va.model_idx)
   # Validate row identities against the archived split, not merely target labels.
   archived=pd.read_csv(ROOT/f'runs_full/make_v1/seed{seed}/splits/val.csv')
   assert archived.rel_path.tolist()==va.rel_path.tolist()
   result[(seed,name)]=d.prediction.to_numpy(int)
 return result

def crop(im):
 w,h=im.size
 if w<h: nw,nh=256,int(256*h/w)
 else: nh,nw=256,int(256*w/h)
 left=int(round((nw-224)/2));top=int(round((nh-224)/2))
 resized=im.resize((nw,nh),Image.Resampling.BILINEAR)
 box=(left*w/nw,top*h/nh,(left+224)*w/nw,(top+224)*h/nh)
 return resized.crop((left,top,left+224,top+224)),box,224/nw,224/nh

def prepare():
 tr,va,ci,mapping=frames();ps=predictions(va)
 correct=np.stack([p==va.model_idx.to_numpy() for p in ps.values()])
 wrong=np.flatnonzero(~correct.any(0));good=np.flatnonzero(correct.all(0))
 assert len(wrong)==321
 rng=np.random.default_rng(20261022);controls=np.sort(rng.choice(good,100,replace=False))
 rows=[]
 for group,indices in [('all9_wrong',wrong),('all9_correct_control',controls)]:
  for i in indices:
   r=va.iloc[i];rows.append({'val_index':int(i),'group':group,'rel_path':r.rel_path,'model_idx':int(r.model_idx),'make_model':r.make_model})
 audit=pd.DataFrame(rows).sort_values(['group','model_idx','val_index']).reset_index(drop=True)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14) if Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf').exists() else ImageFont.load_default()
 (RAW/'contact_sheets').mkdir(parents=True,exist_ok=True)
 for page,start in enumerate(range(0,len(audit),12),1):
  sheet=Image.new('RGB',(1500,1000),'white');draw=ImageDraw.Draw(sheet)
  draw.text((8,4),f'Stage0 page {page}: original + RED validation crop boundary | actual center crop; sorted by group/class',fill='black',font=font)
  for slot,j in enumerate(range(start,min(start+12,len(audit)))):
   row=audit.iloc[j];x=(slot%3)*500;y=30+(slot//3)*240
   with Image.open(DATA/row.rel_path) as src: im=src.convert('RGB')
   cc,box,fw,fh=crop(im)
   annotated=im.copy();di=ImageDraw.Draw(annotated);di.rectangle(box,outline='red',width=max(2,im.width//140))
   thumb=ImageOps.contain(annotated,(280,190));ct=ImageOps.contain(cc,(190,190))
   sheet.paste(thumb,(x,y+40));sheet.paste(ct,(x+290,y+40))
   draw.text((x+3,y),f'#{j:03d} val={row.val_index} {row.make_model}',fill='black',font=font)
   draw.text((x+3,y+18),f'{row.group} {im.width}x{im.height}',fill='black',font=font)
   for k,v in [('audit_id',j),('page',page),('slot',slot),('width',im.width),('height',im.height),('retained_width',fw),('retained_height',fh)]:audit.loc[j,k]=v
  sheet.save(RAW/f'contact_sheets/page_{page:02d}.png')
 for field in ['primary_category','secondary_categories','crop_cuts_visible_cue','cue_type','confidence','notes','reviewed_by']:audit[field]=''
 audit.to_csv(RAW/'visual_audit_template.csv',index=False)
 audit.drop(columns=['rel_path']).to_csv(OUT/'audit_manifest.csv',index=False)
 savejson(OUT/'protocol.json',{'stage':0,'training':False,'test_read':False,'val_n':len(va),'train_n':len(tr),'all9_wrong':len(wrong),'controls':100,'control_definition':'correct in ALL 9 V1 runs; seed20261022 without replacement','hash':'64-bit horizontal dHash, grayscale PIL Lanczos 9x8, adjacent > comparison','hamming_radius':4,'duplicate_scope':'train vs validation only','duplicate_note':'hash candidates, not proof of identity; subset scores are a sensitivity analysis, not replacement official validation','oracle_note':'true make supplied diagnostically; never deployable performance','audit_categories':['suspected_label_error','multiple_vehicles','partial_vehicle','occlusion','low_resolution','unusual_view','sibling_or_trim_ambiguous','other'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'split_sha256':{n:hashlib.sha256((ROOT/'splits'/n).read_bytes()).hexdigest() for n in ['train.csv','val.csv','class_info.csv']}})
 print('PREPARED',len(audit),'images',page,'sheets',flush=True)

def metrics(y,p,m):
 from sklearn.metrics import f1_score
 return {'n':len(y),'accuracy_pct':100*float((y==p).mean()),'macro_f1_pct':100*float(f1_score(y,p,labels=np.arange(50),average='macro',zero_division=0)),'within_error_pct_val':100*float(((y!=p)&(m[y]==m[p])).mean()),'cross_error_pct_val':100*float((m[y]!=m[p]).mean())}

def oracle():
 import torch,full_train as ft
 from torch.utils.data import DataLoader,Subset
 torch.set_num_threads(2)
 tr,va,ci,m=frames();ps=predictions(va);keys=list(ps);y=va.model_idx.to_numpy(int)
 models=[]
 for seed,name in keys:
  model=ft.CarModel(m,pretrained=False)
  ck=ROOT/f'runs_full/make_v1/seed{seed}'/name/'best.pt'
  model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True),strict=True)
  models.append(model.cuda().eval())
 normal=np.full((9,len(va)),-1,dtype=np.int16);forced=normal.copy();offset=0
 cache=RAW/'oracle_progress.npz'
 if cache.exists():
  z=np.load(cache);normal=z['normal'];forced=z['forced'];offset=int(z['offset'])
 loader=DataLoader(Subset(ft.Cars(va,DATA,False),range(offset,len(va))),batch_size=64,shuffle=False,num_workers=2,pin_memory=True)
 with torch.inference_mode():
  for step,(x,labels) in enumerate(loader):
   x=x.cuda(non_blocking=True);truth=torch.as_tensor(m[labels.numpy()],device='cuda');valid=torch.as_tensor(m,device='cuda')[None,:]==truth[:,None]
   for k,((seed,name),model) in enumerate(zip(keys,models)):
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=torch.cuda.is_bf16_supported()):lp,_=model(x,name!='flat')
    normal[k,offset:offset+len(labels)]=lp.argmax(1).cpu().numpy()
    # Manufacturer term is constant within the true manufacturer, so masked joint argmax equals within argmax.
    forced[k,offset:offset+len(labels)]=lp.masked_fill(~valid,float('-inf')).argmax(1).cpu().numpy()
   offset+=len(labels)
   if step%20==0 or offset==len(va):
    with (RAW/'oracle_progress.tmp').open('wb') as f:np.savez_compressed(f,normal=normal,forced=forced,offset=offset)
    (RAW/'oracle_progress.tmp').replace(cache)
    savejson(RAW/'oracle_status.json',{'state':'running','validation_images_done':offset,'total':len(va),'updated':time.time()});print('ORACLE',offset,'/',len(va),flush=True)
 rows=[]
 for k,(seed,name) in enumerate(keys):
  mismatch=int((normal[k]!=ps[(seed,name)]).sum())
  assert (m[forced[k]]==m[y]).all()
  for mode,p in [('saved_normal',ps[(seed,name)]),('rerun_normal',normal[k]),('oracle_true_make',forced[k])]:rows.append({'seed':seed,'model':name,'mode':mode,'normal_reproduction_mismatches':mismatch,**metrics(y,p,m)})
  pd.DataFrame({'val_index':np.arange(len(va)),'target':y,'normal':normal[k],'oracle_true_make':forced[k]}).to_csv(OUT/f'oracle_seed{seed}_{name}.csv',index=False)
 pd.DataFrame(rows).to_csv(OUT/'oracle_metrics.csv',index=False)
 savejson(RAW/'oracle_status.json',{'state':'complete','validation_images_done':len(va),'normal_mismatches':[int((normal[k]!=ps[key]).sum()) for k,key in enumerate(keys)],'updated':time.time()})
 print('ORACLE_COMPLETE',flush=True)

def dhash(im):
 a=np.asarray(im.convert('L').resize((9,8),Image.Resampling.LANCZOS));bits=(a[:,1:]>a[:,:-1]).ravel();return int.from_bytes(np.packbits(bits).tobytes(),'big')
def hashes():
 import sqlite3
 from concurrent.futures import ThreadPoolExecutor
 tr,va,ci,m=frames();db=sqlite3.connect(RAW/'image_hash_cache.sqlite');db.execute('create table if not exists hashes(path text primary key,h text,w integer,height integer)')
 cached={r[0] for r in db.execute('select path from hashes')}
 paths=list(dict.fromkeys(tr.rel_path.tolist()+va.rel_path.tolist()));todo=[p for p in paths if p not in cached]
 def one(p):
  with Image.open(DATA/p) as src:
   im=src.convert('RGB');return p,f'{dhash(im):016x}',im.width,im.height
 # Tests use deterministic synthetic arrays, not test split images.
 a=Image.fromarray(np.tile(np.arange(9,dtype=np.uint8)*20,(8,1)));assert dhash(a)==(1<<64)-1 and dhash(a)==dhash(a.copy())
 with ThreadPoolExecutor(max_workers=2) as ex:
  for i,row in enumerate(ex.map(one,todo),1):
   db.execute('insert or replace into hashes values(?,?,?,?)',row)
   if i%1000==0:db.commit();print('HASH',len(cached)+i,'/',len(paths),flush=True)
 db.commit();records={p:(int(h,16),w,hg) for p,h,w,hg in db.execute('select * from hashes')};db.close()
 # Five disjoint bands guarantee every Hamming-radius-4 neighbor shares at least one exact band.
 bands=[(0,13),(13,13),(26,13),(39,13),(52,12)];index={}
 for j,p in enumerate(tr.rel_path):
  h=records[p][0]
  for b,(shift,width) in enumerate(bands):index.setdefault((b,(h>>shift)&((1<<width)-1)),[]).append(j)
 pairs=[]
 for i,p in enumerate(va.rel_path):
  h=records[p][0];candidates=set()
  for b,(shift,width) in enumerate(bands):candidates.update(index.get((b,(h>>shift)&((1<<width)-1)),[]))
  for j in candidates:
   dist=(h^records[tr.rel_path.iloc[j]][0]).bit_count()
   if dist<=4:pairs.append({'val_index':i,'train_index':j,'hamming_distance':dist,'same_label':bool(va.model_idx.iloc[i]==tr.model_idx.iloc[j])})
  if i%2000==0:print('MATCH',i,'/',len(va),'pairs',len(pairs),flush=True)
 pairdf=pd.DataFrame(pairs,columns=['val_index','train_index','hamming_distance','same_label']).sort_values(['hamming_distance','val_index','train_index'])
 pairdf.to_csv(OUT/'near_duplicate_candidates.csv',index=False)
 excluded=set(pairdf.val_index);keep=np.array([i not in excluded for i in range(len(va))]);y=va.model_idx.to_numpy(int);rows=[]
 for (seed,name),p in predictions(va).items():
  for subset,mask in [('full_validation',np.ones(len(va),bool)),('excluding_dhash_candidates',keep)]:rows.append({'seed':seed,'model':name,'subset':subset,**metrics(y[mask],p[mask],m)})
 pd.DataFrame(rows).to_csv(OUT/'duplicate_sensitivity_metrics.csv',index=False)
 geometry=[]
 for i,p in enumerate(va.rel_path):
  _,w,h=records[p];nw=256 if w<h else int(256*w/h);nh=256 if h<w else int(256*h/w)
  geometry.append({'val_index':i,'width':w,'height':h,'aspect_ratio':w/h,'width_fraction_removed':1-224/nw,'height_fraction_removed':1-224/nh})
 geo=pd.DataFrame(geometry);geo.to_csv(OUT/'validation_crop_geometry.csv',index=False)
 savejson(OUT/'duplicate_summary.json',{'hash_radius':4,'pairs':len(pairdf),'validation_images_with_candidate':len(excluded),'remaining_validation_n':int(keep.sum()),'cross_label_pairs':int((~pairdf.same_label).sum()),'distance_counts':{str(k):int(v) for k,v in pairdf.hamming_distance.value_counts().sort_index().items()},'validation_landscape_fraction':float((geo.aspect_ratio>1).mean()),'validation_median_aspect_ratio':float(geo.aspect_ratio.median()),'validation_median_width_fraction_removed':float(geo.width_fraction_removed.median()),'visually_confirmed_duplicates':None})
 print('HASH_COMPLETE',len(pairdf),'pairs;',len(excluded),'validation candidates',flush=True)

def main():
 RAW.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
 mode=sys.argv[1];lock=(RAW/f'{mode}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 {'prepare':prepare,'oracle':oracle,'hash':hashes}[mode]()
if __name__=='__main__':main()
