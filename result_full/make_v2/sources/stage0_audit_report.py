"""Aggregate Stage 0 visual screening; no training or test access."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'runs_full/make_v2/stage0'; OUT=ROOT/'result_full/make_v2/stage0'
def main():
 a=pd.read_csv(RAW/'visual_audit_template.csv').fillna('')
 notes=pd.DataFrame(json.loads((RAW/'visual_notes_partial.json').read_text())) if (RAW/'visual_notes_partial.json').exists() else pd.read_csv(OUT/'visual_audit.csv')[['audit_id','primary_category','crop_cuts_visible_cue','confidence','reviewed_by']]
 assert len(notes)==421 and notes.audit_id.nunique()==421 and set(notes.audit_id)==set(range(421))
 for col in notes.columns:
  if col!='audit_id':a[col]=a.audit_id.astype(int).map(notes.set_index('audit_id')[col])
 details={101:'Mustang label; visible red pickup (Ranger-like), not a Mustang body.',102:'Mustang label; visible SUV.',103:'Mustang label; visible Explorer-like SUV.',104:'Mustang label; visible sedan.',105:'Mustang label; Honda-like sedan visible.',106:'Mustang label; visible SUV.',123:'Silverado label; GMC grille lettering visible.',124:'Silverado label; GMC grille lettering visible.',135:'Silverado label; Toyota-like sedan visible.',136:'Silverado label; Dodge-like sedan visible.',181:'Corolla label; Accord-like sedan visible.',187:'Very small source image; inspect native dimensions.',190:'Tacoma label; Honda-like sedan visible.',191:'Tacoma label; minivan visible.',226:'Sentra label; BMW-like sedan visible.',305:'Cherokee label; Grand Cherokee lettering/body suggests taxonomy mismatch.',306:'Cherokee label; Grand Cherokee lettering/body suggests taxonomy mismatch.',337:'BMW 325i label; Mustang-like convertible visible.',344:'High rear/interior view with open roof.',349:'BMW label; non-BMW-like sedan body/grille.',350:'BMW label; non-BMW-like sedan body.',357:'328i label; 528i rear badge appears visible.',366:'BMW 328i label; Infiniti FX35 badge/body visible.',370:'Wide promotional side image; center crop isolates open doors and removes both ends.',379:'GMC label; Chevrolet bowtie on grille.',419:'GMC Jimmy label; Chevrolet bowtie on grille.',420:'Side view; center crop isolates doors and removes vehicle ends.'}
 a['notes']=[details.get(int(i),'Visual contact-sheet screening; category is a review candidate, not expert-confirmed ground truth.') for i in a.audit_id]
 a['review_status']='screened_1_pass'
 a['cue_type']=''
 a['secondary_categories']=''
 a.to_csv(RAW/'visual_audit_full.csv',index=False)
 a.drop(columns='rel_path').to_csv(OUT/'visual_audit.csv',index=False)
 cats=json.loads((OUT/'protocol.json').read_text())['audit_categories']
 counts=pd.crosstab(a.primary_category,a.group).reindex(cats,fill_value=0)
 counts.to_csv(OUT/'visual_category_counts.csv')
 crops=pd.crosstab(a.crop_cuts_visible_cue,a.group).reindex(['yes','no','uncertain'],fill_value=0)
 crops.to_csv(OUT/'visual_crop_counts.csv')
 # All cross-label candidates plus four same-label pairs per distance, deterministic seed.
 d=pd.read_csv(OUT/'near_duplicate_candidates.csv').reset_index(names='pair_id')
 selected=[d[~d.same_label]]
 for dist,g in d[d.same_label].groupby('hamming_distance'):
  selected.append(g.sample(min(4,len(g)),random_state=20261022+int(dist)))
 sample=pd.concat(selected).sort_values(['same_label','hamming_distance','pair_id']).reset_index(drop=True)
 sample['sample_id']=np.arange(len(sample));sample['visual_verdict']='';sample['visual_notes']=''
 prior=OUT/'duplicate_visual_sample.csv'
 if prior.exists():
  old=pd.read_csv(prior).set_index('pair_id')
  for col in ['visual_verdict','visual_notes']:sample[col]=sample.pair_id.map(old[col]).fillna('')
 sample.to_csv(prior,index=False)
 tr=pd.read_csv(ROOT/'splits/train.csv');va=pd.read_csv(ROOT/'splits/val.csv')
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
 (RAW/'duplicate_sheets').mkdir(exist_ok=True)
 for page,start in enumerate(range(0,len(sample),8),1):
  sheet=Image.new('RGB',(1600,1120),'white');draw=ImageDraw.Draw(sheet)
  draw.text((8,5),'dHash candidate sample: LEFT validation / RIGHT training (not test)',fill='black',font=font)
  for slot,j in enumerate(range(start,min(start+8,len(sample)))):
   row=sample.iloc[j];x=slot%2*800;y=35+slot//2*270
   vr=va.iloc[int(row.val_index)];tt=tr.iloc[int(row.train_index)]
   draw.text((x+5,y),f'#{j} pair={row.pair_id} d={row.hamming_distance} same_label={row.same_label}',fill='black',font=font)
   for k,r in enumerate([vr,tt]):
    draw.text((x+k*400+5,y+22),str(r.make_model),fill='black',font=font)
    with Image.open(ROOT/'dataset/VMMRdb'/r.rel_path) as src: im=ImageOps.contain(src.convert('RGB'),(390,215))
    sheet.paste(im,(x+k*400,y+45))
  sheet.save(RAW/f'duplicate_sheets/page_{page:02d}.png')
 # Representative paired original/crop panels stay excluded from Git.
 ids=[101,123,187,305,337,366,370,420]
 fig=Image.new('RGB',(1500,1000),'white');draw=ImageDraw.Draw(fig)
 from experiments.make_v2.stage0_diagnostics import crop
 for slot,j in enumerate(ids):
  row=a[a.audit_id==j].iloc[0];x=slot%2*750;y=slot//2*250
  draw.text((x+5,y+3),f'#{j} {row.make_model}: original / center crop',fill='black',font=font)
  with Image.open(ROOT/'dataset/VMMRdb'/row.rel_path) as src: im=src.convert('RGB')
  cc,box,_,_=crop(im);marked=im.copy();ImageDraw.Draw(marked).rectangle(box,outline='red',width=max(2,im.width//140))
  fig.paste(ImageOps.contain(marked,(440,210)),(x,y+35));fig.paste(ImageOps.contain(cc,(210,210)),(x+450,y+35))
 fig.save(RAW/'representative_cases.png')
 protocol=json.loads((OUT/'protocol.json').read_text())
 for n,h in protocol['split_sha256'].items():assert hashlib.sha256((ROOT/'splits'/n).read_bytes()).hexdigest()==h
 print('AUDIT_COUNTS\n'+counts.to_string());print('CROP_COUNTS\n'+crops.to_string());print('DUPLICATE_SAMPLE',len(sample))
 print('SPLITS_UNCHANGED; TEST_NOT_READ')
if __name__=='__main__':main()
