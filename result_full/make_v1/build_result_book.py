from pathlib import Path
import json,shutil,hashlib,itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'runs_full/make_v1'
OUT=Path(__file__).resolve().parent
names=['flat','hierarchy','hierarchy_make_weighted']; short=['A Flat','B Hierarchy','C Weighted']; colors=['#2866a6','#d28a22','#26836b']
assert json.loads((RAW/'queue_status.json').read_text())['state']=='complete'
assert 'ALL_9_RUNS_COMPLETE' in (RAW/'queue.log').read_text()
rows=[]; make_rows=[]; class_rows=[]; transitions=[]; checked=[]; frames={}; histories={}; cms={}
for seed in range(3):
 d=RAW/f'seed{seed}';cfg=json.loads((d/'config.json').read_text());audit=json.loads((d/'data_audit.json').read_text());ci=pd.read_csv(d/'splits/class_info.csv').sort_values('model_idx');mapping=ci.make_idx.to_numpy();val=pd.read_csv(d/'splits/val.csv');counts=np.asarray(audit['make_train_counts']);minority=np.argsort(counts,kind='stable')[:3]
 for f,h in cfg['split_sha256'].items(): assert hashlib.sha256((d/'splits'/f).read_bytes()).hexdigest()==h
 assert hashlib.sha256((d/'full_train_source.py').read_bytes()).hexdigest()==cfg['source_sha256']
 for name in names:
  src=d/name;m=json.loads((src/'metrics.json').read_text());h=pd.read_csv(src/'history.csv');pr=pd.read_csv(src/'validation_predictions.csv');y=pr.target.to_numpy();p=pr.prediction.to_numpy();assert m['seed']==seed and not m['smoke'] and m['split']=='validation';assert len(y)==17771 and np.array_equal(y,val.model_idx)
  expected={'accuracy':float((y==p).mean()),'macro_f1':f1_score(y,p,average='macro'),'balanced_accuracy':recall_score(y,p,average='macro'),'top3_accuracy':pr[['top1','top2','top3']].eq(pr.target,axis=0).any(axis=1).mean(),'cross_make_error_rate':float((mapping[y]!=mapping[p]).mean()),'within_make_error_rate':float(((y!=p)&(mapping[y]==mapping[p])).mean()),'minority_make_fine_accuracy':float(np.mean([(y[mapping[y]==g]==p[mapping[y]==g]).mean() for g in minority]))}
  for k,v in expected.items(): assert abs(m[k]-v)<1e-9,(seed,name,k)
  assert np.array_equal(np.load(src/'validation_confusion.npy'),confusion_matrix(y,p,labels=np.arange(50)))
  assert abs(h.macro_f1.max()-m['macro_f1'])<1e-9
  best=int(h.loc[h.macro_f1.idxmax(),'epoch']); row=dict(seed=seed,experiment=name,best_epoch=best,epochs_completed=m['epochs_completed'],minutes=float(h.minutes.sum()),**{k:v for k,v in m.items() if isinstance(v,(float,int)) and k not in ['seed','epochs_completed','smoke']});rows.append(row)
  for r in m['per_make']:make_rows.append(dict(seed=seed,experiment=name,**r))
  for r in m['per_class']:class_rows.append(dict(seed=seed,experiment=name,**r))
  dest=OUT/'evidence'/f'seed{seed}'/name;dest.mkdir(parents=True,exist_ok=True)
  for f in ['metrics.json','history.csv','validation_predictions.csv','validation_confusion.npy']:shutil.copy2(src/f,dest/f)
  frames[seed,name]=pr;histories[seed,name]=h;cms[seed,name]=np.load(src/'validation_confusion.npy')
  checked.append(dict(seed=seed,model=name,images=len(y),best_epoch=best,checks='metrics, top3, minority, error decomposition, confusion, source and split hashes PASS'))
 dest=OUT/'evidence'/f'seed{seed}'
 for f in ['config.json','data_audit.json','status.json']:shutil.copy2(d/f,dest/f)
 ci.to_csv(dest/'class_info.csv',index=False)
 val[['rel_path','model_idx','make_idx','make_model']].to_csv(dest/'validation_index.csv',index=False)
 for a,b in itertools.combinations(names,2):
  x=frames[seed,a];z=frames[seed,b];y=x.target.to_numpy();ca=x.prediction.to_numpy()==y;cb=z.prediction.to_numpy()==y
  transitions.append(dict(seed=seed,comparison=b+' minus '+a,both_correct=int((ca&cb).sum()),rescued=int((~ca&cb).sum()),harmed=int((ca&~cb).sum()),both_wrong=int((~ca&~cb).sum())))
df=pd.DataFrame(rows);df.to_csv(OUT/'per_run.csv',index=False);pd.DataFrame(checked).to_csv(OUT/'verification.csv',index=False);pd.DataFrame(transitions).to_csv(OUT/'prediction_transitions.csv',index=False)
keys=['accuracy','macro_f1','balanced_accuracy','top3_accuracy','minority_make_fine_accuracy','cross_make_error_rate','within_make_error_rate']
g=df.groupby('experiment')[keys].agg(['mean','std']);g.columns=['_'.join(c) for c in g.columns];g.to_csv(OUT/'mean_std.csv')
make=pd.DataFrame(make_rows);make.to_csv(OUT/'per_make.csv',index=False);pc=pd.DataFrame(class_rows);pc.to_csv(OUT/'per_class.csv',index=False)
deltas=[]
for a,b in itertools.combinations(names,2):
 for seed in range(3):
  x=df[(df.seed==seed)&(df.experiment==a)].iloc[0];z=df[(df.seed==seed)&(df.experiment==b)].iloc[0]
  deltas.append(dict(seed=seed,comparison=b+' minus '+a,**{k:100*(z[k]-x[k]) for k in keys}))
dd=pd.DataFrame(deltas);dd.to_csv(OUT/'paired_deltas_pp.csv',index=False)
agg=dd.groupby('comparison')[keys].agg(['mean','std']);agg.columns=['_'.join(c) for c in agg.columns];agg.to_csv(OUT/'paired_mean_std_pp.csv')
win=dd.groupby('comparison')[keys].agg(lambda x:int((x>0).sum()));win.to_csv(OUT/'positive_delta_counts.csv')
pairs=[]
for (seed,name),cm in cms.items():
 for i,j in np.argwhere(cm>0):
  if i!=j:pairs.append(dict(seed=seed,experiment=name,true_class=ci.make_model.iloc[i],predicted_class=ci.make_model.iloc[j],count=int(cm[i,j]),same_make=bool(mapping[i]==mapping[j])))
pd.DataFrame(pairs).sort_values('count',ascending=False).to_csv(OUT/'confusion_pairs.csv',index=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
fig,ax=plt.subplots(2,2,figsize=(16,10));fig.suptitle('Vehicle classification | 3 seeds, 9 completed runs | Validation',fontsize=19,fontweight='bold',y=.98)
ax[0,0].axis('off');tablekeys=keys[:5];labels=['Accuracy','Macro-F1','Balanced accuracy','Top-3 accuracy','Minority-make accuracy*'];cells=[]
for k,l in zip(tablekeys,labels):cells.append([l]+[f'{g.loc[n,k+"_mean"]*100:.2f} +/- {g.loc[n,k+"_std"]*100:.2f}' for n in names])
t=ax[0,0].table(cellText=cells,colLabels=['Percent; mean +/- SD']+short,cellLoc='center',colWidths=[.34,.22,.22,.22],bbox=[0,.16,1,.73]);t.auto_set_font_size(False);t.set_fontsize(8)
for j,c in enumerate(colors,1):t[0,j].set_facecolor(c);t[0,j].get_text().set_color('white')
ax[0,0].set_title('Overall metrics and minority subgroup',loc='left');ax[0,0].text(0,0,'* Equal-weight mean of BMW, GMC, Volkswagen fine accuracy.',fontsize=9,transform=ax[0,0].transAxes)
sel=['macro_f1','minority_make_fine_accuracy'];yy=np.arange(2)
for i,n in enumerate(names[1:]):
 sub=dd[dd.comparison==n+' minus flat'];ax[0,1].barh(yy+(i-.5)*.28,sub[sel].mean(),height=.25,color=colors[i+1],label=short[i+1]);
 for j,k in enumerate(sel):ax[0,1].scatter(sub[k],np.full(3,yy[j]+(i-.5)*.28),s=30,color='black',zorder=5)
ax[0,1].axvline(0,color='gray');ax[0,1].set_yticks(yy,['Macro-F1','Minority accuracy']);ax[0,1].set_xlabel('Change vs Flat (percentage points); dots = seeds');ax[0,1].legend(loc='upper left');ax[0,1].set_title('Paired changes: subgroup benefit, no global gain');ax[0,1].set_xlim(-1.3,1.8)
for i,n in enumerate(names):
 hs=pd.concat([histories[s,n].assign(seed=s) for s in range(3)]);z=hs.groupby('epoch').macro_f1.agg(['mean','std']);ax[1,0].plot(z.index,z['mean']*100,color=colors[i],label=short[i]);ax[1,0].fill_between(z.index,(z['mean']-z['std'])*100,(z['mean']+z['std'])*100,color=colors[i],alpha=.13)
ax[1,0].set(xlabel='Epoch',ylabel='Validation Macro-F1 (%)',ylim=(0,100),title='Learning curves; shading = seed SD');ax[1,0].legend()
err=['cross_make_error_rate','within_make_error_rate'];x=np.arange(2)
for i,n in enumerate(names):ax[1,1].bar(x+(i-1)*.24,[g.loc[n,k+'_mean']*100 for k in err],width=.23,color=colors[i],label=short[i])
ax[1,1].set_xticks(x,['Cross-make error','Within-make error']);ax[1,1].set(ylabel='Percent of all validation images',ylim=(0,4),title='Errors shift toward within-make confusions');ax[1,1].legend()
fig.text(.05,.025,'Fixed split: train 82,929 / validation 17,771 / reserved test 17,771. ImageNet-pretrained ResNet18; no CompCars.\nCheckpoint selection uses validation Macro-F1. Three seeds describe training variation; no significance or test-generalization claim.',fontsize=10,color='#475569')
fig.subplots_adjust(left=.05,right=.98,bottom=.14,top=.90,wspace=.35,hspace=.42);fig.savefig(OUT/'overview.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True)
for i,n in enumerate(names):
 for seed in range(3):
  h=histories[seed,n];axes[0,i].plot(h.epoch,h.train_loss,label=f'Train s{seed}');axes[0,i].plot(h.epoch,h.loss,'--',label=f'Val s{seed}');axes[1,i].plot(h.epoch,h.train_accuracy*100);axes[1,i].plot(h.epoch,h.accuracy*100,'--')
 axes[0,i].set_title(short[i]);axes[0,i].legend(fontsize=7);axes[0,i].set_ylabel('Loss');axes[1,i].set(xlabel='Epoch',ylabel='Accuracy (%)',ylim=(0,100))
fig.suptitle('Solid: online train (augmented batches); dashed: validation. C train loss is weighted.',fontsize=13);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(OUT/'learning_curves.png',dpi=160);plt.close(fig)
for f in ['protocol.json','queue_status.json']:shutil.copy2(RAW/f,OUT/f)
source=OUT/'sources';source.mkdir(exist_ok=True)
for f in ['full_train.py','run_full_experiments.py','summarize_full.py']:shutil.copy2(RAW/f,source/f)
summary=[]
for n in names:summary.append('| '+n+' | '+' | '.join(f'{g.loc[n,k+"_mean"]*100:.3f} ± {g.loc[n,k+"_std"]*100:.3f}' for k in keys[:5])+' |')
report='''# 자동차 계층형 분류 본 실험 결과집

기본 CNN이 전체 정확도와 Macro-F1에서 가장 높았다. 제조사 가중치를 적용한 계층형은 소수 제조사 평균 차종 정확도를 높였지만, 전체 성능 향상으로 이어지지 않았다. 아래 수치는 같은 검증 세트에서 세 시드를 반복한 결과이며 최종 테스트 성능은 아니다.

![전체 결과](overview.png)

## 실험 조건과 완료 확인

세 모델 × 시드 0·1·2의 9회 실행을 모두 완료했다. 최대 30 epoch, 검증 Macro-F1이 5회 연속 개선되지 않으면 종료하도록 설정했으며 실제로는 모두 30 epoch를 수행했다. 각 실행의 최고 검증 Macro-F1 체크포인트를 비교한다.

VMMRdb에서 연식을 합친 50개 제조사-모델 클래스를 사용했다. 학습 82,929장, 검증 17,771장, 테스트 17,771장으로 분할했다. 테스트 이미지는 평가하지 않았다. CompCars도 이번 비교에 사용하지 않았다.

세 모델 모두 ImageNet 사전학습 ResNet18 전체를 파인튜닝했다. 입력은 RGB 224×224, 배치 64, AdamW(초기 학습률 0.0003, weight decay 0.0001), cosine 학습률 감소를 사용했다. 학습 증강은 RandomResizedCrop, 수평 반전, ColorJitter이며 검증은 Resize와 CenterCrop이다. 같은 시드의 세 모델은 초기화·학습 순서·증강 난수를 맞췄다.

- A: 차종 50개를 직접 분류한다. 제조사 확률은 차종 확률을 제조사별로 합쳐 산출한다.
- B: 공유 특징에서 제조사 확률과 제조사 내부 차종 확률을 학습하고 두 확률을 곱한다. 모든 제조사 후보를 유지하는 soft 계층형이다. 정답 제조사는 추론에 사용하지 않는다.
- C: B와 같은 구조에 학습 제조사 빈도의 역제곱근 가중치를 적용한다. 가중치는 제조사 손실만이 아니라 최종 차종 NLL의 이미지별 가중치로 작용한다.

## 주요 결과

단위는 %이며 ±는 시드 간 표본 표준편차(ddof=1)다.

| 모델 | 정확도 | Macro-F1 | Balanced accuracy | Top-3 | 소수 제조사 차종 정확도 |
|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(summary)+'''

소수 제조사는 학습 이미지 수가 가장 적은 BMW·GMC·Volkswagen으로 고정했다. 해당 지표는 제조사별 차종 정확도를 동일 비중으로 평균한다. 모든 소수 차종을 포괄하는 지표나 제조사 예측 정확도와는 다르다.

## 가설 판정

B−A의 Macro-F1 차이는 평균 −0.187%p이며 3개 시드 모두 음수였다. C−A는 평균 −0.055%p이며 역시 3개 시드 모두 음수였다. 따라서 제조사 계층과 가중치가 전체 Macro-F1을 높인다는 원래 주장은 이번 조건에서 지지되지 않았다. 당초 목표인 +2%p와 3회 중 2회 개선도 충족하지 못했다. 차이가 작다는 사실만으로 통계적 동등성이 입증된 것은 아니다.

C−B의 Macro-F1은 평균 +0.132%p로 3개 시드 중 2개에서 개선됐다. 가중치가 계층형의 전체 성능을 일부 회복했지만 A를 넘지는 못했다.

C의 소수 제조사 차종 정확도는 A보다 평균 +0.814%p, B보다 +1.420%p 높았다. A 및 B와의 비교에서 소수 제조사 성능은 세 시드 모두 개선됐다. 시드별 차이는 paired_deltas_pp.csv와 positive_delta_counts.csv에 기록했다. 이 결과는 소수 제조사에 대한 목적을 별도로 평가할 필요가 있음을 보여준다.

## 오류가 어떻게 바뀌었는가

A의 제조사 간 차종 오류율은 평균 3.243%, B는 3.198%, C는 3.213%였다. B−A의 감소 폭은 0.045%p에 그쳤고 모든 시드에서 감소한 것은 아니다. 반면 같은 제조사 안에서 틀린 비율은 A 2.536%, B 2.727%, C 2.742%로 증가했다. 오류율의 분모는 오답 수가 아니라 전체 검증 이미지 수다. 따라서 제조사 간 오류의 작은 감소를 전체 우월성으로 확대 해석할 수 없다.

confusion_pairs.csv는 실제 오분류 쌍을 기록하고, prediction_transitions.csv는 같은 이미지에서 개선 모델이 구제한 정답 수와 새로 틀리게 만든 수를 기록한다. 낮은 클래스별 F1만으로 외형 유사성이 원인이라고 단정하지 않는다. 사진 검토와 라벨 검수가 추가로 필요하다.

## 개선이 제한된 이유와 확인해야 할 것

관측된 사실은 전체 성능이 비슷하면서 소수 제조사 성능과 오류 분포가 달라졌다는 점이다. 사전학습된 특징과 차종 학습만으로 제조사 정보가 충분히 표현돼 별도 계층의 추가 정보가 작았을 가능성이 있다. 그러나 사전학습 유무를 통제한 비교를 하지 않았으므로 이 설명은 검증되지 않았다.

제조사 계층은 제조사를 맞혀도 남는 동일 제조사 내 차종 차이를 직접 해결하지 못한다. 다른 제조사의 유사 차종 역시 상위 분류가 불확실할 수 있다. 이 설명은 오류 쌍과 사례 분석으로 더 점검해야 한다. 현재 구현은 hard routing으로 후보를 제거하지 않으므로 첫 제조사 오류가 곧 정답 불가능을 뜻하지 않는다.

가중치는 손실에서 소수 제조사의 영향력을 높인다. 전체 이미지 정확도와 제조사별 균형 성능이 서로 다른 목표이므로 한쪽 개선이 다른 쪽 개선을 보장하지 않는다. 관측된 변화가 실제로 어떤 제조사 사이의 교환에서 생겼는지는 per_make.csv와 이미지별 예측으로 확인할 수 있다.

## 적용 조건과 후속 실험

상위 범주가 안정적으로 구분되고 세부 분류에 유용한 제약을 제공하는 경우 계층형이 도움이 될 가능성이 있다. 소수 제조사에 대한 오류 비용이 큰 사용 목적에서는 C의 집단 성능 개선을 별도로 평가할 수 있다. 배포 우월성을 판단하려면 해당 오류 비용을 먼저 정해야 한다.

후속 검증은 무작위 초기화/사전학습 × 직접분류/계층형을 비교해 사전학습과 계층의 상호작용을 확인하는 것부터 시작한다. 무작위 초기화에도 충분한 학습 예산을 줘야 한다. 차종 빈도 가중치, 고해상도 입력, 다른 상위 그룹은 각각 하나씩 바꿔 비교한다. 혼동 기반 그룹은 학습 내부 검증 또는 out-of-fold 예측에서 만들고 평가용 데이터를 재사용하지 않는다. 이 수정들이 성공을 보장하지는 않는다.

## 해석 범위와 검증

동일 분할의 세 시드는 학습 변동성을 보여주며 분할 변동성이나 통계적 유의성을 입증하지 않는다. 모든 체크포인트는 검증 Macro-F1으로 선택했으므로 선택에 따른 낙관성이 있을 수 있다. 최종 일반화 주장은 별도 테스트 평가 이후에 해야 한다. 예비 결과를 본 뒤 정한 하위 집단 해석은 탐색적 분석으로 구분한다. 이번 실험은 Task 2의 무작위 초기화·무증강 기준 모델과 별개다.

원본 분할 경로·MD5의 교차 중복 검사와 학습 이미지 일부의 실제 해시 확인을 수행했지만 근접 중복은 감사하지 않았다. build_result_book.py는 9회 전부에 대해 원본 예측에서 정확도·Macro-F1·Balanced accuracy·Top-3·소수 제조사 지표·오류 분해를 다시 계산하고 저장 지표와 일치함을 확인했다. 혼동 행렬, 최고 epoch, 소스 및 분할 SHA256도 점검했다. verification.csv에 실행별 통과 기록이 있다.

![학습 곡선](learning_curves.png)

학습 정확도는 증강 배치에서 파라미터가 갱신되는 동안 집계한 값이다. C의 학습 손실은 제조사 가중 손실이며 검증 손실은 비가중 NLL이므로 두 손실의 절댓값을 같은 목적 함수처럼 해석하지 않는다.

## 파일과 재현

- per_run.csv, mean_std.csv: 9회 개별 결과와 평균·표준편차. CSV의 성능 값은 0~1 비율이다.
- paired_deltas_pp.csv, paired_mean_std_pp.csv: 같은 시드의 차이와 집계. 단위는 %p다.
- positive_delta_counts.csv: 양수 차이가 나온 시드 수. 오류율의 양수는 악화를 뜻한다.
- per_make.csv, per_class.csv, confusion_pairs.csv, prediction_transitions.csv: 집단·차종·오류 분석 자료.
- evidence/: 각 실행의 지표, 학습 이력, 예측, 혼동 행렬과 검증 인덱스. 이미지와 모델 가중치는 포함하지 않는다.
- sources/: 실제 실행한 학습·큐·집계 소스 스냅샷. 원본 실행에는 프로젝트 루트의 같은 이름 파일과 runs_pilot/pilot_seed0_v1/splits의 분할 스냅샷이 필요하다. 소스를 루트에 배치하고 환경·데이터를 준비한 뒤 run_full_experiments.py로 실행한다. 이 결과집만으로 원본 데이터 없이 재학습할 수 있는 것은 아니다.
- build_result_book.py: 프로젝트 루트에서 python result_full/make_v1/build_result_book.py를 실행하면 서버의 runs_full/make_v1에서 결과집을 다시 만든다. 재학습은 수행하지 않는다.

GitHub에는 본 실험 결과와 소스 증빙만 공유한다. 원본 이미지·체크포인트·smoke·맛보기 예비 결과는 포함하지 않는다.
'''
(OUT/'README.md').write_text(report)
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.json'}
(OUT/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2))
print('BOOK_VERIFIED',len(rows),'runs',len(manifest),'files',round(sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())/1e6,2),'MB')
print(g.round(6).to_string());print(win.to_string())
