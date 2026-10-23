"""Publish final test tables and project closure docs from saved predictions only."""
from pathlib import Path
import json,hashlib,re,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.make_v2.final_evaluate import KEYS,recompute
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'result_full/make_v2';TEST=BASE/'test'
NAMES=['flat','hierarchy','hierarchy_make_weighted','flat_make_weighted']
LABELS={'flat':'A Flat','hierarchy':'B 제조사 계층 (soft)','hierarchy_make_weighted':'C 계층+제조사 가중치','flat_make_weighted':'D Flat+제조사 가중치'}
def table(df,metrics):
 out='| 모델 | '+' | '.join(label for key,label in metrics)+' |\n|---|'+'---:|'*len(metrics)+'\n'
 for name in NAMES:
  g=df[df.experiment==name]
  out+='| '+LABELS[name]+' | '+' | '.join(f'{100*g[key].mean():.3f} ± {100*g[key].std(ddof=1):.3f}' for key,label in metrics)+' |\n'
 return out

def main():
 done=json.loads((TEST/'completion.json').read_text());assert done['state']=='complete' and done['evaluated_runs']==12
 frozen=json.loads((TEST/'frozen_protocol.json').read_text());cfg=frozen['config']
 for f,h in cfg['source_sha256'].items():
  actual=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
  if actual!=h:
   a=json.loads((TEST/'evaluation_code_amendment.json').read_text())
   assert f=='experiments/make_v2/final_evaluate.py' and a['old_sha256']==h and a['new_sha256']==actual
 val=pd.read_csv(BASE/'per_run.csv');test=pd.read_csv(TEST/'per_run.csv');ci=pd.read_csv(TEST/'class_info.csv');idx=pd.read_csv(TEST/'test_index.csv')
 assert len(test)==12 and len(pd.read_csv(TEST/'verification.csv'))==12
 minority=ci[ci.make.isin(cfg['minority_makes'])].make_idx.unique()
 for row in test.itertuples():
  pred=pd.read_csv(TEST/'evidence'/f'seed{row.seed}'/row.experiment/'test_predictions.csv')
  assert np.array_equal(pred.target,idx.model_idx)
  for k,v in recompute(pred,ci,minority).items():assert abs(getattr(row,k)-v)<1e-9
 metrics=[('accuracy','정확도 (%)'),('macro_f1','Macro-F1 (%)'),('minority_make_fine_accuracy','소수 제조사 평균 정확도 (%)')]
 report='# 최종 실험 결과 및 종료 보고\n\n고정 분할: train 82,929 / validation 17,771 / test 17,771. 4개 모델 × seed 0·1·2. 추가 학습 없이 검증 최고 체크포인트를 고정한 뒤 최종 테스트를 완료했다.\n\n'
 report+='**최종 결론:** 전체 정확도와 Macro-F1은 A Flat이 가장 높았다. 제조사 계층의 주 지표 우위는 확인되지 않았다. 가중치는 테스트 소수 제조사 평균 정확도를 Flat에서 +1.068%p, 계층형에서 +0.942%p 개선했고 모두 3/3이었다. 전체 성능 이득과 집단별 이득을 구분한다. 추가 학습 없이 실험을 종료했다.\n\n'
 report+='## 실험 범위와 선택 고정\n\nA/B/C의 1차 실행 9회를 재사용하고, 2차에서 D 3회와 letterbox 선별 1회를 새로 학습했다. 최종 표의 12회는 모두 기존 중앙 크롭 조건이다. 각 실행의 체크포인트는 검증 Macro-F1로 선택했다. 최종 모델 종류는 테스트 전 A Flat으로 고정했으며, 대표 체크포인트는 A 안에서 검증 Macro-F1이 가장 높은 seed '+str(cfg['representative_seed'])+'로 고정했다. 테스트 결과로 재선정·튜닝·추가 학습하지 않았다.\n\n'
 report+='[고정 프로토콜](../result_full/make_v2/test/frozen_protocol.json)에 12개 체크포인트 SHA-256, 전처리, 분할, 코드 해시와 고정 시각을 기록했다. 각 체크포인트를 테스트에서 한 번씩 추론했고, 이후 검산과 표 생성에는 저장된 예측만 사용했다.\n\n'
 report+='## 검증 성능\n\n'+table(val,metrics)+'\n## 최종 테스트 성능\n\n'+table(test,metrics)
 report+='\n모든 ±는 seed 3개의 표본 표준편차다. 소수 제조사 지표는 학습 빈도가 낮은 BMW·GMC·Volkswagen 각각의 차종 정확도를 구한 뒤 세 값을 동등하게 평균한 값이다. 세 제조사 이미지를 합친 정확도가 아니다.\n\n'
 rep=test[(test.experiment=='flat')&(test.seed==cfg['representative_seed'])].iloc[0]
 report+=f'테스트 전에 선택한 대표 A seed {cfg["representative_seed"]}의 정확도는 **{100*rep.accuracy:.3f}%**, Macro-F1은 **{100*rep.macro_f1:.3f}%**다. 이는 단일 체크포인트 성능이며 위의 3-seed 평균과 구분한다.\n\n'
 report+='## 가설·대조 비교\n\n주 지표는 Macro-F1이다. 사전 고정한 운영상 효과 기준은 같은 seed 차이 3/3 양수와 평균 +0.2%p 초과이며 통계적 유의성 검정은 아니다. 오류율은 낮을수록 좋고, 분모는 각 평가 분할의 이미지 전체다.\n\n'
 report+='| 비교 | 검증 Δ Macro-F1 (%p) | 테스트 Δ Macro-F1 (%p) | 테스트 paired SD (%p) | 테스트 개선 seed | 운영 기준 |\n|---|---:|---:|---:|---:|---|\n'
 vd=pd.read_csv(BASE/'hypothesis_decisions.csv').set_index('comparison');td=pd.read_csv(TEST/'hypothesis_decisions.csv')
 for r in td.itertuples():
  report+=f'| {r.comparison} | {vd.loc[r.comparison,"mean_macro_f1_delta_pp"]:+.3f} | {r.mean_macro_f1_delta_pp:+.3f} | {r.paired_sd_pp:.3f} | {r.improved_seeds}/3 | '+('충족' if r.operational_effect else '미충족')+' |\n'
 wins=pd.read_csv(TEST/'positive_delta_counts.csv')
 h1=td[td.comparison=='B-A hierarchy'].iloc[0];h1w=td[td.comparison=='C-D hierarchy with weights'].iloc[0]
 report+=f'\n**H1 판정:** 검증 B−A는 −0.187%p(개선 0/3)였다. 테스트 B−A는 {h1.mean_macro_f1_delta_pp:+.3f}%p(개선 {int(h1.improved_seeds)}/3), C−D는 {h1w.mean_macro_f1_delta_pp:+.3f}%p(개선 {int(h1w.improved_seeds)}/3)다. 테스트 B−A는 운영상 효과 기준을 '+('충족했다' if h1.operational_effect else '충족하지 못했다')+'. 검증과 테스트의 방향·크기를 함께 보고하며, 모든 계층 구조의 우열로 일반화하지 않는다.\n\n'
 report+='| 비교 | 테스트 소수 제조사 Δ (%p) | 개선 seed | 테스트 같은 제조사 오류율 Δ (%p) | 오류 감소 seed |\n|---|---:|---:|---:|---:|\n'
 for comp in ['D-A weight in flat','C-B weight in hierarchy','C-D hierarchy with weights','B-A hierarchy']:
  a=wins[(wins.comparison==comp)&(wins.metric=='minority_make_fine_accuracy')].iloc[0];b=wins[(wins.comparison==comp)&(wins.metric=='within_make_error_rate')].iloc[0]
  report+=f'| {comp} | {a.mean_delta_pp:+.3f} | {int(a.improved_seeds)}/3 | {b.mean_delta_pp:+.3f} | {int(b.improved_seeds)}/3 |\n'
 report+='\n가중치의 소수 제조사 이득과 전체 Macro-F1의 변화는 별개다. 평균이 올라도 seed 방향이 일관되지 않으면 일관된 개선이라고 쓰지 않는다.\n\n'
 report+='## 전처리·데이터 진단·캐시\n\nLetterbox는 seed 0 정확도와 같은 제조사 오류율에서 일부 이득이 있었으나 Macro-F1이 92.115% → 92.015%(−0.099%p)로 낮아 미채택했다. 단일 seed 탐색 결과로 차량 단서 보존 가설 전체를 기각하지 않는다. 학습 증강과 평가 기하가 함께 바뀌어 중앙 크롭만의 인과 효과도 분리하지 못했다. 형제차 오류의 방향별 수치는 [선별 표](../result_full/make_v2/selection_pair_errors.csv)에 보관했다.\n\n'
 report+='공통 오답 321장과 정답 대조군 100장을 검토했다. 라벨 의심과 구별 어려움은 단일 검토자의 잠정 분류이며 확정 원인이 아니다. 단서 잘림은 오답 246/321(76.6%), 정답 78/100(78.0%)여서 이 선택 표본에서는 오답에 더 집중되지 않았다. 학습↔검증 dHash 근접 중복 후보 670쌍(검증 548장)을 확인했으나 모두 실제 중복이라고 확정하지 않았다. 후보 제외 민감도 분석은 원 분할을 수정한 실험이 아니다. 테스트와 학습 사이 근접 중복은 추가 조사하지 않았다.\n\n'
 report+='원본 uint8 캐시는 입력 동일성을 통과했지만 cache/JPEG 처리량이 0.711·0.711·0.643으로 느려 미채택했다. 이번 보존형 구현에 대한 결과이며 모든 텐서 캐시 방식에 일반화하지 않는다.\n\n'
 report+='## 일반화·과적합 해석\n\n| 모델 | 테스트−검증 정확도 (%p) | 테스트−검증 Macro-F1 (%p) |\n|---|---:|---:|\n'
 for name in NAMES:
  a=val[val.experiment==name];b=test[test.experiment==name]
  report+=f'| {LABELS[name]} | {100*(b.accuracy.mean()-a.accuracy.mean()):+.3f} | {100*(b.macro_f1.mean()-a.macro_f1.mean()):+.3f} |\n'
 report+='\n이 차이는 고정된 테스트 분할에서의 확인값이며 분할 간 변동이나 외부 도메인 일반화를 입증하지 않는다. 학습 중 증강 정확도와 평가 정확도는 측정 조건이 다르고, 가중 학습 손실과 검증 NLL도 직접 같게 비교할 수 없다. 과적합이 완전히 해결됐다고 단정하지 않는다. 사전학습·증강·학습 길이가 함께 달라 Task 2 대비 개선을 사전학습 하나의 효과로 설명하지 않는다.\n\n'
 report+='평가 중 첫 실행 저장 후 argmax와 topk 첫 항목이 항상 같아야 한다는 검사에서 중단됐고, 동점 순위 처리에 부적합한 검사임을 확인해 제거했다. 첫 실행 예측은 해시를 보존하고 재추론하지 않았다. 모델·전처리·지표 정의는 바뀌지 않았다. 변경 이력은 evaluation_code_amendment.json과 수정 전 소스에 남겼다.\n\n'
 report+='## 종료 상태\n\n| 항목 | 상태 |\n|---|---|\n| 고정 분할·1차 A/B/C 3 seed·2차 D 3 seed | 완료 |\n| 0단계 진단·캐시 측정·letterbox 선별 | 완료 |\n| 최종 체크포인트 고정·12개 테스트 추론·저장 예측 검산 | 완료 |\n| H2: 차체 유형/혼동 기반 계층 비교 | 미실행으로 종료, 지지/기각 불가 |\n| 무작위 초기화 2×2 통제·고해상도·다른 손실·Grad-CAM | 미실행으로 종료 |\n| 03 노트북 val 혼동 기반 그룹 생성 코드 | 이번 실행에 미사용. 향후 재사용 시 내부 검증/OOF로 수정 필요 |\n| 발표 슬라이드·실시간 시연 앱 | 이번 실험 종료 작업의 산출물이 아님, 완료로 표시하지 않음 |\n\n추가 학습·추가 전처리 탐색·테스트 재선택 없이 실험을 종료한다. 제조사 계층의 결과와 미실행 H2를 구분한다.\n\n'
 report+='## 그림·근거\n\n![검증과 테스트 성능](../result_full/make_v2/test/validation_test_overview.png)\n\n![seed별 테스트 비교](../result_full/make_v2/test/paired_macro_f1.png)\n\n[테스트 결과집](../result_full/make_v2/test/README.md) · [검증 결과집](../result_full/make_v2/README.md) · [2차 검증 해설](make_v2_results.md) · [프로젝트 폴더 안내](../README.md)\n'
 (ROOT/'reports/final_results.md').write_text(report)
 # Scientific figures derived solely from saved metrics.
 fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');x=np.arange(4)
 for ax,key in zip(axes,['accuracy','macro_f1','minority_make_fine_accuracy']):
  for offset,frame,label,color in [(-.18,val,'Validation','#4c78a8'),(.18,test,'Test','#f58518')]:
   g=frame.groupby('experiment')[key];ax.bar(x+offset,[100*g.mean()[n] for n in NAMES],.36,yerr=[100*g.std()[n] for n in NAMES],label=label,color=color,capsize=3)
  ax.set_xticks(x,['A Flat','B Hier','C Hier+W','D Flat+W'],rotation=20);ax.set_ylabel('%');ax.set_title(key);ax.legend(fontsize=8)
 fig.savefig(TEST/'validation_test_overview.png',dpi=160);plt.close(fig)
 dd=pd.read_csv(TEST/'paired_deltas_pp.csv');fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
 for i,(comp,g) in enumerate(dd.groupby('comparison')):
  ax.scatter([i]*3,g.macro_f1,label=comp);ax.plot([i-.15,i+.15],[g.macro_f1.mean()]*2,color='black')
 ax.axhline(0,color='gray');ax.axhline(.2,color='gray',linestyle='--');ax.set_xticks(range(4),[c for c,g in dd.groupby('comparison')],rotation=12,fontsize=8);ax.set_ylabel('Test Macro-F1 difference (percentage points)');fig.savefig(TEST/'paired_macro_f1.png',dpi=160);plt.close(fig)
 (TEST/'README.md').write_text('# 最終 테스트 결과집\n\n[한국어 최종 보고](../../../../reports/final_results.md) · [검증 결과집](../README.md)\n\n검증으로 고정한 12개 체크포인트를 각 1회 테스트했다. 테스트 후 모델 재선정·튜닝·학습은 없었다. frozen_protocol.json이 평가 전 선택과 SHA-256을 기록한다.\n\n'+table(test,metrics)+'\n지표 값은 per_run.csv에서 0~1, paired 표에서 %p다. 소수 제조사는 BMW·GMC·Volkswagen 정확도의 단순 평균이다.\n\n| 파일 | 내용 |\n|---|---|\n| frozen_protocol.json | 평가 전에 고정한 체크포인트·설정·해시 |\n| per_run.csv / mean_std.csv | 실행별 성능 / 평균·표본 표준편차 |\n| paired_deltas_pp.csv / paired_mean_std_pp.csv | 같은 seed끼리 차이 / 평균·편차 |\n| positive_delta_counts.csv / hypothesis_decisions.csv | 개선 seed 수 / 사전 운영 기준 판정 |\n| per_class.csv / per_make.csv / confusion_pairs.csv | 클래스·제조사·혼동 쌍 분석 |\n| validation_test_comparison.csv | 검증과 테스트 차이 |\n| prediction_transitions.csv | 두 모델 사이 새 정답·새 오답 |\n| evidence/ | 체크포인트 없는 저장 예측·혼동 행렬·지표 |\n| test_index.csv / class_info.csv | 평가 행 순서와 클래스 매핑 |\n| verification.csv / data_audit.json | 검산·분할 점검 |\n| completion.json / SHA256SUMS.json | 완료 상태 / 파일 무결성 |\n| sources/ | 최종 평가·집계 소스 사본 |\n\n동일 분할 seed 3회다. 통계적 유의성이나 외부 데이터 일반화를 주장하지 않는다. 테스트 근접 중복은 점검하지 않았다. 원본 사진·체크포인트는 Git에 포함하지 않는다.\n')
 # Correct relative links from test/ (three levels below repository root).
 q=TEST/'README.md';q.write_text(q.read_text().replace('最終','최종').replace('../../../../reports/','../../../reports/'))
 sources=TEST/'sources';sources.mkdir(exist_ok=True)
 for f in ['full_train.py','experiments/make_v2/final_evaluate.py','experiments/make_v2/finalize_project.py']:shutil.copy2(ROOT/f,sources/Path(f).name)
 # Preserve validation report as a dated record while correcting metric definition.
 old=ROOT/'reports/make_v2_results.md';s=old.read_text();s=s.replace('BMW·GMC·VW에 속하는 검증 이미지 전체의 차종 정확도다.','BMW·GMC·VW 각각의 차종 정확도를 동등하게 평균한 값이다(제조사 3개 단순 평균).')
 if '최종 테스트를 포함한 종료 보고' not in s:s=s.replace('\n','\n\n> 이 문서는 테스트 전 검증 분석 기록이다. [최종 테스트를 포함한 종료 보고](final_results.md)를 우선 읽는다.\n',1)
 old.write_text(s)
 (ROOT/'reports/README.md').write_text('# 결과 문서 읽는 순서\n\n1. [최종 실험 종료 보고](final_results.md): 검증·테스트 성능, 가설 판정, 한계와 종료 상태.\n2. [2차 검증 해설](make_v2_results.md): 전처리 선별·데이터 진단·가중치 비교의 테스트 전 기록.\n3. [테스트 원자료](../result_full/make_v2/test/README.md)와 [검증 원자료](../result_full/make_v2/README.md).\n4. [1차 분석 노트북](../04_results_analysis.ipynb): 당시 분석과 가능한 설명.\n')
 # Replace obsolete root checklist with a current closure summary and folder guide.
 root=ROOT/'README.md';previous=root.read_text();end='결과집의 evidence/' if '결과집의 evidence/' in previous else 'result_full은';folder=previous[previous.index('| 경로 | 용도'):previous.index(end)].rstrip()
 folder=folder.replace('| [reports/](reports/README.md) | 팀원이 읽는 결과 해설. 2차 결과·가설 판정·한계·핵심 그림 | 문서 |','| [reports/](reports/README.md) | 최종 종료 보고·검증 해설·그림 | 문서 |')
 folder=folder.replace('최신 검증 결과집. A/B/C 기존 9회 + D 신규 3회, 전처리 선별·진단 포함','검증 결과집. A/B/C 기존 9회 + D 신규 3회, 전처리 선별·진단 포함')
 if '| [result_full/make_v2/test/]' not in folder:folder=folder.replace('| [result_full/make_v2/stage0/]', '| [result_full/make_v2/test/](result_full/make_v2/test/README.md) | 고정한 12개 체크포인트의 최종 테스트 결과·선택 고정 기록 | CSV·JSON·집계 그림·소스 |\n| [result_full/make_v2/stage0/]')
 refs=previous[previous.index('## 참고문헌'):]
 readme='# 자동차 세부 차종 분류 (Intro to AI · Project 1)\n\n## 최종 실험 종료\n\n추가 학습·설정 탐색을 종료하고, 검증으로 고정한 A/B/C/D × seed 0·1·2의 최종 테스트를 완료했다. **검증과 테스트를 구분해 보고하며, 테스트를 보고 모델을 다시 고르지 않았다.** 최종 모델은 테스트 전에 선정한 A Flat이다.\n\n[최종 결과 보고](reports/final_results.md) → [테스트 원자료](result_full/make_v2/test/README.md) → [검증 결과집](result_full/make_v2/README.md) 순서로 읽는다.\n\n## 최종 테스트 성능\n\n'+table(test,metrics)+'\n3 seed 평균 ± 표본 표준편차. 소수 제조사는 BMW·GMC·Volkswagen의 차종 정확도를 제조사별 동등하게 평균했다.\n\n## 완료·미실행 구분\n\n| 항목 | 최종 상태 |\n|---|---|\n| 데이터 선정·중복 제외·고정 70/15/15 분할 | 완료, 분할과 라벨 유지 |\n| Task 2 기준 CNN·1차 A/B/C 3 seed 비교 | 완료 |\n| 공통 오답 321장+대조군 100장·정답 제조사 진단·근접 중복 후보 분석 | 완료 |\n| uint8 캐시·letterbox 선별 | 완료, 각각 속도/주 지표 이득 없어 미채택 |\n| 2차 D Flat+가중치 3 seed·네 모델 비교 | 완료 |\n| 검증 선택 고정·12개 최종 테스트·예측 재검산 | 완료 |\n| H2 차체 유형·혼동 기반 계층 | 미실행으로 종료. 지지/기각 불가 |\n| 무작위 초기화 통제·고해상도·다른 불균형 기법·Grad-CAM | 미실행으로 종료 |\n| 최종 보고서용 결과 정리 | reports/final_results.md 완료 |\n| 발표 슬라이드·실시간 시연 앱 | 별도 제출 준비 항목. 여기서 완료로 주장하지 않음 |\n\n옛 “앞으로 할 일” 체크리스트는 이 종료표로 대체했다. 초기 계획과 변경 이력은 Git 기록 및 PROJECT_FLOW.md에 남아 있다. **미실행 항목을 추가 필수 실험이나 완료 항목으로 읽지 않는다.**\n\n## 실제 수행한 실험\n\nA Flat / B 제조사 계층(soft) / C 계층+제조사 가중치 / D Flat+제조사 가중치. 공통 ImageNet 사전학습 ResNet-18, 224px, AdamW 3e-4, wd 1e-4, cosine, batch 64, 최대 30 epoch, 검증 Macro-F1 최고 체크포인트. 가중치는 학습 제조사 빈도의 역제곱근이며 이미지 평균 가중치 1로 정규화했다.\n\nB−A와 C−D는 계층 효과, D−A와 C−B는 가중치 효과다. A/B/C 9회는 1차 결과를 재사용하고 D 3회만 추가 학습했다. Letterbox seed 0 선별 1회는 최종 12회와 별도이며, 미채택 후 기존 전처리를 유지했다. 무작위 초기화 통제나 H2 비교를 수행한 것처럼 쓰지 않는다.\n\n## 폴더 지도\n\n'+folder+'\n\nresult_full은 공유할 결과집, runs_full은 서버의 원본 실행 자료다. evidence/는 이미지 대신 예측·지표를, sources/는 실행 당시 코드를 보관한다. 체크포인트·원본 이미지·캐시·smoke 결과는 Git 제외다.\n\n## 재현과 코드 사용 시 주의\n\n패키지는 requirements.txt와 각 실행의 환경 기록을 참고한다. V2 진입점은 experiments.make_v2.run_workflow이며 완료 상태 확인을 위해 다시 실행하지 않는다. 최종 평가 코드는 experiments.make_v2.final_evaluate, 저장 결과로 문서 생성은 experiments.make_v2.finalize_project다. 이미 완료한 테스트는 저장 예측을 재사용한다.\n\n02 노트북의 분할 생성은 기존 splits/를 덮어쓰지 않도록 주의한다. 03 노트북은 초기 연구 도구이며 최종 실행과 다른 hard routing·Cui β 등의 기능을 포함한다. 03의 val 혼동 기반 그룹 생성은 최종 A/B/C/D에서 사용하지 않았다. 향후 재사용할 때는 학습 내부 검증/OOF로 바꿔야 하며, 이 미수정 기능을 이번 결과의 검증 근거로 쓰지 않는다. 과거 캐시 속도 권고나 12 epoch 계획은 이번 서버 측정·30 epoch 실행을 설명하지 않는다.\n\n'+refs
 root.write_text(readme)
 flow=ROOT/'PROJECT_FLOW.md';s=flow.read_text();a=s.index('## 2차 실험') if '## 2차 실험' in s else s.index('## 최종 실험 종료') if '## 2차 실험' in s else s.index('## 최종 실험 종료');b=s.index('\n---',a);s=s[:a]+'## 최종 실험 종료\n\n[최종 보고](reports/final_results.md)와 [최종 테스트 결과](result_full/make_v2/test/README.md)를 우선 읽는다. 검증 선택 고정 후 12개 체크포인트의 테스트 평가와 저장 예측 검산을 완료했다. 테스트 후 재선정·추가 학습은 하지 않았다. Stage 3·H2·고해상도 등은 미실행으로 종료한다. 아래 본문은 1차 실험 시점의 기록이며 할 일을 현재 필수 작업으로 해석하지 않는다.\n'+s[b:];flow.write_text(s)
 q=BASE/'README.md';s=q.read_text();s=s if '> 검증 결과 기록이다.' in s else s.replace('\n','\n\n> 검증 결과 기록이다. 최종 테스트와 종료 판정은 [최종 보고](../../reports/final_results.md) 및 [테스트 결과집](test/README.md)을 본다. 아래 “test untouched” 문구는 테스트 이전 기록을 설명한다.\n',1);q.write_text(s)
 (BASE/'RUNNING.md').write_text('# 실험 종료\n\n학습·진단·최종 테스트 평가가 완료됐다. [최종 보고](../../reports/final_results.md)와 [테스트 결과집](test/README.md)을 본다. 최종 평가 로그는 서버 runs_full/make_v2/final_test/evaluation.log, 상태는 status.json에 있다. 추가 학습·탐색은 종료했다.\n')
 q=ROOT/'experiments/make_v2/README.md';s=q.read_text();s+= '' if '## 최종 평가와 종료' in s else '\n## 최종 평가와 종료\n\nfinal_evaluate.py는 검증으로 고정한 체크포인트의 테스트 추론·검산을, finalize_project.py는 저장된 예측을 이용한 최종 문서·집계를 담당한다. 현재 둘 다 완료했다. [최종 보고](../../reports/final_results.md)를 읽는다. 이 절 이전의 “test를 읽지 않는 경로”는 0.5~2단계 run_workflow.py에만 해당한다.\n';q.write_text(s)
 # Validate newly published relative links.
 docs=[ROOT/'README.md',ROOT/'reports/final_results.md',ROOT/'reports/README.md',TEST/'README.md',ROOT/'reports/make_v2_results.md']
 broken=[]
 for p in docs:
  for target in re.findall(']'+chr(92)+'(([^)]+)'+chr(92)+')',p.read_text()):
   if '://' not in target and not target.startswith('#') and not (p.parent/target.split('#')[0]).exists():broken.append((str(p),target))
 assert not broken,broken
 for out in [TEST,BASE]:
  manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'SHA256SUMS.json' and p.name not in {'smoke_verification.json','implementation_checks.csv'}}
  (out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
 print('FINAL_DOCUMENTS_VERIFIED',len(test),'runs; links pass',flush=True)
 print(td.to_string(index=False),flush=True)
 print(wins[wins.metric=='minority_make_fine_accuracy'].to_string(index=False),flush=True)
if __name__=='__main__':main()
