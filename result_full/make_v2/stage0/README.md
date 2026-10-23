# make_v2 0단계 진단 결과

0단계 완료. 새 학습을 실행하지 않았으며, 1단계는 사용자 승인 대기다. 테스트 세트는 읽지 않았다. 원래 train/val/class_info 파일의 SHA-256은 시작 시점과 일치한다. 검증·학습 이미지와 라벨은 수정하거나 삭제하지 않았다.

## 결론과 제안

**1단계 letterbox Flat seed 0 비교를 제안한다. 다만 개선을 예고할 근거는 충분하지 않다.** 중앙 크롭은 실제 단서를 자르지만, 단서 잘림 비율은 공통 오답 76.6%와 정답 대조군 78%로 비슷했다. 따라서 “가로가 잘려서 오류가 생겼다”는 인과 설명은 아직 지지되지 않는다. letterbox는 단서를 보존하는 대신 차량의 유효 해상도를 낮추고 여백을 늘리므로, 순이득은 실험으로 판정해야 한다.

공통 오답에는 명백한 차체/브랜드 불일치로 보이는 라벨 의심 사례도 있다. 113장은 **검토 후보**이며 확정 라벨 오류 수가 아니다. 별도 전문가 확인과 데이터셋의 연식·명명 규칙 점검이 필요하다. 이 결과를 근거로 검증/테스트를 정제하거나 지표를 교체하지 않는다.

## 0-1. 정답 제조사를 공급한 진단

V1 seed 0·1·2의 A/B/C 체크포인트 9개를 검증 17,771장에만 추론했다. 일반 예측은 **모든 실행에서 기존 CSV와 17,771/17,771개 일치**했다. A는 정답 제조사에 속한 차종 점수만 남겼고 B/C도 정답 제조사 내부 argmax를 사용했다. soft 결합의 제조사 항은 한 제조사 안에서 상수이므로 내부 순위는 동일하다.

아래는 3 seed 평균, 단위 %다. 오류율의 분모는 전체 검증 이미지다.

| model | mode | accuracy_pct | macro_f1_pct | within_error_pct_val | cross_error_pct_val |
|---|---|---|---|---|---|
| flat | oracle_true_make | 96.7288 | 94.4997 | 3.2712 | 0.0000 |
| flat | saved_normal | 94.2209 | 92.0167 | 2.5360 | 3.2431 |
| hierarchy | oracle_true_make | 96.6462 | 94.5349 | 3.3538 | 0.0000 |
| hierarchy | saved_normal | 94.0746 | 91.8295 | 2.7273 | 3.1981 |
| hierarchy_make_weighted | oracle_true_make | 96.6443 | 94.6257 | 3.3557 | 0.0000 |
| hierarchy_make_weighted | saved_normal | 94.0446 | 91.9622 | 2.7423 | 3.2131 |

A 정확도는 94.2209→96.7288(+2.5079%p), Macro-F1은 92.0167→94.4997(+2.4830%p)다. A의 다른 제조사 오류 3.2431%p가 전부 복구되지는 않는다. 약 77.3%만 복구되고 남은 약 0.735%p는 정답 제조사 안에서도 차종을 틀린다. 정답 제조사를 공급한 뒤 같은 제조사 오류율이 증가하는 것은 원래 다른 제조사로 틀렸던 사례 일부가 같은 제조사 오답으로 바뀌기 때문이다. 기존 같은 제조사 오답이 악화됐다는 뜻이 아니다.

이는 정답을 이용한 **진단**이며 실제 운용 성능이나 계층형 학습의 개선 증거가 아니다. 원래 계층형 모델의 B−A 비교와 구분한다. 모델별·seed별 차이와 표준편차는 oracle_paired_deltas.csv, oracle_delta_summary.csv에 있다.

## 0-2. 사진 421장 전수 화면 검토

9회 모두 틀린 321장 전부와 9회 모두 맞힌 사진 중 무작위 100장(seed 20261022)을 차종별 원본/실제 중앙 크롭 쌍으로 보고 1회 분류했다. 각 사진에 주 유형 하나를 부여했다. 배경의 작은 차량은 다중 차량의 주 문제로 세지 않았다. 0건은 해당 현상이 데이터 전체에 없다는 뜻이 아니다.

| primary_category | all9_correct_control | all9_wrong |
|---|---|---|
| suspected_label_error | 0 | 113 |
| multiple_vehicles | 0 | 0 |
| partial_vehicle | 0 | 4 |
| occlusion | 1 | 5 |
| low_resolution | 3 | 7 |
| unusual_view | 0 | 20 |
| sibling_or_trim_ambiguous | 0 | 119 |
| other | 96 | 53 |

라벨 오류는 **의심**으로만 분류했다. 예: Mustang 폴더의 픽업/SUV, BMW 328i 폴더의 Infiniti FX35, Silverado 폴더의 GMC 표시 차량. Cherokee/Grand Cherokee 혼용 의심도 포함되므로 분류체계 확인이 필요하다. 형제차/트림 모호 태그는 닮아 보이는 검토 후보이며 사람도 구분 불가능하다는 판정이 아니다. 저해상도는 적은 픽셀 또는 심한 흐림을 화면에서 판독했다. other는 나열된 문제가 뚜렷하지 않거나 원인을 결정하지 못한 경우다.

### 엠블럼·그릴·테일램프 잘림

yes는 원본에서 보이는 단서 중 하나 이상이 중앙 크롭 경계 밖으로 **일부 또는 전부** 사라졌다는 화면 판정이다. 모든 단서를 잃었다거나 차종 판별에 필수인 단서라는 뜻은 아니다. 불분명하면 uncertain으로 뒀다.

| crop_cuts_visible_cue | all9_correct_control | all9_wrong |
|---|---|---|
| yes | 78 | 246 |
| no | 18 | 65 |
| uncertain | 4 | 10 |

공통 오답: 246/321=76.6%, 정답 대조군: 78/100=78.0%. 두 집단은 전체 검증 무작위 표본이 아니라 예측 결과로 선택됐고 클래스 구성도 다르다. 또한 검토자가 정답/오답 집단과 라벨을 알고 봤으며 독립 검토자가 없으므로 편향·오분류 가능성이 있다. 이 비율로 전체 데이터의 오류 원인 비중이나 전처리의 효과를 추정하지 않는다.

검증 전처리 기하 측정: 가로 이미지 93.174%, 종횡비 중앙값 1.3361, 잘린 가로 폭 비율 중앙값 34.503%. 이는 **이미지 폭**의 손실이며 차량 영역이나 판별 정보의 34.5% 손실을 뜻하지 않는다. 321장의 분모는 전체 검증 오답이 아니라 9회 공통 오답이다.

화면 검토 기록: visual_audit.csv. 원본 경로와 36개 contact sheet 및 대표 사례는 Git 제외 폴더 runs_full/make_v2/stage0/에 보관한다. 대표 패널은 representative_cases.png, 예시 ID 101·123·187·305·337·366·370·420이다.

## 0-3. 학습↔검증 근접 중복 점검

PIL로 직접 구현한 64-bit 수평 dHash(회색조 9×8 Lanczos, 인접 밝기 비교), 해밍 거리 ≤4. 5개 비트 구간으로 후보를 검색한 뒤 전체 64-bit 거리를 다시 계산했다. 새 패키지는 설치하지 않았다. 범위는 학습 82,929장과 검증 17,771장이며 테스트는 제외했다.

670쌍, 서로 다른 검증 이미지 548장(3.084%)이 **후보**다. 거리별 쌍 수는 0:408, 1:113, 2:71, 3:35, 4:43. 라벨이 다른 후보는 11쌍이다. 해시 일치는 같은 파일의 증명이 아니며, 다른 차종의 흰 배경 홍보 사진도 충돌한다. 크롭·반전·시점 변화에 따른 중복을 놓칠 수도 있다.

라벨 불일치 11쌍 전부와 같은 라벨 거리별 4쌍씩 20쌍, 합계 31쌍을 직접 대조했다. 21쌍은 같은 사진/작은 변형으로 보였고 10쌍은 다른 사진이었다. 라벨 불일치 11쌍 중 2쌍은 같은 사진이었다. **나머지 639쌍은 시각 확인하지 않았으며, 전체 확정 중복 이미지 수는 미상**이다. 이 층화 표본의 21/31을 전체 중복률로 외삽하지 않는다. 개별 판정은 duplicate_visual_sample.csv에 있다.

후보가 하나라도 있는 검증 548장 모두를 계산상 제외한 17,223장 부분집합으로 기존 CSV 점수를 다시 계산했다. 물리적 삭제나 분할 수정, 재학습은 없다. 오탐도 제외한 보수적 민감도 분석이므로 공식 검증 점수의 대체값으로 쓰지 않는다.

| model | subset | n | accuracy_pct | macro_f1_pct | within_error_pct_val | cross_error_pct_val |
|---|---|---|---|---|---|---|
| flat | excluding_dhash_candidates | 17223.0000 | 94.0796 | 91.8572 | 2.6070 | 3.3134 |
| flat | full_validation | 17771.0000 | 94.2209 | 92.0167 | 2.5360 | 3.2431 |
| hierarchy | excluding_dhash_candidates | 17223.0000 | 93.9229 | 91.6578 | 2.8063 | 3.2708 |
| hierarchy | full_validation | 17771.0000 | 94.0746 | 91.8295 | 2.7273 | 3.1981 |
| hierarchy_make_weighted | excluding_dhash_candidates | 17223.0000 | 93.8919 | 91.7970 | 2.8218 | 3.2863 |
| hierarchy_make_weighted | full_validation | 17771.0000 | 94.0446 | 91.9622 | 2.7423 | 3.2131 |

평균 Macro-F1: A 92.0167→91.8572, B 91.8295→91.6578, C 91.9622→91.7969. 이 제외 규칙에서도 A>B/C의 평균 순위는 유지된다. 모델별 동일 seed 차이는 다음과 같다.

| subset | comparison | macro_f1_delta_pp | sd_delta_pp | improved_seeds |
|---|---|---|---|---|
| excluding_dhash_candidates | hierarchy minus flat | -0.1994 | 0.1879 | 0 |
| excluding_dhash_candidates | hierarchy_make_weighted minus flat | -0.0603 | 0.0507 | 0 |
| full_validation | hierarchy minus flat | -0.1872 | 0.1986 | 0 |
| full_validation | hierarchy_make_weighted minus flat | -0.0545 | 0.0371 | 0 |

점수 변화에는 표본 수/클래스 구성 변화도 섞인다. 이 차이를 중복으로 부풀려진 성능의 인과 추정치로 해석하지 않는다.

## 산출물과 재현

- stage0_summary.png, visual_categories.png: 원본 사진 없이 집계한 그림.
- oracle_metrics.csv 및 oracle_seed*.csv: 진단/일반 예측과 seed별 지표.
- visual_audit.csv, visual_category_counts.csv, visual_crop_counts.csv: 421장 검토 기록과 집계.
- near_duplicate_candidates.csv, duplicate_visual_sample.csv: 자동 후보와 31쌍 시각 판정.
- duplicate_sensitivity_metrics.csv/mean.csv/paired.csv: 후보 제외 민감도.
- validation_crop_geometry.csv, protocol.json: 전처리 기하와 고정 분할 지문.
- 서버 원본/사진/캐시/로그: /home/work/Intro_AI/intro_ai_project1/runs_full/make_v2/stage0/.
- 소스: experiments/make_v2/stage0_diagnostics.py, stage0_audit_report.py, stage0_finalize.py.

정답 제조사 추론/해시는 완료 상태이며 새 학습은 실행하지 않았다. 원본 이미지·체크포인트·캐시는 Git에 넣지 않는다. 커밋/푸시는 아직 하지 않았다.

## 승인 후 다음 단계

1단계만: 기존 V1 Flat seed 0을 재사용하고 letterbox Flat seed 0을 1회 실행한다. 지표는 Macro-F1, 정확도, 같은/다른 제조사 오류율, 사전에 고정한 형제차 방향별 오류 수, 소수 제조사 차종 정확도다. 학습 전 어떤 쌍/방향을 집계할지 정확한 클래스명으로 고정한다. 1 seed 결과는 선별용이며 3/3 반복 기준을 충족했다는 판정은 하지 않는다.

letterbox가 유리하면 승인 후 4모델×3 seed로 진행한다. 불리하면 기존 A/B/C를 재사용하고 Flat+가중치 D 3회만 수행한다. 계층 효과는 B−A와 C−D, 가중치 효과는 D−A와 C−B로 분리한다. 데이터 처리 효과는 새 A−기존 A다. 가설을 지지하도록 선택적으로 결과/전처리를 바꾸지 않는다.

교수님 제안의 텐서 캐시는 아직 적용하지 않았다. 승인 후 JPEG 디코딩 전/후 병목과 메모리·저장 용량을 먼저 측정하고, 증강 전 RGB uint8 캐시만 후보로 둔다. 무작위 증강을 미리 고정하거나 정규화/리사이즈 결과를 몰래 바꾸지 않는다. 기존 경로와 출력/예측 일치, 처리량을 짧게 확인한 뒤 채택 여부를 결정한다.
