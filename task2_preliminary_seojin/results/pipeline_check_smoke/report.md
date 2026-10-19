# Task 2 예비 실험 보고서

작성자: 한서진 (팀 정보 추가 필요)

> **파이프라인 시험(SMOKE): 최종 성능 결과로 제출하거나 전체 실험과 비교할 수 없음.**
> 일부 학습 배치와 축소된 검증 표본으로 입출력·학습·산출물 생성만 확인했다.

실행: `pipeline_check_smoke` · 완료 epoch: 1 · 검증 macro-F1 최고 checkpoint: epoch 1

## 데이터와 기본 모델

VMMRdb의 기존 50개 차종 분할을 사용하는 Task 2 기준선이다. CompCars는 이번 기준선 학습에 사용하지 않는다.
원본 경로: `C:\Users\hseoj\OneDrive - Fraunhofer\Desktop\켄텍2 한서진\학부\4학년\4-1 (2026 2학기)\03_Introduction to AI\Project 1\VMMRdb`.
분할 명세 표본 수: train 82,929, val 17,771, test 17,771; 클래스 50개. 테스트 성능은 측정하지 않았다.
위 분할 명세 수는 전체 분할 기준이다. 본 시험의 실제 검증 표본 수는 50개이며, train 지표도 제한된 배치에서 계산했다.
저장된 MD5 기준 분할 간 중복은 없으며, 근접 중복 검증 범위는 별도로 확인해야 한다.

구조: torchvision ResNet-18; 초기화: random, weights=None; RGB 224×224.
전처리: deterministic square resize, ToTensor, Normalize {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}.
학습: Adam, lr=0.0003, unweighted CrossEntropyLoss, batch=8, epoch=1, seed=0.
설정 기록: 전이학습=False, 증강=False, 재균형=False, 계층학습=False, 학습률 스케줄=False.

## 학습·검증 결과

검증 macro-F1으로 선택한 동일한 고정 checkpoint를 재평가한 수치이다.

| 분할 | Loss | Accuracy | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Train | 3.9598 | 0.00% | 0.0000 | 0.00% |
| Validation | 3.9107 | 2.00% | 0.0008 | 2.00% |

동일 checkpoint의 train−val 정확도 차이는 -2.00%p, macro-F1 차이는 -0.08%p이며, val−train loss 차이는 -0.0491이다. 축소 표본과 제한된 배치로 측정했으므로 이 값으로 과적합·과소적합이나 실제 분류 성능을 판단하지 않는다.

학습 곡선의 train 값은 가중치 업데이트 중 누적한 값이므로 위 고정 checkpoint 평가와 직접 일치하지 않을 수 있다.

## 어려운 차종과 주요 혼동

검증 recall이 낮은 순서이며 동률에서는 F1, 차종명 순으로 정렬했다.

| 차종 | Recall | F1 | 검증 표본 |
|---|---:|---:|---:|
| bmw/325i | 0.00% | 0.0000 | 1 |
| bmw/328i | 0.00% | 0.0000 | 1 |
| bmw/528i | 0.00% | 0.0000 | 1 |
| bmw/530i | 0.00% | 0.0000 | 1 |
| bmw/x5 | 0.00% | 0.0000 | 1 |
| chevrolet/corvette | 0.00% | 0.0000 | 1 |
| chevrolet/impala | 0.00% | 0.0000 | 1 |
| chevrolet/malibu | 0.00% | 0.0000 | 1 |
| chevrolet/silverado | 0.00% | 0.0000 | 1 |
| dodge/charger | 0.00% | 0.0000 | 1 |

검증 혼동 건수 상위 5개 방향(실제 → 예측):

| 실제 차종 | 예측 차종 | 건수 | 실제 클래스 내 비율 |
|---|---|---:|---:|
| ford/mustang | chevrolet/tahoe | 1 | 100.00% |
| ford/f150 | chevrolet/tahoe | 1 | 100.00% |
| ford/explorer | chevrolet/tahoe | 1 | 100.00% |
| ford/taurus | chevrolet/tahoe | 1 | 100.00% |
| ford/focus | chevrolet/tahoe | 1 | 100.00% |

## 근거 그림과 다음 실험

- [epoch별 학습·검증 곡선](learning_curves.png)
- [검증 혼동 행렬(행 정규화)](confusion_matrix.png)
- [검증 성공·실패 예시](validation_examples.png)

혼동 쌍의 예시 이미지를 확인해 시점·배경·유사 외형의 영향을 검토한다. 이러한 원인은 혼동 횟수만으로 단정할 수 없다. 다음 실험은 동일한 분할과 기준 설정을 유지하고 하나의 개선 요소씩 비교한다. 단일 seed 결과로 통계적 유의성을 주장하지 않는다.

재현 근거: [설정](config.json), [최종 지표](metrics.json), [학습 이력](history.csv), [클래스별 지표](per_class_metrics.csv), [혼동 건수](confusion_matrix_counts.csv).
