# 자동차 세부 차종 분류 (Intro to AI · Project 1)

## 최종 실험 종료

추가 학습·설정 탐색을 종료하고, 검증으로 고정한 A/B/C/D × seed 0·1·2의 최종 테스트를 완료했다. **검증과 테스트를 구분해 보고하며, 테스트를 보고 모델을 다시 고르지 않았다.** 최종 모델은 테스트 전에 선정한 A Flat이다.

[최종 결과 보고](reports/final_results.md) → [테스트 원자료](result_full/make_v2/test/README.md) → [검증 결과집](result_full/make_v2/README.md) 순서로 읽는다.

## 최종 테스트 성능

| 모델 | 정확도 (%) | Macro-F1 (%) | 소수 제조사 평균 정확도 (%) |
|---|---:|---:|---:|
| A Flat | 94.480 ± 0.170 | 92.498 ± 0.222 | 86.743 ± 0.232 |
| B 제조사 계층 (soft) | 94.279 ± 0.073 | 92.429 ± 0.130 | 86.897 ± 0.357 |
| C 계층+제조사 가중치 | 94.071 ± 0.135 | 92.336 ± 0.223 | 87.839 ± 0.659 |
| D Flat+제조사 가중치 | 94.283 ± 0.150 | 92.404 ± 0.155 | 87.811 ± 0.400 |

3 seed 평균 ± 표본 표준편차. 소수 제조사는 BMW·GMC·Volkswagen의 차종 정확도를 제조사별 동등하게 평균했다.

## 완료·미실행 구분

| 항목 | 최종 상태 |
|---|---|
| 데이터 선정·중복 제외·고정 70/15/15 분할 | 완료, 분할과 라벨 유지 |
| Task 2 기준 CNN·1차 A/B/C 3 seed 비교 | 완료 |
| 공통 오답 321장+대조군 100장·정답 제조사 진단·근접 중복 후보 분석 | 완료 |
| uint8 캐시·letterbox 선별 | 완료, 각각 속도/주 지표 이득 없어 미채택 |
| 2차 D Flat+가중치 3 seed·네 모델 비교 | 완료 |
| 검증 선택 고정·12개 최종 테스트·예측 재검산 | 완료 |
| H2 차체 유형·혼동 기반 계층 | 미실행으로 종료. 지지/기각 불가 |
| 무작위 초기화 통제·고해상도·다른 불균형 기법·Grad-CAM | 미실행으로 종료 |
| 최종 보고서용 결과 정리 | reports/final_results.md 완료 |
| 발표 슬라이드·실시간 시연 앱 | 별도 제출 준비 항목. 여기서 완료로 주장하지 않음 |

옛 “앞으로 할 일” 체크리스트는 이 종료표로 대체했다. 초기 계획과 변경 이력은 Git 기록 및 PROJECT_FLOW.md에 남아 있다. **미실행 항목을 추가 필수 실험이나 완료 항목으로 읽지 않는다.**

## 실제 수행한 실험

A Flat / B 제조사 계층(soft) / C 계층+제조사 가중치 / D Flat+제조사 가중치. 공통 ImageNet 사전학습 ResNet-18, 224px, AdamW 3e-4, wd 1e-4, cosine, batch 64, 최대 30 epoch, 검증 Macro-F1 최고 체크포인트. 가중치는 학습 제조사 빈도의 역제곱근이며 이미지 평균 가중치 1로 정규화했다.

B−A와 C−D는 계층 효과, D−A와 C−B는 가중치 효과다. A/B/C 9회는 1차 결과를 재사용하고 D 3회만 추가 학습했다. Letterbox seed 0 선별 1회는 최종 12회와 별도이며, 미채택 후 기존 전처리를 유지했다. 무작위 초기화 통제나 H2 비교를 수행한 것처럼 쓰지 않는다.

## 폴더 지도

| 경로 | 용도 / 읽는 방법 | Git 보관 |
|---|---|---|
| [reports/](reports/README.md) | 최종 종료 보고·검증 해설·그림 | 문서 |
| [result_full/make_v2/](result_full/make_v2/README.md) | 검증 결과집. A/B/C 기존 9회 + D 신규 3회, 전처리 선별·진단 포함 | CSV·JSON·집계 그림·소스 |
| [result_full/make_v2/test/](result_full/make_v2/test/README.md) | 고정한 12개 체크포인트의 최종 테스트 결과·선택 고정 기록 | CSV·JSON·집계 그림·소스 |
| [result_full/make_v2/stage0/](result_full/make_v2/stage0/README.md) | 정답 제조사 진단, 421장 사진 검토 기록, 근접 중복 후보와 민감도 분석 | 표·집계 그림, 원본 사진 제외 |
| [result_full/make_v2/cache_benchmark/](result_full/make_v2/cache_benchmark/README.md) | uint8 캐시 입력 동일성·속도 측정. 이번 구현은 느려서 미채택 | 측정값·설명 |
| [result_full/make_v1/](result_full/make_v1/) | 1차 본 실험 A/B/C × seed 0·1·2 결과집 | 기존 결과 보관 |
| [result_task2/](result_task2/README.md) | Task 2 무작위 초기화 기준 CNN의 팀원별 결과 | 검증 결과·재현 정보 |
| result_pilot/ | 12 epoch 예비 실험의 로컬 참고 결과. 본 실험 근거와 구분 | 이번 업로드 제외 |
| [experiments/make_v2/](experiments/make_v2/README.md) | 2차 진단·전처리·캐시 측정·연속 실행·결과집 생성 소스 | 소스·실행 계획 |
| full_train.py / pilot_train.py | 서버 본 학습 공통 구현 / 기존 예비 실행 구현. V2 실행 진입점은 experiments.make_v2.run_workflow | 소스 |
| runs_full/make_v1/, runs_full/make_v2/ | 서버 학습 원본: 체크포인트·로그·중간 결과·검토용 사진. 삭제하지 않는다 | 전체 제외 |
| runs_pilot/, runs_smoke/ | 예비 실행·구현 점검·GPU 잠금 등 로컬 작업 자료 | 제외 |
| [task2/](task2/README.md) | Task 2 기준 학습 코드와 보고서·제출 패키지 폴더 | 기존 추적 자료, 임시 산출물 제외 |
| [task2_preliminary_seojin/](task2_preliminary_seojin/) | 서진 팀원 기준 모델 실행·검증 보조 코드와 결과 | 기존 자료 |
| [splits/](splits/) | 고정 train/val/test 목록과 클래스 정보. 분할을 다시 만들지 않는다 | CSV |
| dataset/VMMRdb, dataset/COMPCAR | 원본 이미지 데이터. 현재 본 실험은 VMMRdb 사용 | 제외 |
| [outputs/](outputs/) | 클래스 선정·이미지 인덱스·중복 제거 등에 쓴 메타데이터 | 추적 중인 CSV, automation 제외 |
| [figures/](figures/) | Activity 1 데이터 분석·보고서용 그림 | 집계 그림 |
| cache/, .venv-project-runner/, __pycache__/, .vscode/ | 캐시·서버 가상환경·Python 바이트코드·편집기 로컬 설정 | 제외 |
| 01~03 노트북 | 데이터 분석 → 고정 분할 생성 → 학습·평가 도구 | 소스 노트북 |
| [04_results_analysis.ipynb](04_results_analysis.ipynb) | 1차 결과 심층 분석. 2차 결과는 reports/ 문서를 볼 것 | 1차 분석 기록 |
| [PROJECT_FLOW.md](PROJECT_FLOW.md) | 프로젝트 이력·실험 비교의 의미·파일 지도 | 팀원 안내 |

result_full은 공유할 결과집, runs_full은 서버의 원본 실행 자료다. evidence/는 이미지 대신 예측·지표를, sources/는 실행 당시 코드를 보관한다. 체크포인트·원본 이미지·캐시·smoke 결과는 Git 제외다.

## 재현과 코드 사용 시 주의

패키지는 requirements.txt와 각 실행의 환경 기록을 참고한다. V2 진입점은 experiments.make_v2.run_workflow이며 완료 상태 확인을 위해 다시 실행하지 않는다. 최종 평가 코드는 experiments.make_v2.final_evaluate, 저장 결과로 문서 생성은 experiments.make_v2.finalize_project다. 이미 완료한 테스트는 저장 예측을 재사용한다.

02 노트북의 분할 생성은 기존 splits/를 덮어쓰지 않도록 주의한다. 03 노트북은 초기 연구 도구이며 최종 실행과 다른 hard routing·Cui β 등의 기능을 포함한다. 03의 val 혼동 기반 그룹 생성은 최종 A/B/C/D에서 사용하지 않았다. 향후 재사용할 때는 학습 내부 검증/OOF로 바꿔야 하며, 이 미수정 기능을 이번 결과의 검증 근거로 쓰지 않는다. 과거 캐시 속도 권고나 12 epoch 계획은 이번 서버 측정·30 epoch 실행을 설명하지 않는다.

## 참고문헌
1. Tafazzoli et al., *A Large and Diverse Dataset for Improved Vehicle Make and Model Recognition*, CVPRW 2017 — https://github.com/faezetta/VMMRdb
2. Yang et al., *A Large-Scale Car Dataset for Fine-Grained Categorization and Verification*, CVPR 2015 — https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
3. Yan et al., *HD-CNN: Hierarchical Deep Convolutional Neural Networks for Large Scale Visual Recognition*, ICCV 2015
4. Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019
5. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
6. Cao et al., *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss* (DRW), NeurIPS 2019
7. Kang et al., *Decoupling Representation and Classifier for Long-Tailed Recognition* (cRT, 샘플링), ICLR 2020
8. Ren et al., *Balanced Meta-Softmax for Long-Tailed Visual Recognition*, NeurIPS 2020
9. Menon et al., *Long-Tail Learning via Logit Adjustment*, ICLR 2021
