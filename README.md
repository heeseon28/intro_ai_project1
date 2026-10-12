# 자동차 세부 차종 분류 (Intro to AI · Project 1)

VMMRdb 중고차 매물 사진으로 **50개 제조사-모델**을 분류하고, 기준 CNN의 실패를 분석해 개선하는 팀 프로젝트다.

```text
데이터셋 → 기준 CNN → 실패 분석 → 조사 → 실험 → 개선된 모델 → 시연
```

평가는 최고 정확도가 아니라 **분석, 실험적 추론, 구현, 개선 과정**으로 이루어진다.

## 1. 설정

- 데이터 위치는 `path.md`에 적는다. 팀원마다 자기 경로를 한 줄씩 추가하면 노트북이 존재하는 첫 경로를 쓴다.
- 데이터 폴더 구조:
  - `VMMRdb/{make}_{model}_{year}/*.jpg`
  - `COMPCAR/{image,label,part,misc,train_test_split}`
- 과제 PDF, `PLAN.md`, `TODO.md`, 보고서 원본(`report/`)은 이 저장소 밖 상위 폴더 `Project 1/`에 있다.

| 경로 | 내용 |
|---|---|
| `01_dataset_analysis.ipynb` | Activity 1 데이터 분석 (Part A 분석 · B 50개 클래스 선정 · C 가설·전략) |
| `figures/` | 보고서용 그림 |
| `outputs/selected_50_classes.csv` | 확정된 50개 클래스 (제조사, 모델, 이미지 수, 차체 유형) |
| `outputs/selected_md5.csv` | 50개 클래스 이미지의 md5 (중복 제거·분할에 사용) |
| `outputs/*_index.csv` 외 | 인덱스, 통계, 비교표 캐시 |

## 2. Activity 1 결과 (제출 완료)

| 항목 | 결정 |
|---|---|
| 주 데이터셋 | **VMMRdb** (285,086장, 연식을 합치면 차종 1,175개, Gini 0.80, 공식 분할 없음) |
| 보조 데이터셋 | CompCars (차체 유형 라벨을 근거로 사용) |
| 실험 범위 | 10개 제조사 × 5개 모델 = **50개 클래스, 119,032장** (GMC는 형제차 비교를 위해 일부러 포함) |
| 불균형 | 최대 Honda Civic 6,895장 vs 최소 Jeep Patriot 263장 (26배), 그대로 유지 |
| 제외할 데이터 | 변형 폴더(`civic_coupe` 등), 서로 다른 클래스에 걸친 동일 사진 49그룹, 같은 클래스 안의 중복 사본 512장 |
| 핵심 난제 | 많은 클래스 속의 세밀한 분류 (형제차 Silverado↔Sierra, Tahoe↔Yukon, 트림 325i↔328i) |

**가설**
- **H0:** 기준 CNN은 정확도에 비해 Macro-F1이 낮고, 오답은 닮은 차종끼리 몰릴 것이다.
- **H1:** 제조사(10개) → 차종(5개) 계층 분류가 기준 CNN보다 나을 것이다.
- **H2 (핵심):** 1단계 그룹을 **차체 유형**이나 **혼동 기반 그룹**으로 만든 계층이 제조사 계층보다 나을 것이다. 형제차에서는 제조사 계층이 가장 크게 실패할 것이다.

> 이후 실험은 **최종 제출 PDF**(`Project 1/report/인공지능_프로젝트_1 (1).pdf`)를 기준으로 한다. 노트북 Part C와 한국어 `main.tex`는 제출본과 다른 부분이 있다 (아래 3-0 참고).

## 3. 앞으로 할 일

### 3-0. 정리 (바로)
- [ ] 노트북 Part C를 제출본에 맞춘다.
  - 계층 결합: soft conditioning → **hard routing** (1단계에서 고른 그룹의 분류기만 사용)
  - 가중 손실: `(1/n)^α` → **Cui effective number** `w_c ∝ (1-β)/(1-β^{n_c})`
  - 혼동 기반 그룹은 **검증 세트**로만 만든다.
- [ ] 다음 활동 공지(마감·제출물)를 확인하고 여기에 적는다.

### 3-1. 데이터 파이프라인
- [ ] 제외 목록을 적용한 최종 이미지 목록을 만든다.
- [ ] 계층화 **70/15/15** 분할. md5 중복 그룹은 한 분할에만 넣는다. 분할은 seed를 고정해 CSV로 저장하고 모든 실험이 공유한다.
- [ ] 라벨 3종(차종 50 · 제조사 10 · 차체 유형 6)을 이미지마다 붙인다.
- [ ] 전처리
  - 공통: 짧은 변 256px, 흑백은 RGB로 변환, 정규화
  - 학습: 랜덤 224 크롭, 좌우 반전, 약한 밝기·색 변화
  - 검증·테스트: 224 중앙 크롭
- [ ] 테스트 세트는 최종 평가 전까지 쓰지 않는다.

### 3-2. 기준 모델 (H0 검증)
- [ ] 단순 CNN(Flat, 50개 동시 분류)을 학습한다. 입력 크기, epoch, 조기 종료 기준은 이 단계에서 정한다.
- [ ] seed 3개로 반복하고 Macro-F1의 평균±표준편차를 보고한다.
- [ ] 평가 도구를 만들어 이후 모든 모델에 재사용한다.
  - Top-1 · Top-3 정확도, **Macro-F1(주 지표)**, 클래스별 precision · recall · F1
  - 혼동 행렬(차체 유형 순 정렬), 학습·검증 곡선

### 3-3. 실패 분석
- [ ] 정확도 대 Macro-F1, 소수 클래스의 재현율을 본다 → H0 확인.
- [ ] 오답이 같은 차체 유형·형제차 안에 몰리는지 혼동 행렬로 확인한다.
- [ ] 틀린 사진을 직접 보고 원인(시점, 배경, 라벨 노이즈 등)을 분류한다.
- [ ] 검증 세트 혼동 행렬을 군집화해 **혼동 기반 그룹**을 만든다.

### 3-4. 개선 실험
| 실험 | 비교 대상 | 확인할 것 |
|---|---|---|
| 계층형 분류 (전략 1) | Flat vs 제조사 계층(H1) vs 차체 유형 계층 vs 혼동 기반 계층(H2) | 1단계 정확도, 단계별 오류 전파, 형제차 성능 |
| 가중 손실 (전략 2) | CE vs class-balanced(β는 검증으로 선택) vs focal | 소수 클래스 재현율, Macro-F1 |
| (선택) | 최선의 계층 + 가중 손실, ImageNet 사전학습, Grad-CAM | H2가 기각되면 CNN이 엠블럼을 보는지 확인 |

- 모든 모델은 같은 테스트 세트로 평가한다. seed 간 편차보다 일관되게 큰 개선만 인정한다.

### 3-5. 시연
- [ ] 사진 업로드 → Flat 모델과 계층 모델들의 예측을 나란히 보여준다 (1단계 그룹과 차종 Top-3, 신뢰도).
- [ ] 준비할 사례: (a) 쉬운 차, (b) 형제차 Silverado vs Sierra, (c) 소수 클래스의 가중 손실 전후 비교
- [ ] 클래스별 F1 막대그래프와 혼동 행렬(기준 vs 최종)
- [ ] 최고 확률이 검증 세트로 정한 임계값보다 낮으면 신뢰도 낮음 경고를 띄운다.

## 참고문헌
1. Tafazzoli et al., *A Large and Diverse Dataset for Improved Vehicle Make and Model Recognition*, CVPRW 2017 — https://github.com/faezetta/VMMRdb
2. Yang et al., *A Large-Scale Car Dataset for Fine-Grained Categorization and Verification*, CVPR 2015 — https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/
3. Yan et al., *HD-CNN: Hierarchical Deep Convolutional Neural Networks for Large Scale Visual Recognition*, ICCV 2015
4. Cui et al., *Class-Balanced Loss Based on Effective Number of Samples*, CVPR 2019
5. Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017
