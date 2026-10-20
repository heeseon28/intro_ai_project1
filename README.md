# 자동차 세부 차종 분류 (Intro to AI · Project 1)

VMMRdb 중고차 매물 사진으로 **50개 제조사-모델**을 분류하고, 기준 CNN의 실패를 분석해 개선하는 팀 프로젝트다.

```text
데이터셋 → 기준 CNN → 실패 분석 → 조사 → 실험 → 개선된 모델 → 시연
```

평가는 최고 정확도가 아니라 **분석, 실험적 추론, 구현, 개선 과정**으로 이루어진다.

## 1. 시작하기

### 설치
```bash
pip install -r requirements.txt
```
- **Mac (Apple Silicon):** 그대로 설치하면 `mps` 가속을 쓴다.
- **Windows/Linux + NVIDIA GPU:** torch를 먼저 [공식 안내](https://pytorch.org/get-started/locally/)대로 CUDA 버전으로 설치한 뒤 위 명령을 실행한다.
- 03 노트북을 처음 실행할 때 ImageNet 사전학습 ResNet18 가중치(약 45MB)를 자동으로 내려받는다. 인터넷 연결이 필요하다.

### 경로 설정
노트북 **02, 03의 맨 위 셀**에서 자기 컴퓨터의 경로로 `VMMR_ROOT`를 고친다. 다른 셀은 고칠 필요가 없다.
```python
VMMR_ROOT = Path(r"C:\Users\...\VMMRdb")      # Windows 예시
VMMR_ROOT = Path(r"/Users/.../VMMRdb")        # Mac 예시
```
- 데이터 폴더 구조: `VMMRdb/{make}_{model}_{year}/*.jpg`, `COMPCAR/{image,label,part,misc,train_test_split}`
- 01 노트북은 `path.md`에 적힌 경로 중 존재하는 첫 경로를 쓴다.
- 과제 PDF, `PLAN.md`, `TODO.md`, 보고서 원본(`report/`)은 이 저장소 밖 상위 폴더 `Project 1/`에 있다.

### 실행 순서
| 순서 | 노트북 | 하는 일 | 소요 시간 |
|---|---|---|---|
| 1 | `02_data_pipeline.ipynb` | 제외 규칙 적용 → 라벨 3종 → 70/15/15 분할 → `splits/` 저장. 선택으로 256px 리사이즈 캐시 생성 (`BUILD_CACHE = True`) | 분할 1~2분, 캐시 생성 수 분~십여 분 (1회) |
| 2 | `03_train_evaluate.ipynb` | 불균형 실험(6-1) → 혼동 기반 그룹 생성(6-2) → 계층 실험(6-3) → 결과 비교(7) | 실험 1개(12 epoch)당 로컬에서 수십 분~수 시간 |

- `splits/`는 seed가 고정돼 있어 누가 만들어도 같다. git으로 공유된 것을 그대로 써도 된다.
- **캐시를 꼭 만드는 것을 추천한다.** 원본(600×450 JPEG)을 매번 디코딩하는 것보다 학습이 훨씬 빨라진다.
- 처음에는 03 맨 위 셀에서 `SMOKE_TEST = True`로 두고 코드가 끝까지 도는지 먼저 확인한다. 아주 작은 데이터로 1 epoch만 돌리고, 결과는 `runs_smoke/`에 따로 저장된다.
- 03은 이미 끝난 실험(`runs/<실험>/seed<k>/metrics.json`)을 건너뛴다. 중간에 멈춰도 다시 실행하면 이어서 돈다. 팀원끼리 실험을 나눠 돌리고 `runs/` 결과(체크포인트 제외)를 모으면 된다.
- Mac·Windows의 Jupyter에서는 `num_workers = 0`이어야 한다. Linux + CUDA 서버라면 03의 `BASE["num_workers"]`를 4~8로 올린다.

### 파일 구성
| 경로 | 내용 |
|---|---|
| `requirements.txt` | 필요한 패키지 |
| `01_dataset_analysis.ipynb` | Activity 1 데이터 분석 (Part A 분석 · B 50개 클래스 선정 · C 가설·전략) |
| `02_data_pipeline.ipynb` | 학습용 이미지 목록과 분할 만들기 |
| `03_train_evaluate.ipynb` | 모델 · 불균형 툴킷 · 계층형 분류 · 학습 · 평가 · 실험 비교 |
| `04_results_analysis.ipynb` | Task 2 기준 모델과 본 실험(3모델 × 3 seed) 결과 정리와 해석 |
| `result_task2/`, `result_full/make_v1/` | 기준 모델 결과, 본 실험 결과집 (서버 실행분) |
| `splits/{train,val,test}.csv` | 분할 (경로는 `VMMR_ROOT` 기준 상대 경로) |
| `splits/class_info.csv` | 50개 차종의 번호, 제조사, 차체 유형, head/medium/tail 구간, 분할별 개수 |
| `runs/` | 실험 결과 (`metrics.json`, 혼동 행렬, 예측, 학습 곡선, `summary.csv`). `*.pt`는 git 제외 |
| `cache/` | 리사이즈 캐시 (git 제외) |
| `figures/` | 보고서용 그림 |
| `outputs/selected_50_classes.csv` | 확정된 50개 클래스 (제조사, 모델, 이미지 수, 차체 유형) |
| `outputs/selected_md5.csv` | 50개 클래스 이미지의 md5 (중복 제거·분할에 사용) |
| `outputs/*_index.csv` 외 | 인덱스, 통계, 비교표 캐시 |
| `preprocessing.py`, `compcar_proprocessing_hyeonil.py`, `*_metadata*.csv` | 팀원이 작성한 메타데이터 스크립트 (절대 경로 기반, 02·03에서는 쓰지 않음) |

## 2. 학습 코드 요약 (`03_train_evaluate.ipynb`)

- **백본:** ImageNet 사전학습 ResNet18. 모든 모델이 같은 백본을 쓰므로 비교는 공정하지만, 보고서에는 이 점을 명시해야 한다.
- **입력:** train은 랜덤 크롭 224 + 좌우 반전 + 약한 색 변화, val·test는 중앙 크롭 224. 정규화는 ImageNet 값.
- **학습:** AdamW (lr 3e-4) + cosine 감소, 최대 12 epoch. **val Macro-F1이 가장 좋은 epoch**를 고르고 patience 4로 조기 종료한다.
- 실험 하나는 `BASE` 설정에서 **바꿀 값만 덮어쓴 dict 한 줄**로 정의한다.

### 불균형 툴킷
| 층위 | 기법 | 설정 예 |
|---|---|---|
| 손실 가중 | 역빈도 `(1/n)^α` (수업 코드 일반화) | `weighting="inv", alpha=0.5` |
| 손실 가중 | Class-Balanced `(1-β)/(1-β^n)` (Cui 2019) | `weighting="cb", beta=0.999` |
| 손실 가중 | 수동 보정 | `manual_w={"gmc/sierra_1500": 2.0}` |
| 손실 형태 | Focal (Lin 2017) | `loss="focal", gamma=2` |
| 손실 형태 | Logit-adjusted / Balanced Softmax (Menon 2021, Ren 2020) | `loss="la", tau=1` |
| 샘플링 | 반복 샘플링: 클래스 확률 ∝ n^q (1 그대로 · 0.5 제곱근 · 0 균형) | `q=0.5` |
| 스케줄 | Deferred Re-Weighting (Cao 2019) | `weighting="cb", drw_epoch=8` |
| 사후 보정 | 추론 시 logit 보정, 재학습 없음 (Menon 2021) | `from_run="flat_ce", posthoc_tau=1` |
| 분리 학습 | cRT: 백본 고정, 분류층만 균형 샘플링으로 재학습 (Kang 2020) | `from_run="flat_ce", crt_epochs=10` |

### 계층형 분류 (hard routing)
- `hier="make"`(10 그룹) · `"body"`(6) · `"confusion"`(기준 모델의 혼동 행렬을 spectral clustering으로 6개 그룹)
  - ⚠️ 현재 코드는 **val** 혼동 행렬로 그룹을 만든다. 같은 val로 모델까지 고르고 평가하면, 평가가 그룹 설계에 오염된다. **학습 세트 안에서 떼어낸 내부 검증(또는 out-of-fold 예측)으로 그룹을 만들도록 바꿔야 한다**(4-3 할 일).
- 1단계에서 그룹 하나를 고르고, **그 그룹의 분류기만** 써서 차종을 고른다. 학습 손실은 `CE(그룹) + CE(정답 그룹 안의 차종)`이다.
- 참고용으로 soft 결합(`P(그룹) × P(차종|그룹)`) Top-1도 함께 기록한다.

### 평가 지표 (`runs/<실험>/seed<k>/metrics.json`, test 기준)
- `top1`, `top3`, **`macro_f1` (주 지표)**, `bal_acc`
- `recall_head / medium / tail`: train 이미지 수 3분위 구간별 평균 recall
- `make_acc`, `body_acc`: 틀려도 제조사·차체 유형은 맞혔는지
- `twin_confusion`: 형제차·트림 쌍끼리의 상호 오분류율
- 계층 전용: `stage1_acc`, `stage2_acc_given1`, `err_from_stage1`(오답 중 1단계에서 생긴 비율), `soft_top1`
- 클래스별 recall·F1

## 3. Activity 1 결과 (제출 완료)

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

> 이후 실험은 **최종 제출 PDF**(`Project 1/report/인공지능_프로젝트_1 (1).pdf`)를 기준으로 한다. 노트북 Part C와 한국어 `main.tex`는 제출본과 다른 부분이 있다 (아래 4-0 참고).

## 3-1. 실험 결과 요약 (Task 2 기준 모델 + 본 실험)

정리와 해석 전체는 **[`04_results_analysis.ipynb`](04_results_analysis.ipynb)**에 있다. 수치는 모두 검증 세트 기준이며, 테스트 세트는 아직 평가하지 않았다.

| 모델 | 조건 | 정확도 | Macro-F1 | 소수 제조사 차종 정확도 |
|---|---|---:|---:|---:|
| Task 2 기준 CNN (2회 실행) | 무작위 초기화, 증강 없음, 12 epoch | 77.0 / 77.1 | 73.5 / 73.2 | – |
| A 단순 CNN (3 seed) | ImageNet 사전학습 + 증강, 30 epoch | **94.22 ± 0.06** | **92.02 ± 0.09** | 86.27 ± 0.29 |
| B 제조사 계층 (soft) | 〃 | 94.07 ± 0.11 | 91.83 ± 0.21 | 85.66 ± 0.78 |
| C 계층 + 제조사 가중치 | 〃 | 94.04 ± 0.14 | 91.96 ± 0.10 | **87.08 ± 0.38** |

- **H1은 지지되지 않는다.** B − A의 Macro-F1은 −0.19%p로, 3개 seed 모두 음수다.
- **다른 제조사 오류는 남아 있었지만 줄지 않았다.** A에서도 검증 이미지의 3.24%(약 576장, 오답의 56%)가 다른 제조사 오류다. B는 이를 평균 약 8장만 줄였고 방향도 일관되지 않았다. 같은 제조사 오류는 3개 seed 모두 늘었다.
- **가중치의 효과(C − B):** 소수 제조사(BMW·GMC·VW) 차종 정확도 +1.42%p로, 3개 seed 모두 양수다. Macro-F1(+0.13%p)·Balanced accuracy(+0.23%p)도 B보다 일부 나아졌다. 그러나 기본 모델 A의 전체 성능은 넘지 못했다(C − A Macro-F1 −0.05%p, 0/3).
- **원인은 아직 입증하지 못했다.** 정보 중복, 형제차(Sierra↔Silverado)의 제조사 불확실성, 공동 학습의 부작용이 후보다. 기준 모델과 A는 사전학습 외에도 증강·학습 설정이 함께 달라, 사전학습의 효과는 통제 실험으로 따로 확인해야 한다.
- **다음 단계:** ① 오답 사진 검토(9회 모두 틀린 321장, 주요 혼동 쌍) → ② 사전학습 × 계층 2×2 비교(나머지 설정 고정) → ③ 혼동 기반·차체 유형 계층 → ④ 고해상도 입력.
- 결과 원본: `result_task2/`(기준 모델), `result_full/make_v1/`(본 실험 결과집과 증빙)

## 4. 앞으로 할 일

### 4-0. 정리 (바로)
- [x] 학습 코드(03)에 제출본 설정을 반영했다: **hard routing**, **Cui β** 가중 손실
- [ ] 혼동 기반 그룹을 val이 아니라 **학습 세트 내부 검증(또는 out-of-fold)**으로 만들도록 03의 6-2를 고친다. val은 모델 선택에만 쓴다.
- [ ] 01 노트북 Part C 문구를 제출본에 맞춘다 (soft conditioning → hard routing, `(1/n)^α` → Cui β).
- [ ] 다음 활동 공지(마감·제출물)를 확인하고 여기에 적는다.

### 4-1. 데이터 파이프라인
> 02 노트북으로 완료. 최종 **118,471장** = 119,032장 − 서로 다른 차종에 걸친 동일 사진 101장 − 같은 차종 안의 중복 사본 460장. train 82,929 / val 17,771 / test 17,771.

- [x] 제외 목록을 적용한 최종 이미지 목록을 만든다.
- [x] 계층화 **70/15/15** 분할. md5 중복 그룹은 한 분할에만 넣는다. 분할은 seed를 고정해 CSV로 저장하고 모든 실험이 공유한다.
- [x] 라벨 3종(차종 50 · 제조사 10 · 차체 유형 6)을 이미지마다 붙인다.
- [x] 전처리 (03 노트북의 transform)
  - 공통: 짧은 변 256px, 흑백은 RGB로 변환, 정규화
  - 학습: 랜덤 224 크롭, 좌우 반전, 약한 밝기·색 변화
  - 검증·테스트: 224 중앙 크롭
- [ ] 테스트 세트는 최종 평가 전까지 쓰지 않는다.

### 4-2. 기준 모델 (H0 검증)
- [ ] Flat 기준 모델(`flat_ce`, 사전학습 ResNet18)을 학습한다. 입력 224, 최대 12 epoch, val Macro-F1 기준 patience 4로 시작하고 결과를 보고 조정한다.
- [ ] seed 3개로 반복하고 Macro-F1의 평균±표준편차를 보고한다. (03의 7장이 자동으로 정리)
- [x] 평가 도구를 만들어 이후 모든 모델에 재사용한다. (03의 `compute_metrics`)
  - Top-1 · Top-3 정확도, **Macro-F1(주 지표)**, 클래스별 precision · recall · F1
  - 혼동 행렬(차체 유형 순 정렬), 학습·검증 곡선

### 4-3. 실패 분석
- [ ] 정확도 대 Macro-F1, 소수 클래스의 재현율을 본다 → H0 확인.
- [ ] 오답이 같은 차체 유형·형제차 안에 몰리는지 혼동 행렬로 확인한다.
- [ ] 틀린 사진을 직접 보고 원인(시점, 배경, 라벨 노이즈 등)을 분류한다.
- [ ] 학습 세트 내부 검증(또는 out-of-fold) 혼동 행렬을 군집화해 **혼동 기반 그룹**을 만든다. (03의 6-2, 현재는 val을 쓰므로 수정 필요)

### 4-4. 개선 실험
| 실험 | 비교 대상 | 확인할 것 |
|---|---|---|
| 계층형 분류 (전략 1) | Flat vs 제조사 계층(H1) vs 차체 유형 계층 vs 혼동 기반 계층(H2) | 1단계 정확도, 단계별 오류 전파, 형제차 성능 |
| 불균형 해소 (전략 2) | CE vs 역빈도 α vs class-balanced β vs focal vs Balanced Softmax vs 샘플링 vs DRW vs 사후 보정 vs cRT (03의 6-1) | tail 재현율, Macro-F1 |
| (선택) | 최선의 계층 + 가중 손실, ImageNet 사전학습, Grad-CAM | H2가 기각되면 CNN이 엠블럼을 보는지 확인 |

- 모든 모델은 같은 테스트 세트로 평가한다. seed 간 편차보다 일관되게 큰 개선만 인정한다.

### 4-5. 시연
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
6. Cao et al., *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss* (DRW), NeurIPS 2019
7. Kang et al., *Decoupling Representation and Classifier for Long-Tailed Recognition* (cRT, 샘플링), ICLR 2020
8. Ren et al., *Balanced Meta-Softmax for Long-Tailed Visual Recognition*, NeurIPS 2020
9. Menon et al., *Long-Tail Learning via Logit Adjustment*, ICLR 2021
