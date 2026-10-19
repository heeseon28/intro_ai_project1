# Task 2 — 한서진 예비 실험

`../task2/README.md` 조건으로 만든 독립 실행 폴더다. 코드는 `../task2/train_baseline.py`에서 출발했으며 Windows 경로, UTF-8 출력, CPU 스레드 설정, 자동 보고서 및 결과 검증을 추가했다. 원본 `task2/`와 기존 다른 조원의 실험 결과는 수정하지 않는다.

## 저장된 실행

| 위치 | 내용 | 성능 해석 |
|---|---|---|
| [results/preflight/preflight_report.md](results/preflight/preflight_report.md) | 기존 분할과 로컬 데이터 검사 완료 | 학습 성능이 아닌 준비 검사 |
| [results/pipeline_check_smoke/report.md](results/pipeline_check_smoke/report.md) | 작은 학습·검증, 그림·보고서·지표 검증 완료 | 정식 성능으로 사용 금지 |
| [results/baseline_seed0/](results/baseline_seed0/) | 정식 12 epoch 실행 및 산출물 저장 위치 | 완료 여부는 `status.json` 확인 |

정식 실험은 CPU로 시작했다. 콘솔 창을 닫아도 백그라운드 프로세스가 계속 실행하며, 컴퓨터가 절전·종료되면 진행할 수 없다. 모든 epoch가 끝나면 `report.md`와 `verification.json`까지 자동으로 저장한다. **`status.json`의 `state`가 `complete`, `metrics.json`의 `smoke_test_only`가 `false`, `verification.json`의 `passed`가 `true`일 때 정식 결과가 완성된다.** 실행 중에는 완료된 전체 성능이 없다.

## 데이터와 실험 조건

기본 데이터 위치는 저장소 바로 위의 `VMMRdb/`다. 사용자가 제공한 로컬 절대 경로는 [dataset_paths.json](dataset_paths.json)에 기록했다. `CompCar/` 경로도 기록하되 이번 Task 2 학습에는 사용하지 않는다.

- 기존 `../splits/{train,val,test,class_info}.csv`를 재사용하며 분할을 생성하지 않는다.
- 50개 클래스, train 82,929장, validation 17,771장, test 17,771장. 테스트는 분할 메타데이터만 검사하며 이미지와 성능은 평가하지 않는다.
- ResNet-18 `weights=None`, RGB 224×224, 양쪽 모두 고정 `Resize → ToTensor → Normalize(mean=0.5, std=0.5)`.
- Adam(lr=0.0003), 일반 CrossEntropyLoss, batch 64, seed 0, 고정 12 epoch.
- 사전학습, 증강, 계층 분류, 클래스 가중치, 특수 손실, 스케줄, 조기 종료 없음.
- epoch마다 train/validation loss·accuracy·macro-F1 기록. validation macro-F1이 최고인 첫 checkpoint를 선택하여 동일한 모델로 전체 train/validation 재평가.

정식 실행은 위 조건을 검사한다. 메모리 부족 시 README가 허용한 batch 32를 Python의 `--batch-size 32`로 선택할 수 있고 설정에 변경을 기록한다. CPU threads와 loader workers는 실행 환경 설정이며 기본값은 각각 4, 0이다.

## Windows에서 실행

아래 명령은 저장소 루트에서 실행한다. 현재 PC의 PowerShell은 `.ps1` 실행을 제한하므로 **Python 직접 실행을 권장한다.** 아래의 Python 경로는 실제 학습에 사용한 환경이다. 다른 환경에서는 실행 파일 경로만 바꾼다.

```powershell
# 데이터 준비 검사
.\.venv-project-runner\Scripts\python.exe task2_preliminary_seojin/preflight.py

# 작은 전체 파이프라인 검사: 새 이름을 사용한다.
.\.venv-project-runner\Scripts\python.exe task2_preliminary_seojin/train_baseline.py --smoke --run-name smoke_new

# 정식 학습: 이미 실행 중인 baseline_seed0를 중복 실행하지 않는다.
.\.venv-project-runner\Scripts\python.exe task2_preliminary_seojin/launch_training.py --run-name baseline_new

# 중단 후 마지막 완료 epoch부터 재개 (기존 프로세스가 종료된 경우)
.\.venv-project-runner\Scripts\python.exe task2_preliminary_seojin/launch_training.py --run-name baseline_seed0 --resume
```

학습 환경 설치가 필요하면 해당 Python에서 `python -m pip install -r task2_preliminary_seojin/requirements.txt`를 사용한다. GPU용 PyTorch가 이미 설치된 환경은 그대로 사용한다.

스크립트 실행이 허용된 환경에서는 `run_task2.ps1 -Mode full|smoke|preflight`도 사용할 수 있다. `-Python`, `-DataRoot`, `-Device`, `-RunName`, `-Threads`, `-Workers`, `-Resume`를 지원하고, 정식 실행의 기존 백그라운드 PID를 검사한다. 기본 Python은 저장소의 `.venv-project-runner`이며 없으면 `python`을 사용한다.

백그라운드 실행과 진행 확인:

```powershell
# 현재 실행이 종료된 뒤 재개할 때만 사용
.\.venv-project-runner\Scripts\python.exe task2_preliminary_seojin/launch_training.py --resume

Get-Content task2_preliminary_seojin/results/baseline_seed0/status.json
Get-Content task2_preliminary_seojin/results/baseline_seed0/run.log -Tail 10
```

`launch_training.py`는 숨김 프로세스로 실행하고 PID·명령을 `process.json`에 기록한다. 같은 실행의 프로세스가 살아 있으면 중복 실행을 거부한다. `--device cuda`, `--threads`, `--workers`, `--data-root`, `--run-name`도 지원한다. 재개에는 원래의 checkpoint와 동일한 학습 소스·설정이 필요하다. 첫 epoch 도중 중단되면 첫 epoch부터 다시 학습한다.

## 결과 파일

각 실행의 `results/<run_name>/`에 다음을 모은다. smoke 이름에는 `_smoke`를 붙여 정식 실행과 구분한다.

| 파일 | 내용 |
|---|---|
| `config.json`, `environment.json`, `data_audit.json` | 설정, 소스·분할 SHA256, 장치·경로, 검사 범위 |
| `history.csv`, `learning_curves.png` | epoch별 loss·accuracy·macro-F1 및 세 곡선 |
| `metrics.json` | 최적 checkpoint의 train/validation 성능 |
| `per_class_metrics.csv`, `difficult_classes.csv` | 클래스별 precision·recall·F1·support와 어려운 10개 차종 |
| `confusion_matrix_counts.csv`, `confusion_matrix.png` | validation 혼동 행렬 원시 개수와 행 정규화 그림 |
| `validation_predictions.csv`, `validation_examples.png` | validation 전체 예측과 실패·성공 표본 |
| `report_facts.md`, `report.md` | 결과 사실과 한국어 분석 초안 |
| `class_info.csv`, `train_source.py` | 실제 실행 클래스 목록·학습 소스 사본 |
| `status.json`, `run.log`, `process.json` | 완료 여부, 진행 로그, 백그라운드 PID |
| `verification.json` | 저장된 예측·혼동 행렬·지표·소스·그림 간 일관성 검사 |
| `checkpoints/best.pt`, `checkpoints/last.pt` | 최적 모델 및 재개 상태, 로컬에만 유지 |

보고서 재생성 및 결과 재검증:

```powershell
.\.venv-project-runner\Scripts\python.exe task2_preliminary_seojin/report.py task2_preliminary_seojin/results/baseline_seed0
.\.venv-project-runner\Scripts\python.exe task2_preliminary_seojin/verify_results.py task2_preliminary_seojin/results/baseline_seed0
```

준비 검사에서는 전체 train/validation 파일 존재와 분할 메타데이터, 표본 train 16장의 MD5·디코딩을 확인했다. 모든 이미지의 해시와 근접 중복은 확인하지 않았다. 그림의 train 곡선은 업데이트 중 집계 값이고, 최종 train/validation 지표는 같은 고정 checkpoint에서 측정한다. 한 seed 결과로 개선의 통계적 유의성을 판단하지 않는다. 보고서의 팀 정보는 제출 전에 추가해야 한다.

체크포인트·프로세스 중복 로그·Python 캐시는 이 폴더의 `.gitignore`로 제외한다. 커밋·푸시는 자동으로 수행하지 않는다.
