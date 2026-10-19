# Task 2 preliminary — 한서진

`../task2/README.md`의 실험 조건에 맞춘 독립 실행 코드입니다. 원본 `task2/`, 기존 조원의 결과, `splits/`는 변경하지 않습니다. 코드와 이번 실행의 산출물은 모두 이 폴더에 둡니다. 기존 분할 CSV는 `../splits/`에서 읽으므로 이 폴더만 복사하지 말고 저장소와 함께 사용하세요.

## 현재 저장된 결과

`results/preflight/`에 **실제 저장소 CSV에 대한 사전 점검 결과**를 저장했습니다.

- 클래스: 50개. 학습 82,929장 / 검증 17,771장 / 테스트 17,771장.
- 클래스 ID·이름·분할별 표본 수, 분할 간 경로 및 저장된 MD5 중복을 검사했습니다.
- 현재 Linux 환경에는 사용자의 Windows `C:` 드라이브와 원본 이미지가 연결되어 있지 않아 **12 epoch 학습은 실행하지 못했습니다. 정확도/F1·학습 곡선·모델은 아직 없습니다.**
- `results/verification/`에는 별도 합성 이미지로 코드의 입출력·체크포인트·결과 저장을 확인한 검사 로그만 저장합니다. 이 검사는 VMMRdb 실험 결과가 아닙니다.

실제 학습 결과는 아래 명령을 **VMMRdb가 있는 Windows PC**에서 실행하면 `results/baseline_seed0_seojin/`에 자동 저장됩니다. 다른 조원의 결과를 이번 실행 결과로 복사하지 않았습니다.

## 실험 조건

| 항목 | 설정 |
|---|---|
| 데이터 | VMMRdb 기존 50개 make/model 클래스 분할 유지 |
| 모델 | torchvision ResNet-18, `weights=None`, 출력 50개 |
| 전처리 | RGB → 고정 Resize(224, 224) → ToTensor → Normalize(mean=0.5, std=0.5) |
| 학습 | Adam(lr=0.0003), 일반 CrossEntropyLoss, batch 64, seed 0, 12 epoch |
| 제외 | CompCar, 사전학습, 증강, 제조사 계층, 클래스 가중치, 특수 손실, LR 스케줄 |
| 선택 | 검증 macro-F1이 가장 높은 checkpoint. 동점이면 먼저 나온 epoch |
| 평가 | 최적 checkpoint에서 학습·검증 재평가. 테스트 성능 측정 안 함 |

매 epoch loss/accuracy/macro-F1을 CSV와 그래프로 저장합니다. 학습 곡선의 train 값은 업데이트 중 누적한 값이고 최종 성능 표는 선택한 checkpoint를 고정하여 다시 측정한 값입니다.

## Windows에서 실행

다음 명령은 저장소 루트에서 실행합니다. 한글·공백 경로는 `dataset_paths.json`에 UTF-8로 저장되어 있습니다.

```powershell
python -m pip install -r task2_preliminary_seojin/requirements.txt

# 원본 이미지 존재 여부 및 분할 점검 (통과 시 종료 코드 0, 차단 시 2)
python task2_preliminary_seojin/preflight.py

# 소규모 실행 확인: 제출용 결과가 아님
python task2_preliminary_seojin/train_baseline.py --smoke

# 실제 12 epoch 학습
python task2_preliminary_seojin/train_baseline.py
```

GPU용 PyTorch 환경이 이미 있다면 해당 환경을 사용하세요. `--device auto`가 기본이며 CUDA, MPS, CPU 순으로 선택합니다. 전체 CPU 학습은 오래 걸릴 수 있습니다. Windows에서는 `workers=0`을 기본으로 사용합니다.

PowerShell 실행 스크립트도 제공합니다. 실행 정책 때문에 `.ps1` 실행이 제한된 PC에서는 위 Python 명령을 그대로 사용하면 됩니다.

```powershell
.\task2_preliminary_seojin\run_task2.ps1 -CheckOnly
.\task2_preliminary_seojin\run_task2.ps1 -Smoke -Device cuda
.\task2_preliminary_seojin\run_task2.ps1 -Device cuda
```

기본 데이터 경로는 사용자가 제공한 다음 경로입니다. `VMMRdb` 바로 아래에 `ford_mustang_1964` 같은 클래스 폴더들이 있어야 합니다.

```text
C:\Users\hseoj\OneDrive - Fraunhofer\Desktop\켄텍2 한서진\학부\4학년\4-1 (2026 2학기)\03_Introduction to AI\Project 1\VMMRdb
```

실제 폴더명이 다르거나 이 환경에 데이터를 연결한 경우 `dataset_paths.json`을 수정하거나 인자로 지정합니다. Linux/WSL에서는 해당 환경에서 접근할 수 있는 경로를 사용합니다.

```powershell
python task2_preliminary_seojin/train_baseline.py --data-root "D:\datasets\VMMRdb" --device cuda
```

## 중단 후 재개 및 실행 이름

```powershell
python task2_preliminary_seojin/train_baseline.py --resume
python task2_preliminary_seojin/train_baseline.py --run-name baseline_seed0_seojin_run2
```

결과 폴더가 이미 있으면 자동 덮어쓰기를 거부합니다. `--resume`은 같은 설정·분할·소스로 마지막 완료 epoch부터 재개합니다. 첫 epoch 전에 중단되었다면 처음부터 학습합니다. 체크포인트를 지우면 완료한 학습을 복원할 수 없습니다. smoke 실행 재개에는 `--smoke --resume`을 함께 사용합니다.

메모리 부족 시 `--batch-size 32`를 사용할 수 있지만 README의 batch 64 조건과 다릅니다. 변경은 config/metrics/report에 기록되며 `readme_conditions_met=false`가 됩니다. 로더 문제 시 `--workers 0`을 사용합니다. 모델·학습률·seed·해상도·12 epoch는 기본 실험에서 고정합니다.

## 실제 학습 완료 후 생성되는 파일

| 파일 | 내용 |
|---|---|
| `config.json`, `environment.json` | 학습 조건, 분할/소스 SHA256, 패키지·장치·데이터 경로·Git 리비전 |
| `data_audit.json`, `split_class_counts.csv` | 기존 분할 점검, 클래스별 표본 수, 실제 학습 이미지 최대 16장의 MD5 확인 |
| `history.csv`, `learning_curves.png` | epoch별 학습/검증 loss·accuracy·macro-F1 |
| `metrics.json` | 최적 checkpoint에서 측정한 학습/검증 성능 |
| `per_class_metrics.csv`, `difficult_classes.csv` | 클래스별 precision/recall/F1, 낮은 recall 상위 10개 |
| `confusion_matrix_counts.csv`, `confusion_matrix.png` | 검증 혼동 행렬 원시 개수 및 행 정규화 그림 |
| `validation_predictions.csv`, `validation_examples.png` | 모든 검증 예측, 오분류 최대 6개·정분류 최대 2개 예시 |
| `report.md`, `report_facts.md` | 측정 결과를 반영한 한국어 보고서 초안 및 사실 요약 |
| `train_source.py`, `preflight.py`, `report.py`, `class_info.csv` | 실행 소스 및 클래스 정보 사본 |
| `run.log`, `status.json` | 진행 로그와 실행 상태 |
| `checkpoints/best.pt`, `checkpoints/last.pt` | 최적 모델 및 마지막 완료 epoch의 재개 상태 |

제출 전 `status.json`의 `state=complete`, `metrics.json`의 `smoke_test_only=false`, `epochs_completed=12`, `readme_conditions_met=true`를 확인하세요. 보고서의 팀명·학번을 기입하고 그림 크기를 조절하여 1–2쪽으로 편집하세요. 체크포인트와 smoke 결과는 Git에서 제외합니다.

`data_audit.json`의 MD5 교집합 검사는 CSV에 저장된 해시 기준입니다. 실제 이미지 전체 해시와 근접 중복까지 재검사한 것은 아닙니다. 테스트 이미지의 존재/내용은 읽지 않습니다. 낮은 정확도 자체는 실패가 아니며 한 seed로 유의한 개선을 단정하지 않습니다.

## 코드 검사

```powershell
python -m unittest discover -s task2_preliminary_seojin/tests -v
```

검사 데이터는 임시 폴더의 합성 이미지입니다. 실제 분할 CSV는 읽기만 하며 새 실험 분할을 만들거나 덮어쓰지 않습니다. PowerShell 스크립트는 Linux에서 직접 실행 검증할 수 없으며, 동일한 Python 진입점을 사용합니다.

## Windows 데이터를 GPU 서버로 전송

Windows PC에 CUDA GPU가 없다면 [TRANSFER.md](TRANSFER.md)를 따라 필요한 학습·검증 이미지만 서버로 옮겨 학습할 수 있습니다. transfer_data.py는 100,700장만 묶고 전송 전후 MD5와 분할 일치 여부를 확인합니다.
