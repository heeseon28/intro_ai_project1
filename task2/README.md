# Task 2: 기본 CNN 학습 및 평가

Task 2 제출용 독립 코드다. 기존 pilot_train.py / run_project.sh와 별도로 실행한다.

> **최종 제출물:** `submission/Activity2_Submission.zip` (보고서 PDF, 이 코드, 분할 CSV, 두 실행 결과, 채점자용 영문 README).
> 보고서 원본은 `report/`. 보고서나 결과를 바꿨으면 `bash task2/submission/make_submission.sh`로 PDF와 zip을 다시 만든다.

## 실험 조건

- VMMRdb의 기존 50개 차종 분할을 그대로 사용한다. 분할 CSV를 다시 생성하지 않는다.
- ResNet-18, weights=None (무작위 초기화), RGB 224x224.
- 양쪽 분할 모두 고정 Resize, ToTensor, Normalize(mean=0.5, std=0.5)만 적용한다.
- Adam(lr=0.0003), 일반 CrossEntropyLoss, batch 64, seed 0, 고정 12 epoch.
- 사전학습, CompCars, 증강, 제조사 계층, 클래스 가중치, 특수 손실, 학습률 스케줄을 사용하지 않는다.
- 매 epoch 학습/검증 loss와 accuracy, macro-F1을 기록한다.
- 검증 Macro-F1이 가장 높은 체크포인트를 보고한다. 테스트 성능은 측정하지 않는다.
- 학습 곡선의 train 값은 업데이트 도중 누적한 수치다. 최종 train/val 성능은 같은 최적 체크포인트에서 다시 측정한다.
- 이후 개선 실험과 비교할 때는 이 분할과 전처리, optimizer, 학습량도 통제해야 한다. 기존 pilot은 조건이 달라 직접적인 공정 비교 결과로 사용하지 않는다.

## 1. 코드를 다른 조원에게 전달

아래 명령은 저장소 루트에서 실행한다. 기존 다른 변경은 함께 stage하지 않는다.

~~~bash
git add task2 result_task2/.gitignore result_task2/README.md
git diff --cached --stat
git commit -m "Add Task 2 random-initialized CNN baseline"
git push
~~~

조원 컴퓨터에서는 해당 브랜치를 pull한다. splits/train.csv, val.csv, test.csv, class_info.csv도 같은 리비전으로 있어야 한다.
데이터 이미지는 Git으로 전달하지 않으므로 조원 컴퓨터에도 VMMRdb 원본이 필요하다. CompCars는 이번 실험에 필요 없다.

## 2. 조원 컴퓨터에서 준비 및 작은 실행 검사

설치된 GPU용 PyTorch 환경을 사용한다. 없는 패키지는 현재 환경에 맞게 설치한다.

~~~bash
python3 -m pip install -r task2/requirements.txt
bash task2/run_task2.sh --data-root /실제/VMMRdb/경로 --smoke --device cuda --run-name baseline_seed0
~~~

Windows는 WSL에서 Bash 명령을 사용하거나 Python을 직접 실행한다.

~~~bash
python task2/train_baseline.py --data-root /실제/VMMRdb/경로 --smoke --run-name baseline_seed0
~~~

데이터 경로는 ford_mustang_1964 같은 클래스 폴더들이 바로 있는 VMMRdb 폴더다.
smoke 결과는 result_task2/baseline_seed0_smoke/에 저장된다. 제출용 성능으로 사용하지 않는다.
GPU가 없으면 --device cpu를 사용할 수 있으나 전체 실험은 오래 걸린다.
메모리가 부족하면 --batch-size 32, 데이터 로더 문제가 있으면 --workers 0을 사용하고 설정 변경을 기록한다.

## 3. 실제 Task 2 실험

~~~bash
bash task2/run_task2.sh --data-root /실제/VMMRdb/경로 --device cuda --run-name baseline_seed0
~~~

결과는 전부 저장소의 result_task2/baseline_seed0/ 안에 모인다. 명령 실행 위치와 무관하다.
Python 실행 파일은 PYTHON 환경변수로 지정할 수 있다.
모든 결과가 생성되고 status.json의 state가 complete, metrics.json의 smoke_test_only가 false인지 확인한다.
같은 실험을 중복 실행하지 않는다. 새 실험은 다른 --run-name을 사용한다.

중단 후 같은 설정으로 이어서 실행:

~~~bash
bash task2/run_task2.sh --data-root /실제/VMMRdb/경로 --device cuda --run-name baseline_seed0 --resume
~~~

마지막 완료 epoch에서 재개한다. 첫 epoch를 마치기 전에 끊기면 첫 epoch부터 다시 학습한다.
체크포인트는 그 컴퓨터에 남겨둬야 재개할 수 있다. Git으로는 체크포인트를 공유하지 않는다.

## 4. 결과만 커밋해서 전달

~~~bash
git add result_task2/baseline_seed0
git diff --cached --stat
git commit -m "Record Task 2 baseline training and validation results"
git push
~~~

result_task2/.gitignore가 체크포인트와 smoke 결과를 제외한다. 강제 추가(-f)는 하지 않는다.
데이터셋, 가상환경, 대용량 모델 파일을 포함하지 않는다.
보고서 작성 컴퓨터에서는 git pull 후 result_task2/baseline_seed0/를 확인하면 된다.

## 결과 파일과 보고서 작성

- config.json: 모델 및 학습 조건, 분할 CSV와 소스의 SHA256.
- environment.json: 실행 환경, device, 데이터 위치, Git 리비전.
- data_audit.json: 분할 크기와 검증 범위. 근접 중복은 별도 검토가 필요하다.
- history.csv / learning_curves.png: epoch별 학습/검증 곡선.
- metrics.json: 최적 체크포인트의 학습/검증 성능.
- per_class_metrics.csv / difficult_classes.csv: 차종별 지표와 어려운 차종.
- confusion_matrix_counts.csv / confusion_matrix.png: 검증 혼동 행렬.
- validation_predictions.csv / validation_examples.png: 성공/실패 예시.
- report_facts.md: 보고서에 참고할 사실과 해석 시 주의점.
- train_source.py / class_info.csv: 실행 소스와 클래스 목록 사본.
- run.log / status.json: 진행 기록 및 완료 여부.
- checkpoints/: 최적/재개용 모델 상태. 로컬에만 유지한다.

Task 2 보고서는 1-2쪽이며 팀 정보, 데이터셋, 구조/설정, 학습/검증 결과, 곡선,
혼동 행렬, 어려운 클래스, 과적합/과소적합 분석과 다음 개선 방향을 포함한다.
낮은 정확도 자체는 실패가 아니다. 한 seed의 결과로 유의한 개선을 단정하지 않는다.
