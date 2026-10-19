# 한서진 Task 2 사전 점검 결과

- 상태: `blocked_missing_data`
- 실제 학습은 실행하지 않았으며 정확도/F1 결과가 없습니다.
- 기존 분할 검증: True
- 클래스 수: 50
- 학습/검증/테스트: 82929 / 17771 / 17771장
- 테스트는 CSV 메타데이터만 점검했으며 이미지를 열거나 성능을 평가하지 않았습니다.
- 분할 CSV의 경로·MD5 교집합을 점검했습니다. 이미지 근접 중복은 검사하지 않았습니다.
- 실제 이미지 MD5 확인 수: 0

## 실행 상태

Windows C: drive is not mounted in this Linux environment. Run on Windows or pass --data-root with a mounted path.

Windows에서 `python task2_preliminary_seojin/train_baseline.py`를 실행하면
12 epoch 학습 후 results/baseline_seed0_seojin/에 실제 결과가 저장됩니다.
원본 이미지는 Git 저장소에 포함되어 있지 않습니다.
