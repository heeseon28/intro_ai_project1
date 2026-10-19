# Task 2 예비 실험 보고서 — 결과 대기

작성자: 한서진 (팀 정보 추가 필요)

> **완료된 학습·검증 결과가 확인되지 않아 수치 성능을 보고하지 않는다.**

대상 실행: `baseline_seed0`

차단 사유: 필수 결과 미생성: metrics.json, history.csv, per_class_metrics.csv, confusion_matrix_counts.csv

확인 기록: `C:\Users\hseoj\OneDrive - Fraunhofer\Desktop\켄텍2 한서진\학부\4학년\4-1 (2026 2학기)\03_Introduction to AI\Project 1\intro_ai_project1\task2_preliminary_seojin\results\baseline_seed0\status.json`

```json
{
  "state": "running"
}
```

확인 기록: `C:\Users\hseoj\OneDrive - Fraunhofer\Desktop\켄텍2 한서진\학부\4학년\4-1 (2026 2학기)\03_Introduction to AI\Project 1\intro_ai_project1\task2_preliminary_seojin\results\baseline_seed0\data_audit.json`

```json
{
  "split_sizes": {
    "train": 82929,
    "val": 17771,
    "test": 17771
  },
  "test_images_opened": false
}
```

확인 기록: `C:\Users\hseoj\OneDrive - Fraunhofer\Desktop\켄텍2 한서진\학부\4학년\4-1 (2026 2학기)\03_Introduction to AI\Project 1\intro_ai_project1\task2_preliminary_seojin\results\preflight\status.json`

```json
{
  "state": "ready",
  "preflight_complete": true,
  "ready_for_training": true,
  "error_count": 0,
  "warning_count": 0
}
```

확인 기록: `C:\Users\hseoj\OneDrive - Fraunhofer\Desktop\켄텍2 한서진\학부\4학년\4-1 (2026 2학기)\03_Introduction to AI\Project 1\intro_ai_project1\task2_preliminary_seojin\results\preflight\data_audit.json`

```json
{
  "ready_for_training": true,
  "errors": [],
  "warnings": [],
  "split_sizes": {
    "train": 82929,
    "val": 17771,
    "test": 17771
  },
  "test_images_opened": false,
  "test_performance_measured": false
}
```

동일 실행 폴더의 status.json이 complete이고 metrics.json의 complete가 true인 뒤 보고서를 다시 생성한다. 정식 결과는 smoke_test_only=false이고 설정한 모든 epoch를 마쳐야 한다.

필요한 근거 파일: `config.json`, `metrics.json`, `history.csv`, `per_class_metrics.csv`, `confusion_matrix_counts.csv`.
