# 2차 실험 전체 결과집 (검증 세트)

> 검증 결과 기록이다. 최종 테스트와 종료 판정은 [최종 보고](../../reports/final_results.md) 및 [테스트 결과집](test/README.md)을 본다. 아래 “test untouched” 문구는 테스트 이전 기록을 설명한다.

[한국어 결과 해설](../../reports/make_v2_results.md) · [프로젝트 안내](../../README.md) · [0단계 진단](stage0/README.md)

0~2단계 완료. A/B/C 9회는 make_v1 재사용, D 3회와 letterbox 선별 1회는 신규 실행이다. 최종 테스트는 미평가다. 성능 CSV는 0~1, 차이 표는 %p이며 모든 비교는 같은 seed를 짝지었다.

# make_v2 final validation result book

{
  "state": "complete",
  "selected_preprocessing": "center",
  "rule": "strictly greater validation Macro-F1 on seed 0; exploratory screening, no multi-seed effect claim",
  "old_macro_f1": 0.9211456922327025,
  "new_macro_f1": 0.9201531931141678,
  "delta_pp": -0.09924991185347176,
  "source": "runs_full/make_v2/seed0/flat",
  "cache_used": false
}

Validation only; test untouched. Same fixed split, 3 seeds. CSV metrics are fractions 0..1; paired deltas are percentage points.

 seed              experiment preprocessing  reused_v1  best_epoch  epochs_completed    minutes  accuracy  macro_f1  balanced_accuracy  top3_accuracy  minority_make_fine_accuracy  cross_make_error_rate  within_make_error_rate
    0                    flat        center       True          28                30 150.462065  0.942659  0.921146           0.918496       0.985763                     0.865982               0.032244                0.025097
    0               hierarchy        center       True          30                30 151.223412  0.941647  0.919793           0.915745       0.985088                     0.864216               0.032075                0.026279
    0 hierarchy_make_weighted        center       True          30                30 152.347778  0.941196  0.920493           0.917350       0.984300                     0.868961               0.031118                0.027686
    0      flat_make_weighted        center      False          26                30 151.912621  0.940746  0.920102           0.917890       0.984638                     0.867908               0.032919                0.026335
    1                    flat        center       True          28                30 150.253590  0.941590  0.919416           0.914295       0.985651                     0.861417               0.033088                0.025322
    1               hierarchy        center       True          28                30 150.862604  0.941084  0.919219           0.915001       0.986045                     0.856984               0.031681                0.027235
    1 hierarchy_make_weighted        center       True          30                30 151.655962  0.938833  0.918567           0.914917       0.985538                     0.868389               0.034269                0.026898
    1      flat_make_weighted        center      False          29                30 151.851717  0.942434  0.919602           0.916912       0.984919                     0.867467               0.032694                0.024872
    2                    flat        center       True          29                30 151.710345  0.942378  0.919940           0.916182       0.985988                     0.860686               0.031962                0.025660
    2               hierarchy        center       True          29                30 152.470658  0.939508  0.915873           0.910936       0.986213                     0.848709               0.032187                0.028305
    2 hierarchy_make_weighted        center       True          27                30 151.796847  0.941309  0.919808           0.916255       0.985707                     0.875161               0.031006                0.027686
    2      flat_make_weighted        center      False          29                30 151.842509  0.940690  0.920290           0.916359       0.983400                     0.866744               0.033031                0.026279

                comparison  mean_macro_f1_delta_pp  paired_sd_pp  improved_seeds  operational_effect  threshold_pp                                                       note
             B-A hierarchy               -0.187196      0.198635               0               False           0.2 operational consistency rule, not statistical significance
   C-B weight in hierarchy                0.132723      0.235741               2               False           0.2 operational consistency rule, not statistical significance
C-D hierarchy with weights               -0.037529      0.071925               1               False           0.2 operational consistency rule, not statistical significance
        D-A weight in flat               -0.016943      0.076197               2               False           0.2 operational consistency rule, not statistical significance
   V2 A-V1 A preprocessing                0.000000      0.000000               0               False           0.2 operational consistency rule, not statistical significance

B-A: hierarchy; D-A: weighting in Flat; C-B: weighting in hierarchy; C-D: hierarchy under matched weights. V2 A-V1 A is the preprocessing bundle comparison. If center selected, A/B/C are reused V1 and V2 A-V1 A is zero by design.
The letterbox intervention changes both training augmentation and validation geometry, so it does not isolate center cropping alone. Cache speed is reported separately and is not evidence for classification improvement.
The 0.2pp threshold with 3/3 favorable seeds was fixed before V2 training; this is an operational rule, not a statistical significance test.
Learning train scores use augmented data and changing weights during the epoch; they are not directly equivalent to evaluation-mode train scores. Weighted train loss differs from unweighted validation NLL.
Observed differences do not establish causes such as label noise, shared platform, pretraining redundancy or gradient conflict. See Stage0 visual screening limitations. No validation labels or images were removed.
Original data, cache, smoke and checkpoints stay under ignored runs_full. Evidence includes predictions/metrics only. No test performance is reported.