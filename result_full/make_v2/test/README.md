# 최종 테스트 결과집

[한국어 최종 보고](../../../reports/final_results.md) · [검증 결과집](../README.md)

검증으로 고정한 12개 체크포인트를 각 1회 테스트했다. 테스트 후 모델 재선정·튜닝·학습은 없었다. frozen_protocol.json이 평가 전 선택과 SHA-256을 기록한다.

| 모델 | 정확도 (%) | Macro-F1 (%) | 소수 제조사 평균 정확도 (%) |
|---|---:|---:|---:|
| A Flat | 94.480 ± 0.170 | 92.498 ± 0.222 | 86.743 ± 0.232 |
| B 제조사 계층 (soft) | 94.279 ± 0.073 | 92.429 ± 0.130 | 86.897 ± 0.357 |
| C 계층+제조사 가중치 | 94.071 ± 0.135 | 92.336 ± 0.223 | 87.839 ± 0.659 |
| D Flat+제조사 가중치 | 94.283 ± 0.150 | 92.404 ± 0.155 | 87.811 ± 0.400 |

지표 값은 per_run.csv에서 0~1, paired 표에서 %p다. 소수 제조사는 BMW·GMC·Volkswagen 정확도의 단순 평균이다.

| 파일 | 내용 |
|---|---|
| frozen_protocol.json | 평가 전에 고정한 체크포인트·설정·해시 |
| per_run.csv / mean_std.csv | 실행별 성능 / 평균·표본 표준편차 |
| paired_deltas_pp.csv / paired_mean_std_pp.csv | 같은 seed끼리 차이 / 평균·편차 |
| positive_delta_counts.csv / hypothesis_decisions.csv | 개선 seed 수 / 사전 운영 기준 판정 |
| per_class.csv / per_make.csv / confusion_pairs.csv | 클래스·제조사·혼동 쌍 분석 |
| validation_test_comparison.csv | 검증과 테스트 차이 |
| prediction_transitions.csv | 두 모델 사이 새 정답·새 오답 |
| evidence/ | 체크포인트 없는 저장 예측·혼동 행렬·지표 |
| test_index.csv / class_info.csv | 평가 행 순서와 클래스 매핑 |
| verification.csv / data_audit.json | 검산·분할 점검 |
| completion.json / SHA256SUMS.json | 완료 상태 / 파일 무결성 |
| sources/ | 최종 평가·집계 소스 사본 |

동일 분할 seed 3회다. 통계적 유의성이나 외부 데이터 일반화를 주장하지 않는다. 테스트 근접 중복은 점검하지 않았다. 원본 사진·체크포인트는 Git에 포함하지 않는다.
