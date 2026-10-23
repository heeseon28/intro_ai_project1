# 2차 실험 소스

[실행 계획](PLAN.md) · [결과 해설](../../reports/make_v2_results.md)

| 파일 | 역할 |
|---|---|
| stage0_diagnostics.py | 정답 제조사 추론·dHash 후보·접촉표 진단 |
| stage0_audit_report.py / stage0_finalize.py | 사진 검토 기록·진단 집계 |
| tensor_cache.py | 원본 uint8 캐시와 입력 동일성·속도 측정 |
| preprocessing.py | 비율 유지 letterbox와 단서 보존 증강 |
| run_workflow.py | 0.5~2단계 자동 선별·재개·학습 순서 |
| build_result_book.py | 12회 결과 재검산·표·그림·해시 생성 |
| ../../full_train.py | 공유 학습 루프. D는 flat+weighted 플래그로 분리 |

서버 저장소 루트에서 가상환경 Python으로 `python -m experiments.make_v2.run_workflow`를 실행한다. 현재 이미 완료되어 있으므로 상태 확인 목적으로 다시 실행하지 않는다. 루트 full_train.py의 단독 main 대신 이 진입점을 쓴다. test를 읽지 않는 V2 train/val 경로다. 데이터는 dataset/VMMRdb, 기존 체크포인트는 runs_full/make_v1에 필요하다. 환경 의존성은 저장소 requirements.txt를 따른다. 결과 sources/ 사본은 실행 당시 코드다.
