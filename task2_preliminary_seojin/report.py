"""Create a concise Korean report from measured train/validation results."""
import json


def write_report(out, summary, history, args, table):
    train = summary["train_at_best_checkpoint"]
    val = summary["validation_at_best_checkpoint"]
    audit = json.loads((out / "data_audit.json").read_text(encoding="utf-8"))
    gap = train["accuracy"] - val["accuracy"]
    if args.smoke:
        interpretation = "이 실행은 소량의 데이터로 입출력만 확인한 smoke 검사입니다. 과적합/과소적합이나 실제 성능을 판단할 수 없습니다."
    elif gap > .1:
        interpretation = "같은 체크포인트의 학습 정확도가 검증 정확도보다 10%p 이상 높아 과적합 가능성이 있습니다. 곡선의 검증 loss 추세와 함께 해석해야 합니다."
    elif train["accuracy"] < .5:
        interpretation = "학습 정확도 자체가 50% 미만이므로 학습 부족 또는 표현력 부족 가능성을 검토할 수 있습니다. 이 기준만으로 과소적합을 확정하지 않습니다."
    else:
        interpretation = "학습·검증 성능 차이와 loss 곡선을 함께 살펴 일반화 수준을 판단해야 합니다. 단일 실행의 최종 수치만으로 과적합/과소적합을 확정하지 않습니다."
    difficult = table.sort_values(["recall", "f1", "model_idx"], kind="stable").head(5)
    rows = [f"| {row.make_model} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {int(row.support)} |"
            for row in difficult.itertuples()]
    sizes = audit["split_sizes"]
    body = [
        "# Task 2: 기본 CNN 학습 및 평가 — 한서진", "",
        "팀명/학번: 제출 전 기입", "",
        "**실행 확인용이며 제출용 학습 결과가 아닙니다.**" if args.smoke else "기존 분할을 사용한 단일 seed 기준 실험 결과입니다.", "",
        "## 데이터셋과 실험 조건", "",
        f"VMMRdb의 기존 50개 make/model 클래스 분할(학습 {sizes['train']:,}장, 검증 {sizes['val']:,}장, 테스트 {sizes['test']:,}장)을 사용했습니다. "
        "테스트는 분할 메타데이터 중복 점검에만 사용했으며 성능은 측정하지 않았습니다. CompCar는 사용하지 않았습니다.", "",
        "ResNet-18의 최종 출력층을 50개 클래스로 구성하고 weights=None으로 무작위 초기화했습니다. "
        "학습과 검증 모두 RGB 변환, 고정 224×224 Resize, ToTensor, 채널별 mean/std=0.5 정규화만 적용했습니다. "
        f"Adam(lr=0.0003), 일반 CrossEntropyLoss, batch={args.batch_size}, seed=0, {len(history)} epoch로 학습했습니다. "
        "사전학습·증강·제조사 계층·클래스 가중치·특수 손실·학습률 스케줄은 사용하지 않았습니다.", "",
        f"README 기준 조건 충족: **{summary['readme_conditions_met']}**. Smoke 또는 batch 크기를 변경한 실행은 기본 실험과 구분합니다.", "",
        "## 학습 및 검증 결과", "",
        f"검증 macro-F1이 가장 높은 **epoch {summary['best_epoch']}**의 체크포인트를 선택했습니다. "
        "아래 학습·검증 지표는 모두 이 동일한 체크포인트로 다시 측정했습니다.", "",
        "| 분할 | Loss | Accuracy | Macro-F1 |", "|---|---:|---:|---:|",
        f"| 학습 | {train['loss']:.4f} | {train['accuracy']:.2%} | {train['macro_f1']:.4f} |",
        f"| 검증 | {val['loss']:.4f} | {val['accuracy']:.2%} | {val['macro_f1']:.4f} |", "",
        "![학습 곡선](learning_curves.png)", "",
        "곡선의 train 값은 가중치 업데이트 도중 누적한 값이므로 위 표의 고정 체크포인트 재평가와 다릅니다.", "",
        "## 혼동 행렬과 어려운 클래스", "",
        "![검증 혼동 행렬](confusion_matrix.png)", "",
        "혼동 행렬의 행은 실제 클래스, 열은 예측 클래스이며 각 행의 합을 1로 정규화했습니다. "
        "검증 recall이 낮은 5개 클래스는 다음과 같습니다.", "",
        "| 클래스 | Precision | Recall | F1 | 검증 표본 수 |", "|---|---:|---:|---:|---:|", *rows, "",
        "전체 클래스 지표는 per_class_metrics.csv, 오분류 및 정분류 예시는 validation_examples.png에 저장했습니다.", "",
        "## 해석과 다음 개선 방향", "",
        f"같은 체크포인트의 학습−검증 정확도 차이는 **{gap*100:.2f}%p**입니다. {interpretation}", "",
        "다음 실험에서는 기존 분할·전처리·optimizer·학습량을 통제하고 한 번에 하나의 개선 요인만 변경합니다. "
        "후보는 과적합을 줄이는 정규화 또는 데이터 증강이며, 이번 기준 실험에는 적용하지 않았습니다. "
        "여러 seed에서 반복하기 전에는 통계적으로 유의한 개선을 주장하지 않습니다.", "",
        "분할 간 동일 경로 및 저장된 MD5 중복을 점검했습니다. 실제 학습 이미지의 MD5는 최대 16장만 확인했고, "
        "전체 파일 해시 또는 근접 중복을 다시 검사하지 않았습니다. 낮은 정확도 자체가 실행 실패를 뜻하지 않습니다.", "",
        "제출 시 그림 크기를 조절하여 1–2쪽으로 편집하고 팀명/학번을 기입하세요.",
    ]
    (out / "report.md").write_text("\n".join(body) + "\n", encoding="utf-8")
