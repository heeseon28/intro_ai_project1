"""Generate an evidence-based Korean Task 2 report for one explicit run.

Usage: python task2_preliminary_seojin/report.py RUN_DIR [--output REPORT.md]
Only completed, internally consistent artifacts yield a performance report.
Otherwise a pending report is written and the process exits with code 2.
This utility uses the Python standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REQUIRED = (
    "config.json", "metrics.json", "history.csv",
    "per_class_metrics.csv", "confusion_matrix_counts.csv",
)
MEASURES = ("loss", "accuracy", "macro_f1", "balanced_accuracy")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}: JSON 객체가 필요합니다.")
    return value


def optional_json(path: Path) -> dict[str, Any]:
    try:
        return read_json(path) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value: Any, name: str, proportion: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0 or (proportion and result > 1):
        raise ValueError(f"유효하지 않은 지표: {name}")
    return result


def text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def validate_run(run: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED if not (run / name).is_file()]
    if missing:
        raise ValueError("필수 결과 미생성: " + ", ".join(missing))
    config = read_json(run / "config.json")
    metrics = read_json(run / "metrics.json")
    status = optional_json(run / "status.json")
    smoke = bool(metrics.get("smoke_test_only") or config.get("smoke"))
    if metrics.get("complete") is not True:
        raise ValueError("metrics.json에 complete=true가 기록되지 않았습니다.")
    # Training invokes this utility between writing final metrics and changing
    # status from running to complete. Artifact consistency is checked below.
    if status.get("state") not in (None, "running", "complete"):
        raise ValueError(f"실행 상태가 완료되지 않았습니다: {status.get('state')}")
    if metrics.get("test_evaluated") is not False:
        raise ValueError("Task 2의 테스트셋 미평가 조건을 확인할 수 없습니다.")
    history = read_csv(run / "history.csv")
    if not history or len(history) != int(metrics["epochs_completed"]):
        raise ValueError("history.csv와 완료 epoch 수가 일치하지 않습니다.")
    if int(metrics["epochs_completed"]) != int(config["epochs"]):
        raise ValueError("계획한 epoch가 모두 완료되지 않았습니다.")
    epochs = [int(row["epoch"]) for row in history]
    if epochs != list(range(1, len(history) + 1)):
        raise ValueError("학습 이력에 누락되거나 중복된 epoch가 있습니다.")
    for row in history:
        for split in ("train", "val"):
            for key in ("loss", "accuracy", "macro_f1"):
                number(row[f"{split}_{key}"], f"{split}_{key}", key != "loss")
    best_epoch = int(metrics["best_epoch"])
    chosen = next((row for row in history if int(row["epoch"]) == best_epoch), None)
    if chosen is None or not math.isclose(
        float(chosen["val_macro_f1"]), max(float(row["val_macro_f1"]) for row in history),
        rel_tol=1e-6, abs_tol=1e-8,
    ):
        raise ValueError("선택한 checkpoint가 검증 macro-F1 최고 epoch와 일치하지 않습니다.")
    for split in ("train_at_best_checkpoint", "validation_at_best_checkpoint"):
        for key in MEASURES:
            number(metrics[split][key], f"{split}.{key}", key != "loss")
    validation = metrics["validation_at_best_checkpoint"]
    for key in ("loss", "accuracy", "macro_f1"):
        if not math.isclose(float(chosen[f"val_{key}"]), float(validation[key]), rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"최적 epoch와 최종 검증 {key}가 일치하지 않습니다.")

    classes = read_csv(run / "per_class_metrics.csv")
    labels = [row["make_model"] for row in classes]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("클래스별 결과가 없거나 클래스 이름이 중복됩니다.")
    for row in classes:
        for key in ("precision", "recall", "f1"):
            number(row[key], f"{row['make_model']}.{key}", True)
        if int(row["support"]) <= 0:
            raise ValueError("검증 표본이 없는 클래스가 있습니다.")
    with (run / "confusion_matrix_counts.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    if header[1:] != labels or [row[0] for row in rows] != labels:
        raise ValueError("혼동 행렬과 클래스별 지표의 클래스 순서가 다릅니다.")
    matrix = [[int(value) for value in row[1:]] for row in rows]
    if any(len(row) != len(labels) or any(value < 0 for value in row) for row in matrix):
        raise ValueError("혼동 행렬의 크기 또는 카운트가 올바르지 않습니다.")
    for index, row in enumerate(classes):
        support = sum(matrix[index])
        if int(row["support"]) != support or not math.isclose(
            float(row["recall"]), matrix[index][index] / support, abs_tol=1e-6
        ):
            raise ValueError("클래스별 support/recall과 혼동 행렬이 일치하지 않습니다.")
    expected = {
        "accuracy": sum(matrix[i][i] for i in range(len(labels))) / sum(map(sum, matrix)),
        "macro_f1": sum(float(row["f1"]) for row in classes) / len(classes),
        "balanced_accuracy": sum(float(row["recall"]) for row in classes) / len(classes),
    }
    for key, value in expected.items():
        if not math.isclose(value, float(validation[key]), rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"검증 {key}와 클래스별/혼동 행렬 결과가 일치하지 않습니다.")
    return {"config": config, "metrics": metrics, "history": history,
            "classes": classes, "matrix": matrix, "smoke": smoke}


def interpretation(data: dict[str, Any]) -> str:
    metrics, history = data["metrics"], data["history"]
    train = metrics["train_at_best_checkpoint"]
    val = metrics["validation_at_best_checkpoint"]
    acc_gap = float(train["accuracy"]) - float(val["accuracy"])
    f1_gap = float(train["macro_f1"]) - float(val["macro_f1"])
    loss_gap = float(val["loss"]) - float(train["loss"])
    evidence = (f"동일 checkpoint의 train−val 정확도 차이는 {100 * acc_gap:+.2f}%p, "
                f"macro-F1 차이는 {100 * f1_gap:+.2f}%p이며, val−train loss 차이는 {loss_gap:+.4f}이다.")
    if data["smoke"]:
        return evidence + " 축소 표본과 제한된 배치로 측정했으므로 이 값으로 과적합·과소적합이나 실제 분류 성능을 판단하지 않는다."
    window = history[-min(3, len(history)):]
    diverging = len(window) >= 2 and (
        float(window[-1]["train_loss"]) < float(window[0]["train_loss"])
        and float(window[-1]["val_loss"]) > float(window[0]["val_loss"])
    )
    if acc_gap > 0 and loss_gap > 0:
        finding = "훈련 성능이 검증 성능보다 높아 일반화 차이가 관찰된다."
        if diverging:
            finding += (f" 마지막 {len(window)} epoch에서 train loss가 감소하고 val loss가 증가하여 "
                        "과적합 징후도 관찰된다.")
        else:
            finding += " 이 차이만으로 과적합의 원인을 확정할 수는 없다."
    else:
        finding = "동일 checkpoint의 두 지표에서 일관된 과적합 차이는 확인되지 않는다."
    chance = 1 / len(data["classes"])
    if (float(train["accuracy"]) <= 3 * chance and float(val["accuracy"]) <= 3 * chance
            and float(train["loss"]) >= 0.9 * math.log(len(data["classes"]))):
        finding += (f" train·val 정확도가 균등 무작위 기준({pct(chance)})의 3배 이내이고 "
                    "train loss도 높아 과소적합 또는 학습 실패 가능성이 있다. 추가 epoch와 학습 이력을 확인해야 한다.")
    else:
        finding += " 과소적합 여부는 훈련 성능과 loss의 수렴 추이를 함께 보고 판단해야 한다."
    return evidence + " " + finding


def render_complete(run: Path, data: dict[str, Any]) -> str:
    config, metrics = data["config"], data["metrics"]
    audit = optional_json(run / "data_audit.json")
    environment = optional_json(run / "environment.json")
    train = metrics["train_at_best_checkpoint"]
    val = metrics["validation_at_best_checkpoint"]
    classes = data["classes"]
    lines = ["# Task 2 예비 실험 보고서", "", "작성자: 한서진 (팀 정보 추가 필요)", ""]
    if data["smoke"]:
        lines += ["> **파이프라인 시험(SMOKE): 최종 성능 결과로 제출하거나 전체 실험과 비교할 수 없음.**",
                  "> 일부 학습 배치와 축소된 검증 표본으로 입출력·학습·산출물 생성만 확인했다.", ""]
    lines += [f"실행: `{text(run.name)}` · 완료 epoch: {metrics['epochs_completed']} · "
              f"검증 macro-F1 최고 checkpoint: epoch {metrics['best_epoch']}", "",
              "## 데이터와 기본 모델", "", f"VMMRdb의 기존 {len(classes)}개 차종 분할을 사용하는 Task 2 기준선이다. "
              "CompCars는 이번 기준선 학습에 사용하지 않는다."]
    if environment.get("data_root"):
        lines.append(f"원본 경로: `{text(environment['data_root'])}`.")
    sizes = audit.get("split_sizes", {})
    if sizes:
        lines.append("분할 명세 표본 수: " + ", ".join(f"{name} {sizes[name]:,}" for name in ("train", "val", "test") if name in sizes)
                     + f"; 클래스 {len(classes)}개. 테스트 성능은 측정하지 않았다.")
    else:
        lines.append(f"검증 결과의 클래스 수는 {len(classes)}개이며 테스트 성능은 측정하지 않았다. "
                     "분할 크기는 data_audit.json에서 확인되지 않았다.")
    if data["smoke"]:
        lines.append("위 분할 명세 수는 전체 분할 기준이다. 본 시험의 실제 검증 표본 수는 "
                     f"{sum(int(row['support']) for row in classes):,}개이며, train 지표도 제한된 배치에서 계산했다.")
    if audit.get("cross_split_stored_md5_overlap") is False:
        lines.append("저장된 MD5 기준 분할 간 중복은 없으며, 근접 중복 검증 범위는 별도로 확인해야 한다.")
    lines += ["", f"구조: {text(config.get('architecture', '확인 필요'))}; "
              f"초기화: {text(config.get('initialization', '확인 필요'))}; RGB {config.get('image_size', '?')}×{config.get('image_size', '?')}.",
              f"전처리: {text(config.get('resize', '확인 필요'))}, ToTensor, "
              f"Normalize {text(config.get('normalization', '확인 필요'))}.",
              f"학습: {text(config.get('optimizer', '확인 필요'))}, lr={config.get('learning_rate', '?')}, "
              f"{text(config.get('loss', '확인 필요'))}, batch={config.get('batch_size', '?')}, "
              f"epoch={config.get('epochs', '?')}, seed={config.get('seed', '?')}.",
              "설정 기록: " + ", ".join(f"{label}={text(config.get(key, '확인 필요'))}" for key, label in
                  (("transfer_learning", "전이학습"), ("augmentation", "증강"), ("rebalancing", "재균형"),
                   ("hierarchy", "계층학습"), ("scheduler", "학습률 스케줄"))) + ".",
              "", "## 학습·검증 결과", "", "검증 macro-F1으로 선택한 동일한 고정 checkpoint를 재평가한 수치이다.", "",
              "| 분할 | Loss | Accuracy | Macro-F1 | Balanced accuracy |",
              "|---|---:|---:|---:|---:|"]
    for name, values in (("Train", train), ("Validation", val)):
        lines.append(f"| {name} | {float(values['loss']):.4f} | {pct(float(values['accuracy']))} | "
                     f"{float(values['macro_f1']):.4f} | {pct(float(values['balanced_accuracy']))} |")
    lines += ["", interpretation(data), "", "학습 곡선의 train 값은 가중치 업데이트 중 누적한 값이므로 "
              "위 고정 checkpoint 평가와 직접 일치하지 않을 수 있다.", "",
              "## 어려운 차종과 주요 혼동", "", "검증 recall이 낮은 순서이며 동률에서는 F1, 차종명 순으로 정렬했다.", "",
              "| 차종 | Recall | F1 | 검증 표본 |", "|---|---:|---:|---:|"]
    hardest = sorted(classes, key=lambda row: (float(row["recall"]), float(row["f1"]), row["make_model"]))[:10]
    for row in hardest:
        lines.append(f"| {text(row['make_model'])} | {pct(float(row['recall']))} | {float(row['f1']):.4f} | {row['support']} |")
    pairs = sorted(((value, i, j) for i, row in enumerate(data["matrix"]) for j, value in enumerate(row)
                    if i != j and value > 0), key=lambda pair: (-pair[0], pair[1], pair[2]))[:5]
    if pairs:
        lines += ["", "검증 혼동 건수 상위 5개 방향(실제 → 예측):", "",
                  "| 실제 차종 | 예측 차종 | 건수 | 실제 클래스 내 비율 |", "|---|---|---:|---:|"]
        for count, i, j in pairs:
            lines.append(f"| {text(classes[i]['make_model'])} | {text(classes[j]['make_model'])} | {count} | "
                         f"{pct(count / sum(data['matrix'][i]))} |")
    else:
        lines += ["", "검증 혼동 행렬에 비대각 오분류가 없다."]
    lines += ["", "## 근거 그림과 다음 실험", ""]
    for filename, caption in (("learning_curves.png", "epoch별 학습·검증 곡선"),
                              ("confusion_matrix.png", "검증 혼동 행렬(행 정규화)"),
                              ("validation_examples.png", "검증 성공·실패 예시")):
        if (run / filename).is_file():
            lines.append(f"- [{caption}]({filename})")
        else:
            lines.append(f"- {caption}: `{filename}` 미생성.")
    lines += ["", "혼동 쌍의 예시 이미지를 확인해 시점·배경·유사 외형의 영향을 검토한다. "
              "이러한 원인은 혼동 횟수만으로 단정할 수 없다. 다음 실험은 동일한 분할과 기준 설정을 유지하고 "
              "하나의 개선 요소씩 비교한다. 단일 seed 결과로 통계적 유의성을 주장하지 않는다.", "",
              "재현 근거: [설정](config.json), [최종 지표](metrics.json), [학습 이력](history.csv), "
              "[클래스별 지표](per_class_metrics.csv), [혼동 건수](confusion_matrix_counts.csv)."]
    return "\n".join(lines) + "\n"


def render_pending(run: Path, reason: str) -> str:
    lines = ["# Task 2 예비 실험 보고서 — 결과 대기", "", "작성자: 한서진 (팀 정보 추가 필요)", "",
             "> **완료된 학습·검증 결과가 확인되지 않아 수치 성능을 보고하지 않는다.**", "",
             f"대상 실행: `{text(run.name)}`", "", f"차단 사유: {text(reason)}", ""]
    candidates = [run / "status.json", run / "preflight.json", run / "data_audit.json",
                  run / "preflight" / "status.json", run / "preflight" / "data_audit.json",
                  run.parent / "preflight" / "status.json", run.parent / "preflight" / "data_audit.json"]
    candidates.extend(sorted(run.glob("*preflight*.json")))
    if run.name != "results":
        candidates.extend(sorted((run.parent).glob("*preflight*.json")))
    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        info = optional_json(path)
        if info:
            selected = {key: info[key] for key in (
                "state", "error", "preflight_complete", "ready_for_training", "errors", "warnings",
                "error_count", "warning_count", "split_sizes", "test_images_opened", "test_performance_measured",
            ) if key in info}
            for key in ("errors", "warnings"):
                if isinstance(selected.get(key), list) and len(selected[key]) > 10:
                    selected[key] = selected[key][:10] + ["이하 생략: 전체 내용은 원본 JSON 참조"]
            if selected:
                lines += [f"확인 기록: `{text(path)}`", "", "```json",
                          json.dumps(selected, ensure_ascii=False, indent=2), "```", ""]
    lines += ["동일 실행 폴더의 status.json이 complete이고 metrics.json의 complete가 true인 뒤 "
              "보고서를 다시 생성한다. 정식 결과는 smoke_test_only=false이고 설정한 모든 epoch를 마쳐야 한다.", "",
              "필요한 근거 파일: " + ", ".join(f"`{name}`" for name in REQUIRED) + "."]
    return "\n".join(lines) + "\n"


def generate_report(run_dir: str | Path, output: str | Path | None = None) -> Path:
    """Write a report for exactly this run and return its path.

    Missing/inconsistent evidence produces a clearly marked pending report,
    never a fabricated performance report. The CLI returns code 2 for it.
    """
    run = Path(run_dir).expanduser().resolve()
    if not run.is_dir():
        raise ValueError(f"실행 폴더가 없습니다: {run}")
    target = Path(output).expanduser().resolve() if output else run / "report.md"
    try:
        report = render_complete(run, validate_run(run))
    except (OSError, ValueError, KeyError, TypeError, StopIteration, ZeroDivisionError, csv.Error) as error:
        report = render_pending(run, str(error))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(report, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="보고할 실행 폴더. 최신 실행을 자동 선택하지 않습니다.")
    parser.add_argument("--output", type=Path, help="기본값: RUN_DIR/report.md")
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    if not run.is_dir():
        parser.error(f"실행 폴더가 없습니다: {run}")
    output = generate_report(run, args.output)
    code = 2 if output.read_text(encoding="utf-8").startswith("# Task 2 예비 실험 보고서 — 결과 대기") else 0
    print(f"{'COMPLETE' if code == 0 else 'PENDING'} report: {output}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
