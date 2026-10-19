"""Check saved Task 2 artifacts without evaluating images or loading a model.

Usage: python task2_preliminary_seojin/verify_results.py RESULTS_DIRECTORY
The JSON result is saved as verification.json. A smoke run can pass artifact
verification while remaining unsuitable as the twelve-epoch experiment result.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def verify_run(run_dir):
    """Return and save a verification report; failures are reported, not raised.

    This verifies consistency of recorded artifacts. It cannot establish that
    predictions were produced by the checkpoint without another model pass.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    checks = []
    recomputed = {}

    def check(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": str(detail)})

    def section(name, function):
        try:
            function()
        except Exception as exc:
            check(name, False, f"{type(exc).__name__}: {exc}")

    def close(a, b, *, atol=1e-12):
        return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                                rtol=1e-10, atol=atol, equal_nan=False))

    def json_file(name):
        try:
            value = json.loads((run_dir / name).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON root must be an object")
            check(f"read {name}", True)
            return value
        except Exception as exc:
            check(f"read {name}", False, f"{type(exc).__name__}: {exc}")
            return {}

    def csv_file(name, **kwargs):
        try:
            value = pd.read_csv(run_dir / name, encoding="utf-8", **kwargs)
            check(f"read {name}", True)
            return value
        except Exception as exc:
            check(f"read {name}", False, f"{type(exc).__name__}: {exc}")
            return pd.DataFrame()

    config = json_file("config.json")
    status = json_file("status.json")
    metrics = json_file("metrics.json")
    audit = json_file("data_audit.json")
    smoke = config.get("smoke")
    check("smoke flag is boolean", type(smoke) is bool)
    check("run status is complete", status.get("state") == "complete")
    check("metrics are complete", metrics.get("complete") is True)
    check("smoke flags agree", type(smoke) is bool
          and metrics.get("smoke_test_only") is smoke and status.get("smoke") is smoke)
    check("test was not evaluated", metrics.get("test_evaluated") is False)
    check("test images were not opened", audit.get("test_images_opened") is False)
    check("stored MD5 has no cross-split overlap", audit.get("cross_split_stored_md5_overlap") is False)
    expected_epochs = 1 if smoke is True else 12
    check("configured epoch count", config.get("epochs") == expected_epochs)
    check("completed epoch count", metrics.get("epochs_completed") == expected_epochs)
    check("checkpoint selection rule", metrics.get("selected_by") == "maximum validation macro-F1")

    def contract():
        required = {
            "architecture": "torchvision ResNet-18", "initialization": "random, weights=None",
            "image_size": 224, "learning_rate": 0.0003, "optimizer": "Adam",
            "loss": "unweighted CrossEntropyLoss", "seed": 0,
            "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3},
            "resize": "deterministic square resize", "augmentation": False,
            "transfer_learning": False, "rebalancing": False, "hierarchy": False,
            "scheduler": False, "precision": "float32",
        }
        for key, expected in required.items():
            check(f"baseline setting {key}", config.get(key) == expected,
                  f"recorded={config.get(key)!r}; required={expected!r}")
        batch = config.get("batch_size")
        check("baseline batch size", batch in (32, 64))
        if batch == 32:
            documented = any(isinstance(config.get(key), str) and config[key].strip()
                             for key in ("batch_size_reason", "batch_size_note"))
            for filename in ("report_facts.md", "run.log"):
                path = run_dir / filename
                if path.is_file():
                    documented |= any("32" in line and re.search(r"memory|메모리|fallback", line, re.I)
                                      for line in path.read_text(encoding="utf-8").splitlines())
            check("batch 32 memory fallback is documented", documented)

    if smoke is False:
        section("baseline contract", contract)

    def provenance():
        source = run_dir / "train_source.py"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        check("saved training source SHA256", config.get("source_sha256") == digest, digest)
        hashes = config.get("split_sha256", {})
        for filename in ("train.csv", "val.csv", "test.csv", "class_info.csv"):
            digest = hashes.get(filename) if isinstance(hashes, dict) else None
            check(f"split SHA256 recorded: {filename}",
                  isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None)

    section("source and split provenance", provenance)
    ci = csv_file("class_info.csv")
    predictions = csv_file("validation_predictions.csv")
    table = csv_file("per_class_metrics.csv")
    cm_saved = csv_file("confusion_matrix_counts.csv", index_col=0)
    history = csv_file("history.csv")

    def prediction_metrics():
        nclasses = 50
        check("exactly 50 class indices", len(ci) == nclasses
              and ci["model_idx"].tolist() == list(range(nclasses)))
        check("class names are unique", ci["make_model"].notna().all() and ci["make_model"].is_unique)
        check("audit class count", audit.get("nclasses") == nclasses)
        if len(ci) != nclasses:
            raise ValueError("Expected 50 classes before checking prediction labels")
        labels = ci["make_model"].to_numpy()
        expected_n = 50 if smoke is True else 17771
        check("prediction row count", len(predictions) == expected_n,
              f"actual={len(predictions)}, expected={expected_n}")
        if smoke is False:
            check("validation audit count", audit.get("split_sizes", {}).get("val") == expected_n)
        check("prediction paths are nonempty and unique", predictions["rel_path"].notna().all()
              and predictions["rel_path"].is_unique)
        for name in ("model_idx", "prediction_idx"):
            values = pd.to_numeric(predictions[name], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(values).all() and ((values >= 0) & (values < nclasses)).all()
            valid = valid and (values == np.floor(values)).all()
            check(f"valid prediction indices: {name}", valid)
            if not valid:
                raise ValueError(f"Invalid integer class index in {name}")
        y = predictions["model_idx"].to_numpy(dtype=int)
        p = predictions["prediction_idx"].to_numpy(dtype=int)
        check("true class labels match indices", np.array_equal(predictions["make_model"].to_numpy(), labels[y]))
        check("predicted labels match indices", np.array_equal(predictions["prediction_label"].to_numpy(), labels[p]))
        saved_correct = predictions["correct"].astype(str).str.lower()
        check("saved correctness flags", saved_correct.isin(["true", "false"]).all()
              and np.array_equal(saved_correct.eq("true").to_numpy(), y == p))
        cm = np.zeros((nclasses, nclasses), dtype=np.int64)
        np.add.at(cm, (y, p), 1)
        support = cm.sum(axis=1)
        predicted = cm.sum(axis=0)
        true_positive = np.diag(cm).astype(float)
        precision = np.divide(true_positive, predicted, out=np.zeros(nclasses), where=predicted != 0)
        recall = np.divide(true_positive, support, out=np.zeros(nclasses), where=support != 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
                       out=np.zeros(nclasses), where=(precision + recall) != 0)
        check("all validation classes represented", (support > 0).all())
        if smoke is True:
            check("smoke contains one example per class", np.array_equal(support, np.ones(nclasses)))
        recomputed.update({"samples": int(len(y)), "accuracy": float(true_positive.sum() / len(y)),
                           "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean())})
        vm = metrics.get("validation_at_best_checkpoint", {})
        for name in ("accuracy", "macro_f1", "balanced_accuracy"):
            check(f"recomputed validation {name}", close(vm.get(name), recomputed[name]),
                  f"recorded={vm.get(name)!r}; recomputed={recomputed[name]}")
        check("per-class row indices", table["model_idx"].tolist() == list(range(nclasses)))
        check("per-class names", np.array_equal(table["make_model"].to_numpy(), labels))
        for name, values in (("precision", precision), ("recall", recall), ("f1", f1)):
            check(f"recomputed per-class {name}", len(table) == nclasses and close(table[name], values))
        check("recomputed per-class support", np.array_equal(table["support"].to_numpy(), support))
        check("confusion matrix dimensions", cm_saved.shape == (nclasses, nclasses))
        check("confusion matrix row labels", cm_saved.index.tolist() == labels.tolist())
        check("confusion matrix column labels", cm_saved.columns.tolist() == labels.tolist())
        check("recomputed full confusion matrix", cm_saved.shape == cm.shape
              and np.array_equal(cm_saved.to_numpy(), cm))

    section("prediction and metric consistency", prediction_metrics)

    def epoch_consistency():
        check("history epoch sequence", history["epoch"].tolist() == list(range(1, expected_epochs + 1)))
        for phase in ("train", "val"):
            for metric in ("loss", "accuracy", "macro_f1", "balanced_accuracy"):
                values = history[f"{phase}_{metric}"].to_numpy(dtype=float)
                valid = np.isfinite(values).all() and (values >= 0).all()
                if metric != "loss":
                    valid = valid and (values <= 1).all()
                check(f"finite valid history {phase}_{metric}", valid)
        # np.argmax returns the first maximum, matching the strict '>' save rule.
        best = history.iloc[int(np.argmax(history["val_macro_f1"].to_numpy(dtype=float)))]
        check("earliest maximum macro-F1 selected", metrics.get("best_epoch") == int(best["epoch"]),
              f"recorded={metrics.get('best_epoch')}; expected={int(best['epoch'])}")
        for metric in ("loss", "accuracy", "macro_f1", "balanced_accuracy"):
            frozen = metrics.get("validation_at_best_checkpoint", {}).get(metric)
            check(f"frozen validation matches selected epoch: {metric}",
                  close(frozen, best[f"val_{metric}"], atol=1e-7 if metric == "loss" else 1e-12))
            train_value = metrics.get("train_at_best_checkpoint", {}).get(metric)
            valid = isinstance(train_value, (int, float)) and np.isfinite(train_value) and train_value >= 0
            if metric != "loss":
                valid = valid and train_value <= 1
            check(f"finite frozen training {metric}", valid)

    section("checkpoint selection and history consistency", epoch_consistency)
    for filename in ("learning_curves.png", "confusion_matrix.png", "validation_examples.png"):
        def verify_image(name=filename):
            with Image.open(run_dir / name) as picture:
                check(f"PNG format: {name}", picture.format == "PNG")
                check(f"positive image dimensions: {name}", min(picture.size) > 0)
                picture.verify()
            check(f"readable image: {name}", True)
        section(f"readable image: {filename}", verify_image)
    for filename in ("checkpoints/best.pt", "checkpoints/last.pt", "report_facts.md", "report.md", "run.log"):
        path = run_dir / filename
        check(f"nonempty artifact: {filename}", path.is_file() and path.stat().st_size > 0)
    def verify_report():
        report = (run_dir / 'report.md').read_text(encoding='utf-8')
        check('report contains finalized evidence', report.startswith('# Task 2 예비 실험 보고서\n'))
        if smoke is True:
            check('smoke report is explicitly marked', '파이프라인 시험(SMOKE)' in report)
    section('finalized report', verify_report)

    errors = [f"{item['name']}: {item['detail']}" if item["detail"] else item["name"]
              for item in checks if not item["passed"]]
    result = {"passed": not errors, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "run_dir": str(run_dir), "smoke_test_only": smoke,
              "checks_passed": sum(item["passed"] for item in checks), "checks_total": len(checks),
              "recomputed_validation": recomputed, "errors": errors, "checks": checks,
              "scope": "Saved-artifact consistency only; no checkpoint inference or test image evaluation."}
    output = run_dir / "verification.json"
    if run_dir.is_dir():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = verify_run(args.run_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
