"""Audit the existing manifests without changing splits or evaluating test data."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFESTS = ("train.csv", "val.csv", "test.csv", "class_info.csv")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def default_data_root():
    return json.loads((HERE / "dataset_paths.json").read_text(encoding="utf-8"))["vmmrdb_root"]


def safe_relative_path(value):
    # Reject Windows drive paths even when the audit itself runs on Linux.
    posix, windows = PurePosixPath(value), PureWindowsPath(value)
    return bool(value) and not (posix.is_absolute() or windows.anchor or "\\" in value
                               or ".." in posix.parts or ":" in value)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def inspect_inputs(split_dir, data_root, expected_classes=50):
    """Return audit and parsed CSVs; test images are never opened or stat'ed."""
    folder = Path(split_dir).expanduser().resolve()
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_dir": str(folder), "data_root_requested": str(data_root),
        "state": "checking", "errors": [], "split_sha256": {},
        "test_images_opened": False, "test_evaluated": False,
        "near_duplicates_checked": False, "train_hashes_verified": 0,
    }
    ci, parts = [], {}
    try:
        for name in MANIFESTS:
            audit["split_sha256"][name] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        ci = sorted(read_csv(folder / "class_info.csv"), key=lambda row: int(row["model_idx"]))
        ids = [int(row["model_idx"]) for row in ci]
        if ids != list(range(expected_classes)):
            raise ValueError(f"Expected {expected_classes} contiguous class IDs, found {ids}")
        labels = [row["make_model"] for row in ci]
        if len(set(labels)) != len(labels):
            raise ValueError("Duplicate class labels in class_info.csv")
        audit["nclasses"] = len(ci)
        audit["split_sizes"] = {}
        audit["class_counts"] = {name: {} for name in ("train", "val", "test")}
        for name in ("train", "val", "test"):
            rows = read_csv(folder / f"{name}.csv")
            paths, counts = set(), Counter()
            for row in rows:
                idx = int(row["model_idx"])
                path = row["rel_path"]
                if idx not in ids or labels[idx] != row["make_model"]:
                    raise ValueError(f"Invalid class mapping in {name}: {row}")
                if not safe_relative_path(path) or path in paths:
                    raise ValueError(f"Unsafe or duplicate path in {name}: {path}")
                if not re.fullmatch(r"[0-9a-fA-F]{32}", row["md5"]):
                    raise ValueError(f"Invalid MD5 in {name}: {path}")
                row["md5"] = row["md5"].lower()
                paths.add(path)
                counts[idx] += 1
            if set(counts) != set(ids):
                raise ValueError(f"Missing classes in {name}")
            for row in ci:
                idx = int(row["model_idx"])
                if int(row[f"n_{name}"]) != counts[idx]:
                    raise ValueError(f"class_info count mismatch: {name}, {row['make_model']}")
            parts[name] = rows
            audit["split_sizes"][name] = len(rows)
            audit["class_counts"][name] = dict(sorted(counts.items()))
        audit["cross_split_overlap"] = {}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlaps = {key: len({r[key] for r in parts[a]} & {r[key] for r in parts[b]})
                        for key in ("rel_path", "md5")}
            audit["cross_split_overlap"][f"{a}/{b}"] = overlaps
            if any(overlaps.values()):
                raise ValueError(f"Cross-split leakage: {a}/{b}, {overlaps}")
        audit["manifest_checks_passed"] = True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        audit.update(state="invalid_manifests", manifest_checks_passed=False)
        audit["errors"].append(str(exc))
        return audit, ci, parts

    foreign_windows_path = os.name != "nt" and bool(PureWindowsPath(str(data_root)).drive)
    data = Path(data_root).expanduser()
    audit["data_root_accessible"] = not foreign_windows_path and data.is_dir()
    if not audit["data_root_accessible"]:
        audit["state"] = "blocked_missing_data"
        audit["errors"].append(
            "Windows C: drive is not mounted in this Linux environment. Run on Windows or pass --data-root with a mounted path."
            if foreign_windows_path else f"Dataset directory is unavailable: {data}")
        audit["image_existence_checked"] = False
        return audit, ci, parts

    audit["data_root_resolved"] = str(data.resolve())
    audit["missing_images"] = {}
    for name in ("train", "val"):
        missing = [row["rel_path"] for row in parts[name] if not (data / row["rel_path"]).is_file()]
        audit["missing_images"][name] = {"count": len(missing), "examples": missing[:5]}
    audit["image_existence_checked"] = True
    if any(value["count"] for value in audit["missing_images"].values()):
        audit["state"] = "blocked_missing_data"
        audit["errors"].append("Missing train/validation images; data root must directly contain the class folders.")
        return audit, ci, parts
    audit["hash_sample_method"] = "random.Random(0).sample(train manifest, min(16, n))"
    try:
        for row in random.Random(0).sample(parts["train"], min(16, len(parts["train"]))):
            actual = hashlib.md5((data / row["rel_path"]).read_bytes()).hexdigest()
            if actual != row["md5"]:
                raise ValueError(f"Training image differs from manifest: {row['rel_path']}")
            audit["train_hashes_verified"] += 1
    except (OSError, ValueError) as exc:
        audit["state"] = "invalid_images"
        audit["errors"].append(str(exc))
        return audit, ci, parts
    audit["state"] = "ready"
    return audit, ci, parts


def save_audit(out, audit, ci):
    out = Path(out)
    write_json(out / "data_audit.json", audit)
    if audit.get("manifest_checks_passed"):
        with (out / "split_class_counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["model_idx", "make_model", "train", "val", "test"])
            for row in ci:
                idx = int(row["model_idx"])
                writer.writerow([idx, row["make_model"], *[audit["class_counts"][n][idx] for n in ("train", "val", "test")]])


def environment():
    info = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for name in ("torch", "torchvision", "numpy", "pandas", "Pillow", "matplotlib", "scikit-learn"):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    info["dependencies_present"] = all(info["packages"].values())
    info["package_imports_verified"] = False
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=default_data_root())
    parser.add_argument("--split-dir", type=Path, default=ROOT / "splits")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "preflight")
    args = parser.parse_args()
    audit, ci, _ = inspect_inputs(args.split_dir, args.data_root)
    save_audit(args.output_dir, audit, ci)
    env = environment()
    write_json(args.output_dir / "environment.json", env)
    state = audit["state"]
    if state == "ready" and not env["dependencies_present"]:
        state = "blocked_missing_dependencies"
    write_json(args.output_dir / "status.json", {
        "state": state, "training_started": False, "training_results_available": False,
        "test_evaluated": False, "updated_utc": datetime.now(timezone.utc).isoformat(),
    })
    sizes = audit.get("split_sizes", {})
    lines = ["# 한서진 Task 2 사전 점검 결과", "", f"- 상태: `{state}`",
             "- 실제 학습은 실행하지 않았으며 정확도/F1 결과가 없습니다.",
             f"- 기존 분할 검증: {audit.get('manifest_checks_passed', False)}",
             f"- 클래스 수: {audit.get('nclasses', '미확인')}",
             f"- 학습/검증/테스트: {sizes.get('train', '?')} / {sizes.get('val', '?')} / {sizes.get('test', '?')}장",
             "- 테스트는 CSV 메타데이터만 점검했으며 이미지를 열거나 성능을 평가하지 않았습니다.",
             "- 분할 CSV의 경로·MD5 교집합을 점검했습니다. 이미지 근접 중복은 검사하지 않았습니다.",
             f"- 실제 이미지 MD5 확인 수: {audit['train_hashes_verified']}", "",
             "## 실행 상태", "", *audit["errors"], "",
             "Windows에서 `python task2_preliminary_seojin/train_baseline.py`를 실행하면",
             "12 epoch 학습 후 results/baseline_seed0_seojin/에 실제 결과가 저장됩니다.",
             "원본 이미지는 Git 저장소에 포함되어 있지 않습니다."]
    (args.output_dir / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PREFLIGHT: {state}; splits={sizes}; saved to {args.output_dir}")
    for error in audit["errors"]:
        print(error)
    return 0 if state == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
