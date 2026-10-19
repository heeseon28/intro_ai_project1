"""Audit the existing Task 2 manifests and local image availability.

The test manifest is inspected as metadata only. No test image is opened.
This audit does not regenerate a split and does not train a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SPLITS = ("train", "val", "test")


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO.parent / "VMMRdb")
    parser.add_argument("--compcar-root", type=Path, default=REPO.parent / "CompCar")
    parser.add_argument("--split-dir", type=Path, default=REPO / "splits")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "preflight")
    args = parser.parse_args()
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    for field in ("data_root", "compcar_root", "split_dir", "output_dir"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    environment = {
        "started_utc": started_utc,
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "data_root": str(args.data_root),
        "compcar_root": str(args.compcar_root),
        "split_dir": str(args.split_dir),
        "git_revision": revision,
        "packages": {name: package_version(name) for name in ("torch", "torchvision", "Pillow", "numpy", "pandas")},
        "torch_imported": False,
        "cuda_probe": "Not probed here; training records the selected device.",
        "compcar_required_for_task2": False,
    }
    save_json(args.output_dir / "environment.json", environment)
    save_json(args.output_dir / "status.json", {"state": "running", "started_utc": started_utc})
    audit: dict[str, object] = {
        "preflight_only": True,
        "test_images_opened": False,
        "test_performance_measured": False,
        "splits_regenerated": False,
        "near_duplicates_checked": False,
        "all_image_hashes_verified": False,
        "errors": errors,
        "warnings": warnings,
    }
    parts: dict[str, list[dict[str, str]]] = {}
    class_rows: list[dict[str, str]] = []
    hashes: dict[str, str] = {}
    required_class_columns = {"model_idx", "make_model", "make_idx", "body_idx", "n_train", "n_val", "n_test"}
    required_split_columns = {"rel_path", "md5", "make_model", "model_idx", "make_idx", "body_idx"}
    try:
        for filename in ("class_info.csv", "train.csv", "val.csv", "test.csv"):
            path = args.split_dir / filename
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            columns, rows = read_csv(path)
            needed = required_class_columns if filename == "class_info.csv" else required_split_columns
            missing_columns = sorted(needed - set(columns))
            if missing_columns:
                errors.append(f"{filename}: missing columns {missing_columns}")
            if filename == "class_info.csv":
                class_rows = rows
            else:
                parts[filename[:-4]] = rows
    except (OSError, csv.Error, UnicodeError) as exc:
        errors.append(f"Unable to read split manifests: {type(exc).__name__}: {exc}")
    audit["split_sha256"] = hashes
    audit["split_sizes"] = {name: len(parts.get(name, [])) for name in SPLITS}
    audit["nclasses"] = len(class_rows)
    class_map: dict[int, dict[str, str]] = {}
    class_counts: dict[str, Counter[int]] = {name: Counter() for name in SPLITS}
    if not errors:
        try:
            class_ids = [int(row["model_idx"]) for row in class_rows]
            if sorted(class_ids) != list(range(50)):
                errors.append("class_info.csv must contain each model_idx 0..49 exactly once.")
            class_map = {int(row["model_idx"]): row for row in class_rows}
            if len({row["make_model"] for row in class_rows}) != 50:
                errors.append("class_info.csv must contain 50 unique make_model labels.")
        except (ValueError, TypeError, KeyError) as exc:
            errors.append(f"Invalid class_info metadata: {exc}")
    metadata_summary: dict[str, object] = {}
    if class_map and all(name in parts for name in SPLITS):
        for name, rows in parts.items():
            issues: Counter[str] = Counter()
            issue_examples: list[dict[str, object]] = []
            for line, row in enumerate(rows, start=2):
                row_issues: list[str] = []
                if any(row.get(key) in (None, "") for key in required_split_columns):
                    row_issues.append("empty_required_field")
                try:
                    label = int(row.get("model_idx", ""))
                    class_counts[name][label] += 1
                    info = class_map.get(label)
                    if info is None:
                        row_issues.append("unknown_model_idx")
                    else:
                        for field in ("make_model", "make_idx", "body_idx"):
                            if row.get(field) != info.get(field):
                                row_issues.append(f"inconsistent_{field}")
                except (TypeError, ValueError):
                    row_issues.append("noninteger_model_idx")
                rel = row.get("rel_path") or ""
                normalized = rel.replace("\\", "/")
                if not rel or Path(rel).is_absolute() or PureWindowsPath(rel).drive or ".." in normalized.split("/"):
                    row_issues.append("unsafe_relative_path")
                if not re.fullmatch(r"[0-9a-fA-F]{32}", row.get("md5") or ""):
                    row_issues.append("invalid_md5")
                issues.update(row_issues)
                if row_issues and len(issue_examples) < 10:
                    issue_examples.append({"csv_line": line, "rel_path": rel, "issues": row_issues})
            paths = Counter((row.get("rel_path") or "").replace("\\", "/") for row in rows)
            md5s = Counter((row.get("md5") or "").lower() for row in rows)
            duplicate_paths = sum(count - 1 for count in paths.values() if count > 1)
            duplicate_md5 = sum(count - 1 for count in md5s.values() if count > 1)
            metadata_summary[name] = {
                "row_issue_counts": dict(issues),
                "issue_examples": issue_examples,
                "duplicate_path_rows": duplicate_paths,
                "duplicate_md5_rows": duplicate_md5,
                "class_ids": sorted(class_counts[name]),
            }
            if issues or duplicate_paths:
                errors.append(f"{name}.csv: metadata problems {dict(issues)}, duplicate path rows={duplicate_paths}")
            if set(class_counts[name]) != set(range(50)):
                errors.append(f"{name}.csv does not contain all and only the 50 class labels.")
            if duplicate_md5:
                warnings.append(f"{name}.csv includes {duplicate_md5} additional rows with repeated stored MD5.")
        audit["metadata"] = metadata_summary
        overlaps: dict[str, object] = {}
        for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
            values: dict[str, object] = {}
            for column in ("rel_path", "md5"):
                def normalized_value(row: dict[str, str]) -> str:
                    value = row.get(column) or ""
                    return value.lower() if column == "md5" else value.replace("\\", "/")
                overlap = {normalized_value(row) for row in parts[first]} & {normalized_value(row) for row in parts[second]}
                values[column] = {"count": len(overlap), "examples": sorted(overlap)[:10]}
                if overlap:
                    errors.append(f"Cross-split {column} leakage {first}/{second}: {len(overlap)} shared values.")
            overlaps[f"{first}/{second}"] = values
        audit["cross_split_overlap"] = overlaps
    count_rows: list[dict[str, object]] = []
    for label, info in sorted(class_map.items()):
        row: dict[str, object] = {"model_idx": label, "make_model": info["make_model"]}
        for name in SPLITS:
            actual = class_counts[name][label]
            try:
                expected = int(info[f"n_{name}"])
            except (ValueError, TypeError, KeyError):
                expected = -1
            row[f"n_{name}"] = actual
            row[f"expected_n_{name}"] = expected
            if actual != expected:
                errors.append(f"Class {label} {name} count {actual} differs from class_info count {expected}.")
        count_rows.append(row)
    count_fields = ["model_idx", "make_model", "n_train", "expected_n_train", "n_val", "expected_n_val", "n_test", "expected_n_test"]
    with (args.output_dir / "split_class_counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=count_fields)
        writer.writeheader()
        writer.writerows(count_rows)
    metadata_valid = not errors
    audit["metadata_valid"] = metadata_valid
    # Enumerate each referenced train/validation directory once; test image
    # content is never read, even when a directory contains images from all splits.
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for name in ("train", "val"):
        for row in parts.get(name, []):
            rel = row.get("rel_path") or ""
            path = Path(rel)
            if path.is_absolute() or PureWindowsPath(rel).drive or ".." in rel.replace("\\", "/").split("/"):
                continue
            grouped[str(path.parent)].append((name, rel, path.name))
    access_errors: list[dict[str, str]] = []
    missing_images: list[dict[str, str]] = []
    missing_counts: Counter[str] = Counter()
    root_accessible = True
    try:
        with os.scandir(args.data_root) as entries:
            next(entries, None)
    except OSError as exc:
        root_accessible = False
        access_errors.append({"path": str(args.data_root), "error": f"{type(exc).__name__}: {exc}"})
        errors.append(f"VMMRdb directory cannot be accessed: {type(exc).__name__}: {exc}")
    for parent, references in grouped.items():
        names: set[str] = set()
        reason = "file_missing"
        if root_accessible:
            try:
                with os.scandir(args.data_root / parent) as entries:
                    names = {entry.name for entry in entries if entry.is_file()}
            except OSError as exc:
                reason = "parent_directory_inaccessible"
                access_errors.append({"path": str(args.data_root / parent), "error": f"{type(exc).__name__}: {exc}"})
        else:
            reason = "dataset_root_inaccessible"
        for name, rel, filename in references:
            if filename not in names:
                missing_counts[name] += 1
                missing_images.append({"split": name, "rel_path": rel, "reason": reason})
    if missing_images:
        errors.append(f"Unavailable train/validation images: train={missing_counts['train']}, val={missing_counts['val']}.")
        with (args.output_dir / "missing_images.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["split", "rel_path", "reason"])
            writer.writeheader()
            writer.writerows(missing_images)
    elif (args.output_dir / "missing_images.csv").exists():
        (args.output_dir / "missing_images.csv").unlink()
    audit["image_availability"] = {
        "data_root_accessible": root_accessible,
        "directories_enumerated": len(grouped) if root_accessible else 0,
        "images_checked": {name: len(parts.get(name, [])) for name in ("train", "val")},
        "missing_counts": {name: missing_counts[name] for name in ("train", "val")},
        "first_missing_examples": missing_images[:20],
        "access_errors": access_errors,
        "test_image_existence_checked": False,
    }
    try:
        from PIL import Image
        pil_available = True
    except ImportError:
        pil_available = False
        warnings.append("Pillow is unavailable: sampled image decoding was not checked.")
    samples: list[dict[str, object]] = []
    # In particular, an invalid manifest with train/test overlap must never
    # cause a reserved test image to be read through the training manifest.
    train_rows = parts.get("train", []) if metadata_valid else []
    if not metadata_valid:
        warnings.append("Image-content sampling was skipped because manifest metadata failed validation.")
    unavailable = {row["rel_path"] for row in missing_images if row["split"] == "train"}
    for row in random.Random(0).sample(train_rows, min(16, len(train_rows))):
        sample: dict[str, object] = {"rel_path": row["rel_path"], "expected_md5": row["md5"], "hash_verified": False, "decoded": False}
        if row["rel_path"] in unavailable:
            sample["error"] = "Sampled train image is unavailable."
        else:
            try:
                path = args.data_root / row["rel_path"]
                actual_hash = hashlib.md5(path.read_bytes()).hexdigest()
                sample["actual_md5"] = actual_hash
                sample["hash_verified"] = actual_hash == row["md5"].lower()
                if not sample["hash_verified"]:
                    errors.append(f"Sampled train MD5 mismatch: {row['rel_path']}")
                if pil_available:
                    with Image.open(path) as image:
                        image.load()
                        sample.update({"decoded": True, "image_size": list(image.size), "image_mode": image.mode})
            except (OSError, ValueError) as exc:
                sample["error"] = f"{type(exc).__name__}: {exc}"
                errors.append(f"Sampled train image failed: {row['rel_path']}: {exc}")
        samples.append(sample)
    audit["sampled_train_images"] = {
        "sampling_seed": 0,
        "requested": 16,
        "sampled": len(samples),
        "hashes_verified": sum(bool(item["hash_verified"]) for item in samples),
        "decoded": sum(bool(item["decoded"]) for item in samples),
        "pillow_available": pil_available,
        "samples": samples,
    }
    ready = not errors
    elapsed = time.monotonic() - started
    audit.update({"ready_for_training": ready, "runtime_seconds": elapsed})
    save_json(args.output_dir / "data_audit.json", audit)
    status = {
        "state": "ready" if ready else "blocked",
        "preflight_complete": True,
        "ready_for_training": ready,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "runtime_seconds": elapsed,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(args.output_dir / "status.json", status)
    lines = [
        "# Task 2 local data preflight", "",
        f"Status: **{status['state']}**. Runtime: {elapsed:.2f} seconds.", "",
        f"VMMRdb: `{args.data_root}`", "",
        "CompCar is not required for the Task 2 baseline.", "",
        "| Split | Manifest images | Unavailable images |", "|---|---:|---:|",
        *[f"| {name} | {len(parts.get(name, [])):,} | {missing_counts[name]:,} |" for name in ("train", "val")],
        f"| test | {len(parts.get('test', [])):,} | Not inspected |", "",
        f"Classes: {len(class_rows)}. Sampled training hashes verified: {sum(bool(item['hash_verified']) for item in samples)}/{len(samples)}; decoded: {sum(bool(item['decoded']) for item in samples)}/{len(samples)}.", "",
        "This is a data audit, not a training result. No test image was opened or evaluated. The existing split CSVs were not changed. Stored paths and MD5s were checked for cross-split overlap; near duplicates and all image hashes were not audited.", "",
        "## Errors", "", *([f"- {message}" for message in errors] or ["None."]), "",
        "## Warnings", "", *([f"- {message}" for message in warnings] or ["None."]), "",
    ]
    if missing_images:
        lines.extend(["## First unavailable images", "", *[f"- `{item['split']}`: `{item['rel_path']}` ({item['reason']})" for item in missing_images[:10]], "", "Full unavailable-image list: `missing_images.csv`.", ""])
    (args.output_dir / "preflight_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
