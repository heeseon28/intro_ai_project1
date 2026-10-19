"""Pack selected VMMRdb train/val images on Windows; verify/unpack on the server.

Uses only the Python standard library. Never includes CompCar or test images.
Archives and unpacked images default to the repository's ignored dataset/ folder.
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from preflight import ROOT, default_data_root, inspect_inputs, write_json

INFO = "_task2_bundle.json"
CHUNK = 1024 * 1024


def required_inputs(split_dir, data_root=None):
    if data_root is None:
        # Validate CSVs without requiring or touching any original image.
        with tempfile.TemporaryDirectory(prefix="task2_metadata_") as temp:
            audit, _, parts = inspect_inputs(split_dir, Path(temp) / "absent")
        expected_state = "blocked_missing_data"
    else:
        audit, _, parts = inspect_inputs(split_dir, data_root)
        expected_state = "ready"
    if not audit.get("manifest_checks_passed") or audit["state"] != expected_state:
        raise ValueError("; ".join(audit["errors"]))
    return audit, parts["train"] + parts["val"]


def copy_and_verify(source, target, expected_md5, label):
    digest = hashlib.md5()
    while chunk := source.read(CHUNK):
        target.write(chunk)
        digest.update(chunk)
    if digest.hexdigest() != expected_md5:
        raise ValueError(f"Image MD5 does not match the existing split CSV: {label}")


def check_space(parent, required_bytes):
    available = shutil.disk_usage(parent).free
    if available < required_bytes:
        raise OSError(f"Not enough free disk space: need {required_bytes/2**30:.2f} GiB, "
                      f"available {available/2**30:.2f} GiB at {parent}")


def progress(action, done, total):
    if done % 2000 == 0 or done == total:
        print(f"{action}: {done:,}/{total:,} images ({done/total:.1%})", flush=True)


def pack(data_root, split_dir, output):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}. Reuse it or choose another --output.")
    print("Checking existing splits and source images...", flush=True)
    audit, rows = required_inputs(split_dir, data_root)
    data = Path(data_root).expanduser().resolve()
    total_bytes = sum((data / row["rel_path"]).stat().st_size for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    check_space(output.parent, total_bytes + len(rows) * 1024 + CHUNK)
    info = {
        "format": "task2-vmmrdb-train-val-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_sha256": audit["split_sha256"],
        "train_images": audit["split_sizes"]["train"],
        "validation_images": audit["split_sizes"]["val"],
        "test_images_included": False,
        "image_count": len(rows), "image_bytes": total_bytes,
        "all_image_md5_verified": True,
        "compression": "ZIP_STORED (JPEG images are already compressed)",
    }
    print(f"Packing {len(rows):,} train/validation images; {total_bytes/2**30:.2f} GiB. "
          "Original images remain in place.", flush=True)
    with tempfile.TemporaryDirectory(prefix=".task2_pack_", dir=output.parent) as temp:
        staged = Path(temp) / "bundle.zip"
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for number, row in enumerate(rows, 1):
                with (data / row["rel_path"]).open("rb") as source:
                    with archive.open(row["rel_path"], "w", force_zip64=True) as target:
                        copy_and_verify(source, target, row["md5"], row["rel_path"])
                progress("PACK", number, len(rows))
            archive.writestr(INFO, json.dumps(info, ensure_ascii=False, indent=2))
        if output.exists():
            raise FileExistsError(f"Archive appeared during packing; preserved existing file: {output}")
        staged.rename(output)
    print(f"PACK_COMPLETE: {output}", flush=True)
    print(f"Transfer this one ZIP file to the server ({output.stat().st_size/2**30:.2f} GiB).", flush=True)
    return info


def unpack(archive_path, split_dir, destination):
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}. Existing data was preserved.")
    audit, rows = required_inputs(split_dir)
    expected = {row["rel_path"]: row["md5"] for row in rows}
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected) | {INFO}:
            raise ValueError("Archive members differ from the train/validation manifests; no files extracted.")
        if archive.getinfo(INFO).file_size > 65536:
            raise ValueError("Invalid archive metadata size")
        info = json.loads(archive.read(INFO))
        if info.get("format") != "task2-vmmrdb-train-val-v1":
            raise ValueError("Unsupported bundle format")
        if info.get("split_sha256") != audit["split_sha256"]:
            raise ValueError("Windows/server split CSVs differ; use the same repository revision on both.")
        if info.get("image_count") != len(rows) or info.get("test_images_included") is not False:
            raise ValueError("Invalid image count or test-image flag")
        total_bytes = sum(archive.getinfo(name).file_size for name in expected)
        if total_bytes != info.get("image_bytes"):
            raise ValueError("Archive size does not match bundle metadata")
        destination.parent.mkdir(parents=True, exist_ok=True)
        check_space(destination.parent, total_bytes + len(rows) * 4096 + CHUNK)
        # Publish the data directory only after all images have passed their original MD5 checks.
        with tempfile.TemporaryDirectory(prefix=".task2_unpack_", dir=destination.parent) as temp:
            staged = Path(temp) / "images"
            staged.mkdir()
            for number, (name, digest) in enumerate(expected.items(), 1):
                target = staged / name  # Names already matched validated relative manifest paths.
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("xb") as stream:
                    copy_and_verify(source, stream, digest, name)
                progress("UNPACK", number, len(expected))
            info["unpacked_utc"] = datetime.now(timezone.utc).isoformat()
            info["all_extracted_image_md5_verified"] = True
            write_json(staged / INFO, info)
            if destination.exists():
                raise FileExistsError(f"Destination appeared during extraction: {destination}")
            staged.rename(destination)
    print(f"UNPACK_COMPLETE: {destination}", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    sender = subparsers.add_parser("pack", help="Run on Windows where VMMRdb is stored")
    sender.add_argument("--data-root", default=default_data_root())
    sender.add_argument("--split-dir", type=Path, default=ROOT / "splits")
    sender.add_argument("--output", type=Path, default=ROOT / "dataset/transfer/vmmrdb_task2_train_val.zip")
    receiver = subparsers.add_parser("unpack", help="Run on the GPU server after uploading the ZIP")
    receiver.add_argument("--archive", type=Path, default=ROOT / "dataset/transfer/vmmrdb_task2_train_val.zip")
    receiver.add_argument("--split-dir", type=Path, default=ROOT / "splits")
    receiver.add_argument("--destination", type=Path, default=ROOT / "dataset/VMMRdb")
    args = parser.parse_args()
    try:
        if args.action == "pack":
            pack(args.data_root, args.split_dir, args.output)
        else:
            unpack(args.archive, args.split_dir, args.destination)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"TRANSFER_ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
