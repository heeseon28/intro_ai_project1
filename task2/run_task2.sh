#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
exec "${PYTHON:-python3}" "$ROOT/task2/train_baseline.py" "$@"
