#!/usr/bin/env bash
# Rebuild the report PDF and the Activity 2 submission archive from files tracked in this repository.
# Usage: bash task2/submission/make_submission.sh
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SUB="$ROOT/task2/submission"
NAME="Activity2_Submission"
STAGE="$(mktemp -d)/$NAME"
trap 'rm -rf "$(dirname "$STAGE")"' EXIT

(cd "$ROOT/task2/report" && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex >/dev/null)
pages=$(sed -n 's/^Output written on .*(\([0-9]*\) pages\{0,1\},.*/\1/p' "$ROOT/task2/report/build/main.log")
[[ "$pages" =~ ^[0-9]+$ ]] || { echo "could not read the page count from main.log" >&2; exit 1; }
[ "$pages" -le 2 ] || { echo "report has $pages pages; the limit is 2" >&2; exit 1; }
cp "$ROOT/task2/report/build/main.pdf" "$SUB/Activity2_Report.pdf"

mkdir -p "$STAGE/task2/report/figures" "$STAGE/splits" "$STAGE/result_task2"
cp "$SUB/Activity2_Report.pdf" "$SUB/README.md" "$STAGE/"
cp "$ROOT"/task2/{train_baseline.py,run_task2.sh,requirements.txt} "$STAGE/task2/"
cp "$ROOT"/task2/report/{main.tex,make_figures.py} "$STAGE/task2/report/"
cp "$ROOT"/task2/report/figures/*.pdf "$STAGE/task2/report/figures/"
cp "$ROOT"/splits/{train,val,test,class_info}.csv "$STAGE/splits/"
for run in baseline_seed0_DonghyunShin baseline_seed0_Hyeonil_Cho; do
  rsync -a --exclude checkpoints --exclude '.DS_Store' "$ROOT/result_task2/$run" "$STAGE/result_task2/"
done

# The shipped script must be the one that produced run A.
want=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['source_sha256'])" "$ROOT/result_task2/baseline_seed0_DonghyunShin/config.json")
got=$(shasum -a 256 "$STAGE/task2/train_baseline.py" | cut -d' ' -f1)
[ "$want" = "$got" ] || { echo "train_baseline.py differs from the script recorded in run A" >&2; exit 1; }

rm -f "$SUB/$NAME.zip"
(cd "$(dirname "$STAGE")" && zip -rqX "$SUB/$NAME.zip" "$NAME")
echo "Wrote $SUB/$NAME.zip ($(du -h "$SUB/$NAME.zip" | cut -f1))"
