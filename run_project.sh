#!/usr/bin/env bash
# Run from any directory. PYTHON and DATASET_ROOT can be overridden.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${1:-analysis}"
case "$MODE" in
  pilot|smoke)
    PILOT_DIR="$ROOT/runs_pilot/pilot_seed0_v1"
    mkdir -p "$PILOT_DIR"
    exec > >(tee -a "$PILOT_DIR/launcher.log") 2>&1
    RUNTIME="$ROOT/.venv-project-runner/bin/python"
    [[ -x "$RUNTIME" ]] || { echo 'Run bash run_project.sh setup first'; exit 1; }
    "$RUNTIME" -u "$ROOT/pilot_train.py" --out-dir "$ROOT/runs_pilot/smoke_v1" --smoke
    if [[ "$MODE" == pilot ]]; then
      "$RUNTIME" -u "$ROOT/pilot_train.py" --out-dir "$PILOT_DIR"
    fi
    exit 0 ;;
  setup)
    python3 -m venv --system-site-packages "$ROOT/.venv-project-runner"
    "$ROOT/.venv-project-runner/bin/python" -m pip install --no-deps 'seaborn==0.13.2'
    exec bash "$ROOT/run_project.sh" check ;;
  check|analysis|prepare) ;;
  -h|--help|help)
    echo 'Usage: bash run_project.sh [setup|check|analysis|prepare|smoke|pilot]'
    echo 'pilot: smoke check, then 3 experiments x seed 0, up to 12 epochs; resume from checkpoints'
    echo 'smoke: tiny end-to-end execution check of all three models'
    echo 'setup: create compatible local runtime, then check'
    echo 'check: validate paths, packages, notebook syntax and Python kernel'
    echo 'analysis (default): execute dataset analysis notebook'
    echo 'prepare: execute analysis, then create data splits (cache disabled)'
    echo 'pilot uses pilot_train.py; the original training notebook remains unchanged.'
    exit 0 ;;
  *) echo "Unknown stage: $MODE" >&2; exit 2 ;;
esac
export DATASET_ROOT="${DATASET_ROOT:-$ROOT/dataset}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
PYTHON="${PYTHON:-$ROOT/.venv-project-runner/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="${PYTHON:-python3}"; fi
command -v "$PYTHON" >/dev/null || { echo "Missing Python: $PYTHON" >&2; exit 1; }
mkdir -p outputs/automation
RUN_DIR="$(mktemp -d "$ROOT/outputs/automation/${MODE}_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
export RUN_DIR
exec > >(tee "$RUN_DIR/run.log") 2>&1
trap 'rc=$?; echo "FAILED (exit $rc, line $LINENO). Log: $RUN_DIR/run.log"; exit "$rc"' ERR
printf 'Stage: %s\nResults: %s\n' "$MODE" "$RUN_DIR"
"$PYTHON" - "$MODE" <<'PY'
import ast, importlib, os, sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

mode = sys.argv[1]
root = Path.cwd()
data = Path(os.environ['DATASET_ROOT']).expanduser().resolve()
out = Path(os.environ['RUN_DIR'])
for name in ('VMMRdb', 'COMPCAR'):
    p = data / name
    if not p.is_dir() or next(p.iterdir(), None) is None:
        raise RuntimeError(f'Missing or empty dataset folder: {p}')
for name in ('numpy', 'pandas', 'scipy', 'sklearn', 'matplotlib', 'seaborn', 'PIL', 'tqdm', 'IPython', 'jinja2'):
    importlib.import_module(name)
# Use the exact Python interpreter running this script, not an unrelated kernelspec.
ksm = KernelSpecManager(kernel_dirs=[])
# A per-run kernel specification makes interpreter selection reproducible.
kdir = out / 'kernels' / 'project-python'
kdir.mkdir(parents=True)
import json
(kdir / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Project Python', 'language': 'python'
}))
ksm = KernelSpecManager(kernel_dirs=[str(kdir.parent)])

def client_for(nb):
    c = NotebookClient(nb, timeout=None, kernel_name='project-python',
                       resources={'metadata': {'path': str(root)}}, allow_errors=False)
    c.km = c.create_kernel_manager()
    c.km.kernel_spec_manager = ksm
    return c

def prepared(name):
    nb = nbformat.read(root / name, as_version=4)
    patched = 0
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        cell.outputs = []
        cell.execution_count = None
        s = cell.source
        if name.startswith('01_'):
            old = 'ROOT = load_dataset_root()'
            if old in s:
                s = s.replace(old, f'ROOT = Path({str(data)!r})')
                patched += 1
        else:
            old = 'Path(r"/Users/yuhuiseon/Downloads/dataset/VMMRdb")'
            if old in s:
                s = s.replace(old, f'Path({str(data / "VMMRdb")!r})')
                patched += 1
        # AppleGothic is unavailable on typical Linux servers.
        # Prefer an installed Korean font when possible.
        from matplotlib import font_manager
        fonts = {f.name for f in font_manager.fontManager.ttflist}
        font = next((f for f in ('Noto Sans CJK KR', 'NanumGothic', 'Noto Sans CJK JP') if f in fonts), 'DejaVu Sans')
        s = s.replace('AppleGothic', font)
        cell.source = s
        if not s.lstrip().startswith(('%', '!')):
            ast.parse(s)
    if patched != 1:
        raise RuntimeError(f'{name}: expected one path setting; found {patched}. Review updated notebook.')
    return nb

names = ['01_dataset_analysis.ipynb', '02_data_pipeline.ipynb']
books = {name: prepared(name) for name in names}
print('Paths, packages and notebook syntax: OK', flush=True)
print('Dataset root:', data, flush=True)
if mode == 'check':
    smoke = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('assert 2 + 2 == 4; print("KERNEL_OK")')])
    client_for(smoke).execute()
    assert any('KERNEL_OK' in o.get('text', '') for o in smoke.cells[0].outputs)
    print('Python notebook kernel: OK. No dataset analysis or training was started.', flush=True)
    sys.exit(0)

selected = names if mode == 'prepare' else names[:1]
for name in selected:
    nb = books[name]
    target = out / name
    nbformat.write(nb, target)
    client = client_for(nb)
    async def report(cell, cell_index, **kwargs):
        print(f'[{name}] cell {cell_index + 1}/{len(nb.cells)}', flush=True)
    client.on_cell_start = report
    print('START', name, flush=True)
    start = time.monotonic()
    try:
        client.execute()
    finally:
        # Preserve partial outputs and the failing cell, including on execution failure.
        nbformat.write(nb, target)
    print(f'DONE {name}: {(time.monotonic()-start)/60:.1f} min', flush=True)
print('Completed. Tables: outputs/; figures: figures/; splits: splits/ (prepare only).', flush=True)
print('Executed notebooks and run log:', out, flush=True)
PY
printf 'SUCCESS: %s\n' "$RUN_DIR"
