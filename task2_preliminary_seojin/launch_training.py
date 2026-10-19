"""Start Task 2 in the background; training saves its own report and verification."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def process_alive(pid):
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 5:
                raise RuntimeError('Cannot inspect an existing process; do not launch a duplicate.')
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto')
    parser.add_argument('--data-root', type=Path, default=HERE.parent.parent / 'VMMRdb')
    parser.add_argument('--run-name', default='baseline_seed0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        parser.error('Invalid run name')
    out = HERE / 'results' / args.run_name
    process_file = out / 'process.json'
    if process_file.exists():
        previous = json.loads(process_file.read_text(encoding='utf-8'))
        if process_alive(int(previous['pid'])):
            parser.error(f"Training process {previous['pid']} is already active.")
    if (out / 'config.json').exists() and not args.resume:
        parser.error('This run exists. Resume it with --resume or use a new --run-name.')
    if (out / 'status.json').exists():
        status = json.loads((out / 'status.json').read_text(encoding='utf-8'))
        if status.get('state') == 'complete':
            parser.error('This run is already complete.')
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(HERE / 'train_baseline.py'),
               '--device', args.device, '--data-root', str(args.data_root.resolve()),
               '--run-name', args.run_name, '--threads', str(args.threads),
               '--workers', str(args.workers)]
    if args.resume:
        command.append('--resume')
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    options = {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
    with (out / 'process.stdout.log').open('a', encoding='utf-8') as stdout, (out / 'process.stderr.log').open('a', encoding='utf-8') as stderr:
        process = subprocess.Popen(command, cwd=HERE.parent, env=env, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, **options)
    record = {'pid': process.pid, 'launched_utc': datetime.now(timezone.utc).isoformat(),
              'command': command, 'output_dir': str(out),
              'note': 'Check status.json and run.log for progress. Completion includes report.md and verification.json.'}
    process_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Launched PID {process.pid}: {out}')


if __name__ == '__main__':
    main()
