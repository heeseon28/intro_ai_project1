"""Follow a Task 2 run in a VS Code terminal without interrupting training."""

import argparse
import codecs
from collections import deque
import ctypes
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from training_progress import parse_progress

HERE = Path(__file__).resolve().parent
TERMINAL_STATES = {"complete", "failed"}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def process_alive(pid):
    """Return None if the operating system cannot inspect this process."""
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None if ctypes.get_last_error() == 5 else False
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None


class LogFollower:
    """Keep the file offset and decoder across polls, including partial UTF-8."""

    def __init__(self, path, tail):
        self.path = path
        self.tail = tail
        self.offset = 0
        self.identity = None
        self.boundary = b""
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def read(self):
        try:
            log = self.path.open("rb")
        except FileNotFoundError:
            return ""
        with log:
            stat = os.fstat(log.fileno())
            identity = (stat.st_dev, stat.st_ino)
            initial = self.identity is None
            reset = not initial and (identity != self.identity or stat.st_size < self.offset)
            if not initial and not reset and self.boundary:
                log.seek(self.offset - len(self.boundary))
                reset = log.read(len(self.boundary)) != self.boundary
            if initial:
                if self.tail:
                    chunk = b"".join(deque(log, maxlen=self.tail))
                else:
                    log.seek(0, os.SEEK_END)
                    chunk = b""
            else:
                if reset:
                    self.offset = 0
                    self.decoder.reset()
                log.seek(self.offset)
                chunk = log.read()
            self.offset = log.tell()
            self.identity = identity
            log.seek(max(0, self.offset - 64))
            self.boundary = log.read(self.offset - log.tell())
        notice = "\n[monitor] Log restarted; following the new log.\n" if reset else ""
        return notice + self.decoder.decode(chunk)


def _number(value):
    return f"{int(value):,}" if isinstance(value, (int, float)) and math.isfinite(value) else "-"


def _duration(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return "-"
    seconds = round(value)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_progress(progress, *, log_age=None, threads=None, width=24):
    """Describe only confirmed log progress; time-based estimates never add batches."""
    p = progress
    percent = p.get("phase_percent")
    known_percent = isinstance(percent, (int, float)) and math.isfinite(percent)
    filled = round(width * min(100, max(0, percent)) / 100) if known_percent else 0
    bar = "#" * filled + "-" * (width - filled)
    phase = p.get("phase_label") or "기록 대기"
    state = p.get("state") or "unknown"
    lines = [f"Epoch {_number(p.get('epoch'))}/{_number(p.get('total_epochs'))} | {phase} | 상태: {state}"]
    if known_percent:
        lines.append(f"[{bar}] {percent:.2f}% | 배치 {_number(p.get('batch'))}/{_number(p.get('total_batches'))}"
                     f" | 이미지 {_number(p.get('images_processed'))}/{_number(p.get('total_images'))}")
    else:
        lines.append("현재 단계의 배치 기록을 기다리는 중입니다.")
    overall = p.get("overall_training_percent")
    overall_text = f"{overall:.2f}%" if isinstance(overall, (int, float)) and math.isfinite(overall) else "-"
    lines.append(f"완료 epoch {_number(p.get('epochs_completed'))}/{_number(p.get('total_epochs'))}"
                 f" | 전체 학습(검증 제외) {overall_text}")
    recent_rate = p.get("recent_images_per_second")
    average_rate = p.get("phase_average_images_per_second", p.get("images_per_second"))
    recent_text = f"{recent_rate:.2f}장/초" if isinstance(recent_rate, (int, float)) and math.isfinite(recent_rate) and recent_rate > 0 else "측정 중"
    average_text = f"{average_rate:.2f}장/초" if isinstance(average_rate, (int, float)) and math.isfinite(average_rate) else "-"
    eta_basis = {"recent interval": "최근 구간 기준·추정", "phase average": "평균 기준·임시 추정"}.get(p.get("eta_speed_basis"), "계산 대기")
    lines.append(f"최근 구간 속도 {recent_text}"
                 f" | 이 단계 남은 시간({eta_basis}) {_duration(p.get('phase_remaining_seconds'))}")
    average_label = "현재 단계 평균(긴 공백 포함)" if p.get("phase_long_gap_count", 0) else "현재 단계 평균"
    lines.append(f"{average_label} {average_text}"
                 f" | 경과(로그 기준) {_duration(p.get('elapsed_seconds'))}")
    if p.get("phase_long_gap_count", 0):
        lines.append(f"현재 단계에서 긴 기록 공백 {_number(p['phase_long_gap_count'])}회 관측"
                     " · 남은 시간은 공백 이후의 최근 처리 속도로 계산합니다.")
    details = []
    if threads is not None:
        details.append(f"CPU {_number(threads)}스레드")
    if log_age is not None:
        details.append(f"마지막 로그 {log_age:.0f}초 전")
    details.append("확인된 배치 기준 · 학습 로그는 25배치마다 기록")
    lines.append(" | ".join(details))
    return "\n".join(lines)


def show_progress(out, status):
    try:
        log = (out / "run.log").read_text(encoding="utf-8", errors="replace")
        age = max(0, time.time() - (out / "run.log").stat().st_mtime)
    except FileNotFoundError:
        log, age = "", None
    progress = parse_progress(log, config=read_json(out / "config.json"),
                              audit=read_json(out / "data_audit.json"), status=status)
    threads = read_json(out / "environment.json").get("cpu_threads")
    print(f"\n[{time.strftime('%H:%M:%S')}] 확인된 학습 진행", flush=True)
    print(format_progress(progress, log_age=age, threads=threads), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="baseline_seed0")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--tail", type=int, default=12, help="Existing log lines to show initially")
    parser.add_argument("--heartbeat", type=float, default=15.0,
                        help="Seconds between status reminders while the trainer is quiet; 0 disables")
    parser.add_argument("--once", action="store_true", help="Show saved progress and exit")
    parser.add_argument("--raw", action="store_true", help="Also show original unformatted log lines")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("Invalid run name")
    if not 0 < args.interval < float("inf"):
        parser.error("--interval must be a positive finite number")
    if args.tail < 0:
        parser.error("--tail must be non-negative")
    if not 0 <= args.heartbeat < float("inf"):
        parser.error("--heartbeat must be a non-negative finite number")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", newline="")
    out = HERE / "results" / args.run_name
    if not out.is_dir():
        parser.error(f"Run directory does not exist: {out}")
    print(f"[monitor] {out / 'run.log'}", flush=True)
    if not args.once:
        print("[monitor] Ctrl+C stops this viewer; background training continues.", flush=True)
    follower = LogFollower(out / "run.log", args.tail)
    last_status = None
    last_output = time.monotonic()
    exited_pid = None
    exited_since = None
    while True:
        text = follower.read()
        if text and args.raw:
            print(text, end="", flush=True)
        status = read_json(out / "status.json")
        status_changed = status != last_status
        if text or status_changed or args.once:
            if status_changed and args.raw:
                print(f"\n[monitor] Status: {json.dumps(status, ensure_ascii=False)}", flush=True)
            show_progress(out, status)
            last_status = status
            last_output = time.monotonic()
        state = status.get("state")
        if args.once:
            return int(state == "failed")
        record = read_json(out / "process.json")
        pid = record.get("pid")
        alive = process_alive(pid) if isinstance(pid, int) and pid > 0 else None
        # The trainer writes complete before its final verification/log flush.
        if state in TERMINAL_STATES and alive is not True:
            final_text = follower.read()
            if final_text and args.raw:
                print(final_text, end="", flush=True)
            if final_text:
                show_progress(out, read_json(out / "status.json"))
            return int(state == "failed")
        if alive is False and state not in TERMINAL_STATES and state != "reconfiguring":
            if exited_pid != pid:
                exited_pid = pid
                exited_since = time.monotonic()
        else:
            exited_pid = None
            exited_since = None
        if exited_since is not None and time.monotonic() - exited_since >= max(2, args.interval):
            final_text = follower.read()
            if final_text and args.raw:
                print(final_text, end="", flush=True)
            print(f"\n[monitor] Process {pid} exited; saved status is {state!r}. "
                  f"Check {out / 'process.stderr.log'}.", flush=True)
            return 1
        if args.heartbeat and time.monotonic() - last_output >= args.heartbeat:
            show_progress(out, status)
            last_output = time.monotonic()
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[monitor] Viewer stopped. Background training continues.", flush=True)
