"""Read progress from the existing training log without touching the trainer.

Batch counters are observations, never estimates based on the current time.
The legacy trainer logs only every 25 batches; a quiet interval therefore does
not change the reported counts.  The logged speed is the average since this
pass began.  A separate recent speed uses elapsed-time differences, and long
observed intervals are excluded from that estimate without assuming a cause.
"""

import ast
from collections import deque
import json
import math
import re
from statistics import median


_START = re.compile(r"^.*?TASK2_BASELINE_START\s+(.*)$", re.MULTILINE)
_RESUME = re.compile(r"RESUME at epoch\s+(\d+)")
_BATCH = re.compile(r"\b(train|eval) batch (\d+)/(\d+)")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_LABELS = {
    "waiting": "학습 로그 대기",
    "train": "학습",
    "val": "검증",
    "final_train": "최종 학습 데이터 평가",
    "final_val": "최종 검증 데이터 평가",
    "final_evaluation": "최종 평가 및 결과 저장",
    "complete": "완료",
    "failed": "실패",
}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _positive_int(value, default):
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError):
        return default
    return result if result > 0 else default


def _nonnegative_int(value, default=0):
    try:
        result = int(value)
    except (ValueError, TypeError, OverflowError):
        return default
    return result if result >= 0 else default


def _read_json_record(line, marker):
    try:
        return _mapping(json.loads(line.split(marker, 1)[1].strip()))
    except (ValueError, IndexError, TypeError):
        return {}


def _number(line, key):
    match = re.search(r"\b" + re.escape(key) + r"=(" + _NUMBER + r")", line)
    if not match:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) else None


class _IntervalSpeed:
    """Track distinct pass boundaries and a short window of normal intervals."""

    def __init__(self, batch_size):
        self.reference_images = 25 * batch_size
        self.long_gap_count = 0
        self.long_gap_seconds = 0.0
        self.last_long_gap_seconds = None
        self.reset_phase()

    def reset_phase(self):
        self.key = None
        self.previous = None
        self.intervals = deque(maxlen=3)
        self.normal_durations = deque(maxlen=9)
        self.waiting_after_gap = False
        self.phase_long_gap_count = 0
        self.phase_long_gap_seconds = 0.0

    def observe(self, key, images, elapsed):
        if key != self.key:
            self.reset_phase()
            self.key = key
        if images is None or elapsed is None or elapsed < 0:
            return
        current = (images, elapsed)
        previous = self.previous
        self.previous = current
        if previous is None:
            return
        image_delta, seconds = images - previous[0], elapsed - previous[1]
        if image_delta <= 0 or seconds <= 0:
            # Rounded elapsed minutes can repeat; timer/counter resets and
            # duplicate records also cannot establish a meaningful rate.
            self.intervals.clear()
            return
        normalized_seconds = seconds * self.reference_images / image_delta
        threshold = max(300.0, 3 * median(self.normal_durations)
                        if self.normal_durations else 300.0)
        if normalized_seconds >= threshold:
            self.long_gap_count += 1
            self.long_gap_seconds += seconds
            self.last_long_gap_seconds = seconds
            self.phase_long_gap_count += 1
            self.phase_long_gap_seconds += seconds
            self.intervals.clear()
            self.waiting_after_gap = True
            return
        self.normal_durations.append(normalized_seconds)
        self.intervals.append((image_delta, seconds))
        self.waiting_after_gap = False

    @property
    def recent_speed(self):
        if not self.intervals:
            return None
        images = sum(item[0] for item in self.intervals)
        seconds = sum(item[1] for item in self.intervals)
        return images / seconds if seconds > 0 else None


def parse_progress(log_text, config=None, audit=None, status=None):
    """Return a JSON-serializable progress snapshot for a legacy ``run.log``.

    ``config``, ``audit``, and ``status`` are optional decoded JSON objects.
    Dataset sizes default to the task's fixed split when metadata is absent.
    ``images_processed`` and ``phase_percent`` describe the current pass;
    ``training_images_processed`` and ``overall_training_percent`` count only
    training updates across epochs, excluding validation and final evaluation.
    ``elapsed_seconds`` belongs to the current pass and is rounded in the log.
    ``images_per_second`` remains the logged phase average for compatibility;
    ``phase_average_images_per_second`` names the same value explicitly.
    ``recent_images_per_second`` weights up to three contiguous normal log
    intervals by their durations.  A long interval clears that window, so ETA
    waits for a subsequent normal interval.  Before any interval is available,
    ETA can use the phase average, identified by ``eta_speed_basis``.
    ``long_gap_count`` and ``long_gap_seconds`` cover the selected attempt;
    ``phase_long_gap_*`` cover only the current pass.  These seconds are the
    observed intervals' full durations, not a claim about idle or sleep time.
    ``complete`` requires TASK2_COMPLETE or a complete terminal status; reaching
    the last training epoch alone does not mean evaluation has finished.
    """
    config = dict(_mapping(config))
    audit = _mapping(audit)
    status = _mapping(status)
    if not isinstance(log_text, str):
        log_text = ""

    # Select the current attempt before parsing its counters.  A restart can
    # append to this same log, and an interrupted epoch must not be added twice.
    starts = list(_START.finditer(log_text))
    resume_epoch = None
    inline_sizes = {}
    if starts:
        start = starts[-1]
        previous_end = starts[-2].end() if len(starts) > 1 else 0
        prefix = log_text[previous_end:start.start()]
        prefix = prefix.rsplit("RUNTIME_CHANGE", 1)[-1]
        resumes = list(_RESUME.finditer(prefix))
        if resumes:
            resume_epoch = int(resumes[-1].group(1))
        payload = start.group(1)
        config_text, separator, size_text = payload.partition(" sizes ")
        try:
            config.update(_mapping(ast.literal_eval(config_text)))
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
        if separator:
            try:
                inline_sizes = _mapping(ast.literal_eval(size_text))
            except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
                pass
        active_log = log_text[start.end():]
    else:
        active_log = log_text
        resumes = list(_RESUME.finditer(log_text))
        if resumes:
            resume_epoch = int(resumes[-1].group(1))

    smoke = config.get("smoke") is True or status.get("smoke") is True
    total_epochs = 1 if smoke else _positive_int(config.get("epochs"), 12)
    batch_size = 8 if smoke else _positive_int(config.get("batch_size"), 64)
    sizes = dict(inline_sizes)
    sizes.update(_mapping(audit.get("split_sizes")))
    train_size = _positive_int(sizes.get("train"), 82929)
    val_size = _positive_int(sizes.get("val"), 17771)
    if smoke:
        # data_audit.json precedes the trainer's per-class smoke sampling.
        nclasses = _positive_int(audit.get("nclasses"), 50)
        train_size, val_size = 2 * nclasses, nclasses
    train_work = min(train_size, 2 * batch_size) if smoke else train_size
    total_training = total_epochs * train_work

    epoch = min(resume_epoch or 1, total_epochs) if starts or resume_epoch else None
    completed = min(max((resume_epoch or 1) - 1, 0), total_epochs)
    phase = "train" if epoch is not None and completed < total_epochs else "waiting"
    if completed == total_epochs:
        phase = "final_evaluation"
    batch = total_batches = images = total_images = None
    speed = elapsed = loss = None
    current_train_images = 0
    final_pass = 0
    terminal = None
    interval_speed = _IntervalSpeed(batch_size)

    def clear_pass():
        nonlocal batch, total_batches, images, total_images, speed, elapsed, loss
        batch = total_batches = images = total_images = None
        speed = elapsed = loss = None
        interval_speed.reset_phase()

    for line in active_log.splitlines():
        if "EPOCH_RESULT" in line:
            row = _read_json_record(line, "EPOCH_RESULT")
            result_epoch = _positive_int(row.get("epoch"), None)
            if result_epoch is not None:
                completed = max(completed, min(result_epoch, total_epochs))
                epoch = min(completed + 1, total_epochs)
                current_train_images = 0
                phase = "train" if completed < total_epochs else "final_evaluation"
                final_pass = 0
                clear_pass()
            continue
        if "TASK2_COMPLETE" in line:
            terminal = "complete"
            continue
        if "FINAL_METRICS" in line:
            result = _read_json_record(line, "FINAL_METRICS")
            if result:
                completed = max(completed, min(
                    _nonnegative_int(result.get("epochs_completed"), completed),
                    total_epochs))
                phase = "final_evaluation"
                clear_pass()
            continue

        match = _BATCH.search(line)
        if not match:
            continue
        kind, observed_batch, loader_batches = match.groups()
        observed_batch, loader_batches = int(observed_batch), int(loader_batches)
        if observed_batch < 1 or loader_batches < 1 or observed_batch > loader_batches:
            continue
        observed_loss = _number(line, "loss")
        observed_speed = _number(line, "images/s")
        observed_elapsed = _number(line, "elapsed_min")
        if any(value is None for value in (observed_loss, observed_speed, observed_elapsed)):
            # A reader may catch the append part-way through a print call.
            continue
        if kind == "train":
            phase = "train"
            if epoch is None:
                epoch = min(completed + 1, total_epochs)
            current_train_images = min(observed_batch * batch_size, train_work)
        elif completed < total_epochs:
            phase = "val"
            # Entering validation proves that the training pass finished,
            # including batches between the final periodic log and this line.
            if epoch is not None:
                current_train_images = train_work
        else:
            expected_train_batches = math.ceil(train_size / batch_size)
            expected_val_batches = math.ceil(val_size / batch_size)
            if final_pass == 0:
                if loader_batches == expected_train_batches:
                    phase, final_pass = "final_train", 1
                elif loader_batches == expected_val_batches:
                    phase, final_pass = "final_val", 2
                else:
                    phase = "final_evaluation"
            elif final_pass == 1 and (
                (batch is not None and observed_batch <= batch)
                or (loader_batches == expected_val_batches
                    and expected_train_batches != expected_val_batches)
            ):
                phase, final_pass = "final_val", 2
            elif final_pass == 1:
                phase = "final_train"
            else:
                phase = "final_val"

        batch = observed_batch
        if phase in ("train", "final_train"):
            total_images = train_work
            total_batches = min(loader_batches, 2) if smoke else loader_batches
        elif phase in ("val", "final_val"):
            total_images = val_size
            total_batches = loader_batches
        else:
            total_images = None
            total_batches = loader_batches
        images = min(batch * batch_size, total_images) if total_images is not None else None
        speed = observed_speed
        elapsed = observed_elapsed * 60
        loss = observed_loss
        interval_speed.observe((epoch, phase, loader_batches), images, elapsed)

    # status.json is written before the matching epoch-result log record.  It
    # can therefore establish a completed epoch even in a snapshot of that gap.
    status_completed = min(_nonnegative_int(status.get("epoch_completed")), total_epochs)
    if status_completed > completed:
        completed = status_completed
        epoch = min(completed + 1, total_epochs)
        current_train_images = 0
        phase = "train" if completed < total_epochs else "final_evaluation"
        clear_pass()
    state = status.get("state")
    if state == "failed":
        terminal = "failed"
    elif state == "complete" and terminal != "failed":
        terminal = "complete"
    if terminal == "complete":
        completed = total_epochs
        epoch = total_epochs
        phase = "complete"
    elif terminal == "failed":
        phase = "failed"
    state = terminal or ("running" if epoch is not None or state == "running" else "waiting")
    training_images = min(completed * train_work + current_train_images, total_training)
    if epoch is None and completed == 0:
        training_images = None
    phase_percent = 100.0 * images / total_images if total_images else None
    recent_speed = interval_speed.recent_speed
    eta_speed = recent_speed
    eta_basis = "recent interval" if recent_speed is not None else None
    if recent_speed is None and not interval_speed.waiting_after_gap and speed is not None and speed > 0:
        eta_speed, eta_basis = speed, "phase average"
    if total_images is not None and images is not None and eta_speed is not None and eta_speed > 0 and terminal is None:
        remaining = (total_images - images) / eta_speed
    else:
        remaining, eta_basis = None, None
    return {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "phase": phase,
        "phase_label": _LABELS[phase],
        "batch": batch,
        "total_batches": total_batches,
        "images_processed": images,
        "total_images": total_images,
        "phase_percent": phase_percent,
        "epochs_completed": completed,
        "training_images_processed": training_images,
        "total_training_images": total_training,
        "overall_training_percent": (100.0 * training_images / total_training
                                     if training_images is not None else None),
        "images_per_second": speed,
        "phase_average_images_per_second": speed,
        "recent_images_per_second": recent_speed,
        "recent_speed_samples": len(interval_speed.intervals),
        "eta_speed_basis": eta_basis,
        "long_gap_count": interval_speed.long_gap_count,
        "long_gap_seconds": interval_speed.long_gap_seconds,
        "last_long_gap_seconds": interval_speed.last_long_gap_seconds,
        "phase_long_gap_count": interval_speed.phase_long_gap_count,
        "phase_long_gap_seconds": interval_speed.phase_long_gap_seconds,
        "elapsed_seconds": elapsed,
        "phase_remaining_seconds": remaining,
        "loss": loss,
        "state": state,
        "complete": terminal == "complete",
    }
