"""Meaningful state-transition checks for the read-only training progress viewer."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from training_progress import parse_progress
import monitor_training


CONFIG = {"epochs": 12, "batch_size": 64, "smoke": False}
AUDIT = {"split_sizes": {"train": 82929, "val": 17771, "test": 17771},
         "nclasses": 50}


def start(config=None):
    return "TASK2_BASELINE_START " + repr(config or CONFIG) + " sizes " + repr(AUDIT["split_sizes"])


def batch(number=26, total=1296, phase="train", rate=11.1, elapsed=2.5):
    return (f"  {phase} batch {number}/{total}, loss=3.4855, "
            f"images/s={rate:.2f}, elapsed_min={elapsed:.1f}")


def epoch_result(epoch):
    return "EPOCH_RESULT " + json.dumps({"epoch": epoch, "train_loss": 3.0, "val_loss": 3.1})


def parse(lines, config=None, status=None):
    return parse_progress("\n".join(lines) + "\n", config or CONFIG, AUDIT,
                          status or {"state": "running"})


class TrainingProgressTests(unittest.TestCase):
    def test_current_epoch_counts_images_and_percent_from_last_logged_batch(self):
        result = parse([start(), batch()])
        self.assertEqual(result["epoch"], 1)
        self.assertEqual(result["total_epochs"], 12)
        self.assertEqual(result["phase"], "train")
        self.assertEqual(result["batch"], 26)
        self.assertEqual(result["total_batches"], 1296)
        self.assertEqual(result["images_processed"], 1664)
        self.assertEqual(result["total_images"], 82929)
        self.assertAlmostEqual(result["phase_percent"], 100 * 1664 / 82929)
        self.assertEqual(result["epochs_completed"], 0)
        self.assertEqual(result["training_images_processed"], 1664)
        self.assertEqual(result["total_training_images"], 12 * 82929)
        self.assertAlmostEqual(result["overall_training_percent"], 100 * 1664 / (12 * 82929))
        self.assertEqual(result["images_per_second"], 11.1)
        self.assertEqual(result["elapsed_seconds"], 150)
        self.assertAlmostEqual(result["phase_remaining_seconds"], (82929 - 1664) / 11.1)

    def test_last_training_batch_clamps_images_to_actual_dataset_size(self):
        result = parse([start(), batch(1296)])
        self.assertEqual(result["images_processed"], 82929)
        self.assertEqual(result["phase_percent"], 100)
        self.assertEqual(result["epochs_completed"], 0)

    def test_validation_means_training_pass_completed_but_epoch_not_yet_saved(self):
        result = parse([start(), batch(1276), batch(26, 278, "eval")])
        self.assertEqual(result["phase"], "val")
        self.assertEqual(result["epoch"], 1)
        self.assertEqual(result["epochs_completed"], 0)
        self.assertEqual(result["batch"], 26)
        self.assertEqual(result["total_batches"], 278)
        self.assertEqual(result["images_processed"], 1664)
        self.assertEqual(result["total_images"], 17771)
        self.assertEqual(result["training_images_processed"], 82929)

    def test_epoch_result_then_train_belongs_to_next_epoch(self):
        result = parse([start(), batch(), epoch_result(1), batch(1)])
        self.assertEqual(result["epoch"], 2)
        self.assertEqual(result["epochs_completed"], 1)
        self.assertEqual(result["phase"], "train")
        self.assertEqual(result["images_processed"], 64)
        self.assertEqual(result["training_images_processed"], 82929 + 64)

    def test_resume_epoch_is_preserved_across_subsequent_start_marker(self):
        result = parse([start(), batch(76), "RESUME at epoch 4", start(), batch()])
        self.assertEqual(result["epoch"], 4)
        self.assertEqual(result["epochs_completed"], 3)
        self.assertEqual(result["training_images_processed"], 3 * 82929 + 1664)

    def test_restart_discards_abandoned_attempt_and_old_batch_counts(self):
        result = parse([start(), batch(76), "RUNTIME_CHANGE: CPU threads updated", start(), batch()])
        self.assertEqual(result["epoch"], 1)
        self.assertEqual(result["batch"], 26)
        self.assertEqual(result["images_processed"], 1664)
        self.assertEqual(result["training_images_processed"], 1664)

    def test_restart_without_resume_discards_prior_epoch_completion(self):
        result = parse([start(), epoch_result(3), start(), batch(1)])
        self.assertEqual(result["epoch"], 1)
        self.assertEqual(result["epochs_completed"], 0)
        self.assertEqual(result["training_images_processed"], 64)

    def test_final_training_evaluation_is_distinct_from_epoch_13(self):
        result = parse([start(), epoch_result(12), batch(26, 1296, "eval")])
        self.assertEqual(result["epoch"], 12)
        self.assertEqual(result["epochs_completed"], 12)
        self.assertEqual(result["phase"], "final_train")
        self.assertEqual(result["training_images_processed"], 12 * 82929)
        self.assertEqual(result["overall_training_percent"], 100)
        self.assertFalse(result["complete"])

    def test_final_validation_evaluation_has_validation_counts(self):
        result = parse([start(), epoch_result(12), batch(1276, 1296, "eval"),
                        batch(26, 278, "eval")])
        self.assertEqual(result["phase"], "final_val")
        self.assertEqual(result["total_images"], 17771)
        self.assertEqual(result["images_processed"], 1664)
        self.assertEqual(result["training_images_processed"], 12 * 82929)

    def test_complete_status_is_complete_after_final_evaluation(self):
        result = parse([start(), epoch_result(12), "TASK2_COMPLETE saved-run"],
                       status={"state": "complete", "smoke": False})
        self.assertTrue(result["complete"])
        self.assertEqual(result["phase"], "complete")
        self.assertEqual(result["epoch"], 12)
        self.assertEqual(result["epochs_completed"], 12)
        self.assertEqual(result["overall_training_percent"], 100)

    def test_empty_log_does_not_claim_a_first_batch_or_epoch(self):
        result = parse_progress("", CONFIG, AUDIT, {"state": "running"})
        self.assertIsNone(result["epoch"])
        self.assertIsNone(result["batch"])
        self.assertIn(result["images_processed"], (None, 0))
        self.assertFalse(result["complete"])

    def test_new_start_before_first_logged_batch_does_not_reuse_previous_batch(self):
        result = parse([start(), batch(76), start()])
        self.assertIsNone(result["batch"])
        self.assertIn(result["images_processed"], (None, 0))
        self.assertIn(result["training_images_processed"], (None, 0))

    def test_truncated_batch_record_does_not_replace_last_complete_record(self):
        result = parse([start(), batch(1), "  train batch 26/1296, loss="])
        self.assertEqual(result["batch"], 1)
        self.assertEqual(result["images_processed"], 64)

    def test_arbitrary_text_and_incomplete_marker_do_not_fabricate_batches(self):
        result = parse(["traceback content", "EPOCH_RESULT {", "  train batch 26/"])
        self.assertIsNone(result["batch"])
        self.assertIn(result["images_processed"], (None, 0))

    def test_elapsed_time_does_not_extrapolate_unlogged_processed_images(self):
        result = parse([start(), batch(rate=999.9, elapsed=120)])
        self.assertEqual(result["images_processed"], 1664)
        self.assertEqual(result["training_images_processed"], 1664)
        self.assertEqual(result["batch"], 26)

    def test_invalid_or_zero_rate_does_not_supply_a_remaining_time(self):
        result = parse([start(), batch(rate=0)])
        self.assertIn(result["phase_remaining_seconds"], (None,))
        self.assertEqual(result["images_processed"], 1664)

    def test_smoke_training_goal_is_two_batches_not_all_thirteen(self):
        config = {"epochs": 1, "batch_size": 8, "smoke": True}
        result = parse([start(config), batch(1, 13)], config)
        self.assertEqual(result["total_batches"], 2)
        self.assertEqual(result["total_images"], 16)
        self.assertEqual(result["images_processed"], 8)
        self.assertEqual(result["total_training_images"], 16)
        self.assertEqual(result["phase_percent"], 50)

    def test_smoke_validation_uses_fifty_images_and_full_seven_batches(self):
        config = {"epochs": 1, "batch_size": 8, "smoke": True}
        result = parse([start(config), batch(1, 13), batch(1, 7, "eval")], config)
        self.assertEqual(result["phase"], "val")
        self.assertEqual(result["total_batches"], 7)
        self.assertEqual(result["total_images"], 50)
        self.assertEqual(result["images_processed"], 8)
        self.assertEqual(result["training_images_processed"], 16)
        self.assertEqual(result["overall_training_percent"], 100)

    def test_failed_status_retains_last_known_counts_without_marking_complete(self):
        result = parse([start(), batch()], status={"state": "failed", "error": "boom"})
        self.assertEqual(result["state"], "failed")
        self.assertFalse(result["complete"])
        self.assertEqual(result["batch"], 26)
        self.assertEqual(result["images_processed"], 1664)


class RecentSpeedTests(unittest.TestCase):
    @staticmethod
    def observed_epoch_two():
        # These log observations include the two long gaps from the live run.
        # They establish elapsed intervals, not the cause of the interruptions.
        return ["RESUME at epoch 2", start(),
                batch(1051, elapsed=90.1, rate=12.44),
                batch(1076, elapsed=92.3, rate=12.43),
                batch(1101, elapsed=167.5, rate=7.01),
                batch(1126, elapsed=169.2, rate=7.10),
                batch(1151, elapsed=171.4, rate=7.16),
                batch(1176, elapsed=326.6, rate=3.85),
                batch(1201, elapsed=328.2, rate=3.90)]

    def test_reported_speed_is_exposed_as_phase_average(self):
        result = parse([start(), batch()])
        self.assertEqual(result["phase_average_images_per_second"], 11.1)
        self.assertEqual(result["images_per_second"], 11.1)
        self.assertIsNone(result["recent_images_per_second"])
        self.assertEqual(result["recent_speed_samples"], 0)
        self.assertEqual(result["eta_speed_basis"], "phase average")

    def test_recent_speed_uses_observed_image_and_elapsed_differences(self):
        result = parse([start(), batch(1, elapsed=0.1, rate=10),
                        batch(26, elapsed=1.8, rate=11),
                        batch(51, elapsed=3.5, rate=12)])
        recent = 25 * 64 / (1.7 * 60)
        self.assertAlmostEqual(result["recent_images_per_second"], recent)
        self.assertEqual(result["phase_average_images_per_second"], 12)
        self.assertEqual(result["recent_speed_samples"], 2)
        self.assertEqual(result["eta_speed_basis"], "recent interval")
        self.assertAlmostEqual(result["phase_remaining_seconds"], (82929 - 51 * 64) / recent)

    def test_immediately_after_first_long_gap_recent_speed_and_eta_are_unknown(self):
        result = parse(self.observed_epoch_two()[:5])
        self.assertEqual(result["batch"], 1101)
        self.assertEqual(result["phase_average_images_per_second"], 7.01)
        self.assertIsNone(result["recent_images_per_second"])
        self.assertIsNone(result["phase_remaining_seconds"])
        self.assertIsNone(result["eta_speed_basis"])
        self.assertEqual(result["recent_speed_samples"], 0)
        self.assertEqual(result["long_gap_count"], 1)
        self.assertAlmostEqual(result["long_gap_seconds"], 75.2 * 60)

    def test_first_normal_interval_after_gap_restores_recent_speed_and_eta(self):
        result = parse(self.observed_epoch_two()[:6])
        recent = 25 * 64 / (1.7 * 60)
        self.assertAlmostEqual(result["recent_images_per_second"], recent)
        self.assertEqual(result["recent_speed_samples"], 1)
        self.assertEqual(result["eta_speed_basis"], "recent interval")
        self.assertAlmostEqual(result["phase_remaining_seconds"], (82929 - 1126 * 64) / recent)
        self.assertEqual(result["long_gap_count"], 1)

    def test_second_long_gap_resets_the_recent_window_again(self):
        result = parse(self.observed_epoch_two()[:8])
        self.assertEqual(result["batch"], 1176)
        self.assertEqual(result["phase_average_images_per_second"], 3.85)
        self.assertIsNone(result["recent_images_per_second"])
        self.assertIsNone(result["phase_remaining_seconds"])
        self.assertEqual(result["recent_speed_samples"], 0)
        self.assertEqual(result["long_gap_count"], 2)
        self.assertAlmostEqual(result["long_gap_seconds"], (75.2 + 155.2) * 60)

    def test_current_recent_speed_recovers_despite_slow_phase_average(self):
        result = parse(self.observed_epoch_two())
        recent = 25 * 64 / (1.6 * 60)
        self.assertEqual(result["phase_average_images_per_second"], 3.90)
        self.assertAlmostEqual(result["recent_images_per_second"], recent)
        self.assertEqual(result["recent_speed_samples"], 1)
        self.assertAlmostEqual(result["phase_remaining_seconds"], (82929 - 1201 * 64) / recent)
        self.assertEqual(result["phase_long_gap_count"], 2)
        self.assertAlmostEqual(result["phase_long_gap_seconds"], (75.2 + 155.2) * 60)
        self.assertEqual(result["epoch"], 2)
        self.assertEqual(result["batch"], 1201)
        self.assertEqual(result["images_processed"], 1201 * 64)
        self.assertEqual(result["training_images_processed"], 82929 + 1201 * 64)

    def test_validation_resets_recent_baseline_and_phase_gap_totals(self):
        result = parse(self.observed_epoch_two() + [batch(1, 278, "eval", elapsed=0.1, rate=30)])
        self.assertEqual(result["phase"], "val")
        self.assertEqual(result["phase_average_images_per_second"], 30)
        self.assertIsNone(result["recent_images_per_second"])
        self.assertEqual(result["recent_speed_samples"], 0)
        self.assertEqual(result["phase_long_gap_count"], 0)
        self.assertEqual(result["phase_long_gap_seconds"], 0)
        self.assertEqual(result["long_gap_count"], 2)

    def test_epoch_transition_does_not_derive_a_rate_from_the_previous_epoch(self):
        result = parse([start(), batch(1, elapsed=0.1), batch(26, elapsed=1.8),
                        epoch_result(1), batch(1, elapsed=0.1, rate=7)])
        self.assertEqual(result["epoch"], 2)
        self.assertEqual(result["phase_average_images_per_second"], 7)
        self.assertIsNone(result["recent_images_per_second"])
        self.assertEqual(result["recent_speed_samples"], 0)

    def test_restart_drops_recent_speed_and_gap_history_from_previous_attempt(self):
        result = parse(self.observed_epoch_two() + [start(), batch(1, elapsed=0.1, rate=7)])
        self.assertEqual(result["epoch"], 1)
        self.assertIsNone(result["recent_images_per_second"])
        self.assertEqual(result["recent_speed_samples"], 0)
        self.assertEqual(result["long_gap_count"], 0)
        self.assertEqual(result["long_gap_seconds"], 0)

    def test_final_training_to_final_validation_resets_recent_baseline(self):
        result = parse([start(), epoch_result(12), batch(1, 1296, "eval", elapsed=0.1),
                        batch(26, 1296, "eval", elapsed=1.8),
                        batch(1, 278, "eval", elapsed=0.1, rate=30)])
        self.assertEqual(result["phase"], "final_val")
        self.assertIsNone(result["recent_images_per_second"])
        self.assertEqual(result["recent_speed_samples"], 0)
        self.assertEqual(result["phase_average_images_per_second"], 30)

    def test_nonfinite_elapsed_records_do_not_change_confirmed_counts_or_rates(self):
        observations = [start(), batch(1, elapsed=0.1), batch(26, elapsed=1.8)]
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(elapsed=invalid):
                result = parse(observations + [batch(51, elapsed=invalid)])
                self.assertEqual(result["batch"], 26)
                self.assertEqual(result["images_processed"], 26 * 64)
                self.assertAlmostEqual(result["recent_images_per_second"], 25 * 64 / 102)

    def test_zero_elapsed_difference_never_divides_or_fabricates_a_recent_rate(self):
        result = parse([start(), batch(1, elapsed=0), batch(26, elapsed=0)])
        self.assertEqual(result["batch"], 26)
        self.assertEqual(result["images_processed"], 26 * 64)
        self.assertIsNone(result["recent_images_per_second"])
        self.assertEqual(result["recent_speed_samples"], 0)

    def test_zero_reported_average_does_not_hide_a_valid_recent_interval(self):
        result = parse([start(), batch(1, elapsed=0.1, rate=0), batch(26, elapsed=1.8, rate=0)])
        recent = 25 * 64 / 102
        self.assertEqual(result["phase_average_images_per_second"], 0)
        self.assertAlmostEqual(result["recent_images_per_second"], recent)
        self.assertAlmostEqual(result["phase_remaining_seconds"], (82929 - 26 * 64) / recent)


class ProgressViewerTests(unittest.TestCase):
    def test_dashboard_labels_recent_speed_separately_from_average_including_gaps(self):
        result = parse(RecentSpeedTests.observed_epoch_two())
        text = monitor_training.format_progress(result)
        for expected in ("최근 구간 속도 16.67장/초", "현재 단계 평균(긴 공백 포함) 3.90장/초",
                         "최근 구간 기준·추정", "긴 기록 공백 2회"):
            self.assertIn(expected, text)
        self.assertNotIn("최근 구간 속도 3.90", text)

    def test_dashboard_waits_for_a_new_interval_immediately_after_long_gap(self):
        result = parse(RecentSpeedTests.observed_epoch_two()[:8])
        text = monitor_training.format_progress(result)
        for expected in ("최근 구간 속도 측정 중", "이 단계 남은 시간(계산 대기) -",
                         "현재 단계 평균(긴 공백 포함) 3.85장/초", "긴 기록 공백 2회"):
            self.assertIn(expected, text)
        self.assertNotIn("최근 구간 기준·추정", text)

    def test_dashboard_shows_epoch_batch_image_counters_and_cpu_setting(self):
        text = monitor_training.format_progress(parse([start(), batch()]), log_age=4, threads=16)
        for expected in ("Epoch 1/12", "26/1,296", "1,664/82,929", "2.01%", "11.10", "CPU 16"):
            self.assertIn(expected, text)

    def test_dashboard_unknown_progress_does_not_show_fake_zero_percent(self):
        result = parse_progress("", CONFIG, AUDIT, {"state": "running"})
        text = monitor_training.format_progress(result)
        self.assertIn("Epoch -/12", text)
        self.assertNotIn("0.00%", text)
        self.assertNotIn("nan", text.lower())

    def test_dashboard_nonfinite_numbers_render_as_unknown(self):
        result = parse([start(), batch()])
        for key in ("phase_percent", "overall_training_percent", "images_processed", "images_per_second",
                    "recent_images_per_second", "phase_average_images_per_second"):
            result[key] = float("nan")
        result["elapsed_seconds"] = float("inf")
        result["phase_remaining_seconds"] = float("inf")
        text = monitor_training.format_progress(result)
        self.assertNotIn("nan", text.lower())
        self.assertNotIn("inf", text.lower())

    def _run_once(self, *, status=None, raw=False, tail_zero=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "results" / "demo"
            run.mkdir(parents=True)
            fixtures = {"config.json": CONFIG, "data_audit.json": AUDIT,
                        "status.json": status or {"state": "running"},
                        "environment.json": {"cpu_threads": 16}}
            for name, data in fixtures.items():
                (run / name).write_text(json.dumps(data), encoding="utf-8")
            (run / "run.log").write_text(start() + "\n" + batch() + "\n", encoding="utf-8")
            args = ["--run-name", "demo", "--once"]
            if raw:
                args.append("--raw")
            if tail_zero:
                args.extend(["--tail", "0"])
            output = io.StringIO()
            with mock.patch.object(monitor_training, "HERE", root), redirect_stdout(output):
                code = monitor_training.main(args)
            return code, output.getvalue()

    def test_once_dashboard_reads_full_log_even_when_initial_tail_is_disabled(self):
        code, text = self._run_once(tail_zero=True)
        self.assertEqual(code, 0)
        self.assertIn("Epoch 1/12", text)
        self.assertIn("26/1,296", text)
        self.assertIn("1,664/82,929", text)
        self.assertNotIn("train batch 26/1296", text)

    def test_raw_option_retains_original_records_as_well_as_dashboard(self):
        code, text = self._run_once(raw=True)
        self.assertEqual(code, 0)
        self.assertIn("train batch 26/1296", text)
        self.assertIn("Epoch 1/12", text)

    def test_failed_once_returns_nonzero_and_shows_saved_progress(self):
        code, text = self._run_once(status={"state": "failed", "error": "boom"})
        self.assertEqual(code, 1)
        self.assertIn("failed", text)
        self.assertIn("26/1,296", text)


if __name__ == "__main__":
    unittest.main()
