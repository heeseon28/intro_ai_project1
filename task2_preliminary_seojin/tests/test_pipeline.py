"""Temporary synthetic fixtures check plumbing; these are NOT VMMRdb results."""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from preflight import inspect_inputs


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root, nclasses):
    from PIL import Image
    data, splits = root / '합성 이미지 with spaces', root / 'splits'
    data.mkdir()
    splits.mkdir()
    ci = []
    parts = {name: [] for name in ('train', 'val', 'test')}
    for idx in range(nclasses):
        label = f'fixture/model_{idx}'
        ci.append(dict(model_idx=idx, make_model=label, n_train=2, n_val=1, n_test=1))
        for name, size in (('train', 2), ('val', 1), ('test', 1)):
            for k in range(size):
                rel = f'model_{idx}/{name}_{k}.png'
                if name == 'test':
                    # Test images intentionally do not exist; only their metadata may be read.
                    digest = hashlib.md5(rel.encode()).hexdigest()
                else:
                    target = data / rel
                    target.parent.mkdir(exist_ok=True)
                    Image.new('RGB', (32, 40), (idx*4, k*100, 25 if name=='train' else 225)).save(target)
                    digest = hashlib.md5(target.read_bytes()).hexdigest()
                parts[name].append(dict(rel_path=rel, md5=digest, make_model=label, model_idx=idx))
    write_rows(splits / 'class_info.csv', ci)
    for name, rows in parts.items():
        write_rows(splits / f'{name}.csv', rows)
    return data, splits, parts


class ManifestTests(unittest.TestCase):
    def test_real_manifest_audit_records_blocked_data_without_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cmd = [sys.executable, str(HERE/'preflight.py'), '--data-root', str(root/'absent'),
                   '--output-dir', str(root/'audit')]
            done = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=60)
            self.assertEqual(done.returncode, 2, done.stdout+done.stderr)
            audit = json.loads((root/'audit/data_audit.json').read_text(encoding='utf-8'))
            self.assertTrue(audit['manifest_checks_passed'])
            self.assertEqual(audit['nclasses'], 50)
            self.assertEqual(audit['split_sizes'], {'train':82929,'val':17771,'test':17771})
            self.assertEqual(audit['state'], 'blocked_missing_data')
            self.assertFalse((root/'audit/metrics.json').exists())

    def test_test_images_not_needed_and_training_hash_mismatch_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            data, splits, parts = fixture(Path(temp), 2)
            audit, _, _ = inspect_inputs(splits, data, expected_classes=2)
            self.assertEqual(audit['state'], 'ready', audit['errors'])
            self.assertEqual(audit['train_hashes_verified'], 4)
            (data/parts['train'][0]['rel_path']).write_bytes(b'changed image')
            audit, _, _ = inspect_inputs(splits, data, expected_classes=2)
            self.assertEqual(audit['state'], 'invalid_images')

    def test_cross_split_hash_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            data, splits, parts = fixture(Path(temp), 2)
            parts['val'][0]['md5'] = parts['train'][0]['md5']
            write_rows(splits/'val.csv', parts['val'])
            audit, _, _ = inspect_inputs(splits, data, expected_classes=2)
            self.assertEqual(audit['state'], 'invalid_manifests')
            self.assertIn('leakage', audit['errors'][0])

    def test_unsafe_manifest_paths_rejected_on_all_platforms(self):
        with tempfile.TemporaryDirectory() as temp:
            data, splits, parts = fixture(Path(temp), 2)
            for bad in ('../outside.png', 'C:/outside.png', '/outside.png', '..\\outside.png'):
                with self.subTest(path=bad):
                    parts['train'][0]['rel_path'] = bad
                    write_rows(splits/'train.csv', parts['train'])
                    audit, _, _ = inspect_inputs(splits, data, expected_classes=2)
                    self.assertEqual(audit['state'], 'invalid_manifests')


class PipelineTests(unittest.TestCase):
    def test_smoke_artifacts_metrics_resume_and_overwrite_protection(self):
        import numpy as np
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, splits, _ = fixture(root, 50)
            cmd = [sys.executable, str(HERE/'train_baseline.py'), '--data-root', str(data),
                   '--split-dir', str(splits), '--output-dir', str(root/'results'),
                   '--run-name', 'synthetic_verification', '--smoke', '--device', 'cpu']
            def run(extra=()):
                return subprocess.run(cmd+list(extra), capture_output=True, text=True,
                                      encoding='utf-8', timeout=240)
            done = run()
            self.assertEqual(done.returncode, 0, done.stdout+done.stderr)
            out = root/'results/synthetic_verification_smoke'
            expected = ['config.json','environment.json','data_audit.json','history.csv',
                        'learning_curves.png','metrics.json','per_class_metrics.csv',
                        'difficult_classes.csv','confusion_matrix_counts.csv','confusion_matrix.png',
                        'validation_predictions.csv','validation_examples.png','report.md','report_facts.md',
                        'train_source.py','class_info.csv','status.json','run.log',
                        'checkpoints/best.pt','checkpoints/last.pt']
            for name in expected:
                self.assertTrue((out/name).is_file(), name)
            summary = json.loads((out/'metrics.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['smoke_test_only'])
            self.assertFalse(summary['test_evaluated'])
            self.assertFalse(summary['readme_conditions_met'])
            self.assertEqual(summary['best_epoch'], 1)
            predictions = pd.read_csv(out/'validation_predictions.csv')
            self.assertEqual(len(predictions), 50)
            y, p = predictions.model_idx, predictions.prediction_idx
            measured = summary['validation_at_best_checkpoint']
            self.assertAlmostEqual(measured['accuracy'], accuracy_score(y, p))
            self.assertAlmostEqual(measured['macro_f1'], f1_score(y, p, average='macro', labels=range(50), zero_division=0))
            matrix = pd.read_csv(out/'confusion_matrix_counts.csv', index_col=0).to_numpy()
            self.assertEqual(int(matrix.sum()), 50)
            self.assertAlmostEqual(np.trace(matrix)/matrix.sum(), measured['accuracy'])
            history = pd.read_csv(out/'history.csv')
            self.assertTrue({'train_loss','val_loss','train_accuracy','val_accuracy','train_macro_f1','val_macro_f1'} <= set(history.columns))
            checkpoint_hash = hashlib.sha256((out/'checkpoints/best.pt').read_bytes()).hexdigest()
            done = run(['--resume'])
            self.assertEqual(done.returncode, 0, done.stdout+done.stderr)
            self.assertEqual(summary, json.loads((out/'metrics.json').read_text(encoding='utf-8')))
            self.assertEqual(checkpoint_hash, hashlib.sha256((out/'checkpoints/best.pt').read_bytes()).hexdigest())
            before = (out/'status.json').read_bytes()
            self.assertNotEqual(run().returncode, 0)
            self.assertEqual(before, (out/'status.json').read_bytes())
            # Changed manifests must be rejected without changing saved completed results.
            with (splits/'val.csv').open('a', encoding='utf-8') as stream:
                stream.write('\n')
            rejected = run(['--resume'])
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('existing results were preserved', rejected.stderr)
            self.assertEqual(before, (out/'status.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
