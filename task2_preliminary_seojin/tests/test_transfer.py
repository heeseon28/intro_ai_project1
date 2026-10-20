"""Check transfer integrity with synthetic fixtures, not the real dataset."""
import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from test_pipeline import fixture
from transfer_data import INFO, pack, unpack


class TransferTests(unittest.TestCase):
    def test_round_trip_only_train_val_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, splits, parts = fixture(root, 50)
            archive, target = root/'bundle.zip', root/'received'
            info = pack(source, splits, archive)
            self.assertEqual(info['image_count'], 150)
            self.assertFalse(info['test_images_included'])
            with zipfile.ZipFile(archive) as bundle:
                self.assertEqual(len(bundle.namelist()), 151)
                self.assertFalse(any(row['rel_path'] in bundle.namelist() for row in parts['test']))
            received = unpack(archive, splits, target)
            self.assertTrue(received['all_extracted_image_md5_verified'])
            for row in parts['train']+parts['val']:
                self.assertEqual(hashlib.md5((target/row['rel_path']).read_bytes()).hexdigest(), row['md5'])
                self.assertTrue((source/row['rel_path']).is_file())
            with self.assertRaises(FileExistsError):
                pack(source, splits, archive)
            with self.assertRaises(FileExistsError):
                unpack(archive, splits, target)

    def test_unexpected_archive_path_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, splits, _ = fixture(root, 50)
            archive, target = root/'bundle.zip', root/'received'
            pack(source, splits, archive)
            with zipfile.ZipFile(archive, 'a') as bundle:
                bundle.writestr('../outside.txt', 'do not extract')
            with self.assertRaisesRegex(ValueError, 'Archive members differ'):
                unpack(archive, splits, target)
            self.assertFalse(target.exists())
            self.assertFalse((root/'outside.txt').exists())

    def test_changed_server_manifests_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, splits, _ = fixture(root, 50)
            archive, target = root/'bundle.zip', root/'received'
            pack(source, splits, archive)
            with (splits/'val.csv').open('a', encoding='utf-8') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'split CSVs differ'):
                unpack(archive, splits, target)
            self.assertFalse(target.exists())

    def test_image_tampering_never_publishes_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, splits, parts = fixture(root, 50)
            archive, damaged, target = root/'bundle.zip', root/'damaged.zip', root/'received'
            pack(source, splits, archive)
            changed = parts['val'][0]['rel_path']
            with zipfile.ZipFile(archive) as original, zipfile.ZipFile(damaged, 'w') as replacement:
                for entry in original.infolist():
                    content = original.read(entry.filename)
                    if entry.filename == changed:
                        content = bytes([content[0] ^ 1]) + content[1:]
                    replacement.writestr(entry.filename, content)
            with self.assertRaisesRegex(ValueError, 'Image MD5'):
                unpack(damaged, splits, target)
            self.assertFalse(target.exists())
            self.assertFalse(list(root.glob('.task2_unpack_*')))


if __name__ == '__main__':
    unittest.main()
