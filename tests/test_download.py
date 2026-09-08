import hashlib
from pathlib import Path
import tempfile
import unittest

from recap.download import plan_download, download_bundle


class DownloadTests(unittest.TestCase):
    def manifest(self):
        content = b'fixture'
        return {'models': {'rt1' + suffix: {'files': [{
            'path': 'weights/rt1' + suffix + '/asset', 'bytes': len(content),
            'sha256': hashlib.sha256(content).hexdigest()}]}
            for suffix in ('_tokenizer', '_world', '_action_ranges')}}

    def test_requires_explicit_host_and_revision(self):
        with self.assertRaises(ValueError):
            plan_download(self.manifest(), 'rt1')
        with self.assertRaises(ValueError):
            plan_download(self.manifest(), 'unknown', 'owner/model', 'v1')

    def test_fetches_complete_pair_verifies_and_resumes(self):
        plan = plan_download(self.manifest(), 'rt1', 'owner/model', 'v1')
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            def fetch(**kwargs):
                calls.append(kwargs)
                p = Path(kwargs['local_dir']) / kwargs['filename']
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'fixture')
            self.assertEqual(download_bundle(tmp, plan, fetch)['checked'], 3)
            self.assertEqual(len(calls), 3)
            download_bundle(tmp, plan, fetch)
            self.assertEqual(len(calls), 3)
            (Path(tmp) / plan['files'][0]['path']).write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):
                download_bundle(tmp, plan, lambda **kwargs: None)

    def test_rejects_traversal_and_symlink_escape_before_fetch(self):
        manifest = self.manifest()
        manifest['models']['rt1_world']['files'][0]['path'] = 'weights/../../escape'
        with self.assertRaises(ValueError):
            plan_download(manifest, 'rt1', 'owner/model', 'v1')
        plan = plan_download(self.manifest(), 'rt1', 'owner/model', 'v1')
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as external:
            (Path(tmp) / 'weights').symlink_to(external, target_is_directory=True)
            with self.assertRaises(ValueError):
                download_bundle(tmp, plan, lambda **kwargs: self.fail('Network must not run'))
