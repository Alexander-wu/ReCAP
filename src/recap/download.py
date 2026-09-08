"""Download a matched checkpoint pair and action ranges from a Hub bundle."""
from pathlib import Path

from .assets import safe_path, verify

DATASETS = ('rt1', 'calvin', 'libero', 'bridge')


def plan_download(manifest, dataset, repo_id=None, revision=None):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    distribution = manifest.get('distribution', {})
    repo_id = repo_id or distribution.get('repo_id')
    revision = revision or distribution.get('revision')
    if not repo_id or not revision:
        raise ValueError('No published Hub bundle is configured. Supply its --repo-id '
                         'and --revision, or obtain an updated release manifest.')
    entries = [entry for suffix in ('_tokenizer', '_world', '_action_ranges')
               for entry in manifest['models'][dataset + suffix]['files']]
    for entry in entries:
        path = Path(entry['path'])
        if path.is_absolute() or '..' in path.parts or path.parts[0] != 'weights':
            raise ValueError(f'Invalid bundle path: {path}')
    return {'repo_id': repo_id, 'revision': revision, 'files': entries}


def download_bundle(root, plan, fetch=None):
    root = Path(root).resolve()
    # Resolve all destinations before any network operation (including symlinks).
    for entry in plan['files']:
        safe_path(root, entry['path'])
    if fetch is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError('Install the assets extra: pip install ".[assets]"') from exc
        fetch = hf_hub_download
    for entry in plan['files']:
        if not verify(root, [entry]):
            continue
        fetch(repo_id=plan['repo_id'], revision=plan['revision'],
              filename=entry['path'], local_dir=str(root), repo_type='model')
    failures = verify(root, plan['files'])
    if failures:
        raise RuntimeError('Downloaded asset verification failed: ' + '; '.join(failures))
    return {'root': str(root), 'checked': len(plan['files']), 'failures': []}
