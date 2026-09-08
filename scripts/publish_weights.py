"""Publish an explicitly selected, verified robot bundle; dry-run by default."""
import argparse
import json
from pathlib import Path

from recap.assets import safe_path, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Bundle root containing weights/')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest = json.loads(Path(args.manifest).read_text())
    entries = [f for m in manifest['models'].values() for f in m['files']]
    failures = verify(root, entries)
    if failures:
        raise SystemExit('\n'.join(failures))
    paths = [f['path'] for f in entries]
    # Never upload arbitrary staging contents, credentials, or training states.
    cards = [f'weights/{key}/README.md' for key in manifest['models'] if not key.endswith('_action_ranges')]
    required = ['README.md', 'LICENSE', 'weights_manifest.json', 'THIRD_PARTY_NOTICES.md', *cards]
    required += [p.relative_to(root).as_posix() for p in sorted((root / 'third_party').glob('*LICENSE'))]
    for path in required:
        if not safe_path(root, path).is_file():
            raise FileNotFoundError(f'Missing bundle documentation: {path}')
    paths += required
    plan = {'repo_id': args.repo_id, 'files': sorted(paths),
            'asset_bytes': sum(f['bytes'] for f in entries), 'upload': args.upload}
    print(json.dumps(plan, indent=2))
    if not args.upload:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    identity = api.whoami()
    print('Authenticated as ' + identity['name'])
    api.create_repo(args.repo_id, repo_type='model', exist_ok=True)
    result = api.upload_folder(repo_id=args.repo_id, folder_path=str(root),
                               allow_patterns=paths, repo_type='model',
                               commit_message='Release verified ReCAP robot reference checkpoints')
    print(json.dumps({'repo_id': args.repo_id, 'revision': result.oid, 'commit_url': result.commit_url}))


if __name__ == '__main__':
    main()
