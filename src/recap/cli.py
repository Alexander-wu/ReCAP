"""Portable launcher for the preserved paper reference implementations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .assets import verify

ROOT = Path(__file__).resolve().parents[2]
BACKENDS = {'compare':'infer_compare.py', 'bridge':'bridge_pair.py',
            'rq2':'evaluate_rq2_dataset.py', 'rq3':'evaluate_rq3.py',
            'attention':'probe_rq3_attention.py', 'latency':'benchmark_rq4_single.py',
            'concurrency':'benchmark_rq4_concurrency.py', 'window-sweep':'evaluate_rq4_w_sweep.py'}

def build_command(config, asset_root, data_root, output):
    allowed = {'schema_version','dataset','backend','observed_frames','horizon','arguments','notes'}
    if set(config) - allowed: raise ValueError('unknown configuration keys')
    if config.get('schema_version') != 1: raise ValueError('unsupported schema_version')
    backend = config['backend']
    if backend not in BACKENDS: raise ValueError('unknown backend')
    if backend == 'bridge' and (config['observed_frames'] != 1 or config['horizon'] != 32):
        raise ValueError('Bridge reference requires one observation and horizon 32')
    if backend == 'compare' and config['observed_frames'] != 2:
        raise ValueError('vLLM paper comparison requires two observations')
    args = config['arguments']
    if not isinstance(args,list) or not all(isinstance(x,str) for x in args):
        raise ValueError('arguments must be a list of strings')
    variables = {'assets':str(Path(asset_root).resolve()),'data':str(Path(data_root).resolve()),'output':str(Path(output).resolve())}
    args = [a.format(**variables) for a in args]
    if args.count('--output-dir') != 1 or args[args.index('--output-dir')+1] != variables['output']:
        raise ValueError('exactly one output-dir matching the launcher output is required')
    if backend == 'compare' and ('--num-frames' not in args or int(args[args.index('--num-frames')+1]) != config['horizon']):
        raise ValueError('horizon and num-frames must agree')
    # Never pass an existing output directory to historical scripts that overwrite files.
    if Path(output).exists(): raise FileExistsError(f'Use a new output directory: {output}')
    command = [sys.executable,str(Path(__file__).parent/'reference'/BACKENDS[backend]),*args]
    return command

def main(argv=None):
    parser=argparse.ArgumentParser(prog='recap')
    subs=parser.add_subparsers(dest='command',required=True)
    run=subs.add_parser('run');run.add_argument('--config',required=True);run.add_argument('--assets',required=True);run.add_argument('--data',required=True);run.add_argument('--output',required=True);run.add_argument('--dry-run',action='store_true');run.add_argument('--verify-manifest')
    check=subs.add_parser('verify');check.add_argument('--root',required=True);check.add_argument('--manifest',required=True)
    download=subs.add_parser('download',help='Fetch and verify a matched robot asset bundle')
    download.add_argument('--dataset',required=True,choices=['rt1','calvin','libero','bridge'])
    download.add_argument('--root',default='.')
    download.add_argument('--manifest',default='weights_manifest.json')
    download.add_argument('--repo-id');download.add_argument('--revision')
    download.add_argument('--dry-run',action='store_true')
    doctor=subs.add_parser('doctor')
    args=parser.parse_args(argv)
    if args.command=='download':
        from .download import plan_download, download_bundle
        plan=plan_download(json.loads(Path(args.manifest).read_text()),args.dataset,args.repo_id,args.revision)
        print(json.dumps(plan if args.dry_run else download_bundle(args.root,plan),indent=2))
        return
    if args.command=='doctor':
        import importlib.metadata
        packages={}
        for name in ['torch','transformers','diffusers','vllm','lpips','piqa','numpy']:
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]='not installed'
        print(json.dumps({'python':sys.version,'packages':packages},indent=2));return
    if args.command=='verify':
        manifest=json.loads(Path(args.manifest).read_text())
        entries=manifest.get('files')
        if entries is None:entries=[f for m in manifest['models'].values() for f in m['files']]
        failures=verify(args.root,entries)
        print(json.dumps({'checked':len(entries),'failures':failures},indent=2))
        if failures:raise SystemExit(1)
        return
    config=json.loads(Path(args.config).read_text())
    cmd=build_command(config,args.assets,args.data,args.output)
    if args.dry_run:print(json.dumps({'command':cmd,'protocol':config},indent=2));return
    env=os.environ.copy();env.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']=str(Path(__file__).parent/'reference')+os.pathsep+env.get('PYTHONPATH','')
    # Local paths only: fail before GPU allocation if a model/dataset file is absent.
    for flag in ['--input-npz','--tokenizer-model','--world-model','--action-ranges','--data-root']:
        if flag in cmd:
            p=Path(cmd[cmd.index(flag)+1])
            if not p.exists():raise FileNotFoundError(str(p))
    from .data import validate_episode
    if '--input-npz' in cmd:
        validate_episode(cmd[cmd.index('--input-npz')+1],config['dataset'],config['observed_frames'],config['horizon'])
    if args.verify_manifest:
        manifest=json.loads(Path(args.verify_manifest).read_text())
        dataset=config['dataset']
        # Manifest paths are relative to the parent of weights/.
        entries=[dict(f,path=str(Path(f['path']).relative_to('weights'))) for name in [dataset+'_tokenizer',dataset+'_world',dataset+'_action_ranges'] for f in manifest['models'][name]['files']]
        failures=verify(Path(args.assets).resolve(),entries)
        if failures:raise RuntimeError('Asset integrity failure: '+ '; '.join(failures))
        expected={'--tokenizer-model':dataset+'_tokenizer','--world-model':dataset+'_world',
                  '--action-ranges':dataset+'/action_ranges.pth'}
        for flag,relative in expected.items():
            if flag in cmd and Path(cmd[cmd.index(flag)+1]).resolve()!=(Path(args.assets)/relative).resolve():
                raise ValueError('Model path does not match verified dataset pair')
    subprocess.run(cmd,env=env,check=True)
    dest=Path(args.output)
    if dest.is_dir():
        (dest/'release_launch.json').write_text(json.dumps({'config':config,'argv':cmd,'python':sys.version},indent=2))

if __name__=='__main__':main()
