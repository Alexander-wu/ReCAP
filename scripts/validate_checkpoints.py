"""Verify and strictly load every robot checkpoint on CPU, without sampling."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys

os.environ.update(USE_TF='0', USE_FLAX='0', CUDA_VISIBLE_DEVICES='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Parent of weights/')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import recap
    from recap.assets import verify
    manifest = json.loads(Path(args.manifest).read_text())
    entries = [f for m in manifest['models'].values() for f in m['files']]
    failures = verify(args.root, entries)
    if failures:
        raise RuntimeError('; '.join(failures))
    sys.path.insert(0, str(Path(recap.__file__).parent / 'reference'))
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from safetensors.torch import load_file
    from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ
    torch.set_num_threads(4)
    report = {'device': 'cpu', 'files_verified': len(entries), 'models': [],
              'scope': 'Strict state-dict loading, world forward and range contracts; no GPU rollout'}
    for dataset in ('rt1', 'calvin', 'libero', 'bridge'):
        base = Path(args.root) / 'weights'
        world_dir = base / (dataset + '_world')
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(world_dir, local_files_only=True),
            torch_dtype=torch.float32, attn_implementation='sdpa')
        model.load_state_dict(load_file(str(world_dir / 'model.safetensors')), strict=True)
        model.eval()
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([[4375, 4376, 0, 8750]])).logits
        if not torch.isfinite(logits).all() or logits.shape[-1] != model.config.vocab_size:
            raise RuntimeError('Invalid world forward: ' + dataset)
        report['models'].append({'name': dataset + '_world', 'strict_load': True,
                                 'parameters': sum(p.numel() for p in model.parameters()),
                                 'forward_finite': True, 'vocab_size': model.config.vocab_size})
        del model, logits
        gc.collect()
        tokenizer_dir = base / (dataset + '_tokenizer')
        config = json.loads((tokenizer_dir / 'config.json').read_text())
        tokenizer = CompressiveVQModelFSQ.from_config(config)
        tokenizer.load_state_dict(load_file(str(tokenizer_dir / 'diffusion_pytorch_model.safetensors')), strict=True)
        report['models'].append({'name': dataset + '_tokenizer', 'strict_load': True,
                                 'parameters': sum(p.numel() for p in tokenizer.parameters())})
        del tokenizer
        ranges = torch.load(base / dataset / 'action_ranges.pth', map_location='cpu', weights_only=True)
        if ranges.shape != (13, 2) or not torch.isfinite(ranges).all() or not (ranges[:, 1] >= ranges[:, 0]).all():
            raise RuntimeError('Invalid action range table: ' + dataset)
        gc.collect()
        print('Validated ' + dataset, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
