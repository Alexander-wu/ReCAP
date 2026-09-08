"""Paired Bridge 100k reference evaluation on a selected CUDA device.

Release adaptation: explicit action-range and device arguments; sampling and
the original one-observation protocol are preserved.
"""
import os
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys,time,json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw

from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ
from transformers import AutoConfig,AutoModelForCausalLM
from safetensors.torch import load_file
import piqa,lpips

import argparse
parser=argparse.ArgumentParser(description="Bridge paper FP32 paired evaluation (one observed frame)")
for name in ['output-dir','world-model','tokenizer-model','data-root']:
    parser.add_argument('--'+name,required=True)
parser.add_argument('--cases',nargs='+',help='Explicit validation episode basenames; default first three eligible clips')
parser.add_argument('--action-ranges',required=True,help='Matched Bridge action_ranges.pth')
parser.add_argument('--device',default='cuda:0',help='Visible CUDA device, e.g. cuda:1')
args=parser.parse_args()
torch.set_num_threads(4)
device=torch.device(args.device)
if device.type != 'cuda' or not torch.cuda.is_available():
    raise RuntimeError('The Bridge FP32 reference requires a CUDA device')
torch.cuda.set_device(device)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
out=Path(args.output_dir)
out.mkdir(exist_ok=False)
checkpoint=Path(args.world_model)/'model.safetensors'
tokpath=args.tokenizer_model
data=Path(args.data_root)
model=AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(args.world_model),torch_dtype=torch.float32,attn_implementation='sdpa')
model.load_state_dict(load_file(str(checkpoint)),strict=True);model=model.eval().to(device)
tok=CompressiveVQModelFSQ.from_pretrained(tokpath).eval().to(device)
ranges=torch.load(args.action_ranges,map_location='cpu',weights_only=True).to(device)
ssim=piqa.SSIM(window_size=11,sigma=1.5,n_channels=3,reduction='none').eval().to(device)
perc=lpips.LPIPS(net='vgg').eval().to(device)
report={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'tokenizer':tokpath,'device':torch.cuda.get_device_name(device),'precision':'FP32; TF32 disabled',
        'observed_frames':1,'anchor':'duplicate first observation as initial dynamic block',
        'prediction_frames':32,'actions':'ground-truth actions; action_t after visual_t',
        'sampling':{'seed':20260906,'temperature':1.,'top_k':100},
        'recap':{'window':6,'anchor_blocks':1,'position_rebase':True,'pixel_reencoding':False},
        'comparison_scope':'fixed validation clips, 1 seed; not full dataset or source-independent estimates',
        'cross_run_note':'GPU sampling differs from earlier CPU 50k; compare full vs ReCAP within this run',
        'cases':[]}
selected=[]
for f in ([data / name for name in args.cases] if args.cases else sorted(data.glob('val_eps_*.npz'))):
    with np.load(f) as z:
        if len(z['image'])>=33:
            selected.append((f,z['image'][:33].copy(),z['action'][:33].copy(),str(z['instruction'].item())))
    if not args.cases and len(selected)==3:break
assert len(selected)==(len(args.cases) if args.cases else 3)
print('SELECTION='+json.dumps([(str(x[0]),x[3]) for x in selected]),flush=True)

@torch.inference_mode()
def rollout(ctx,dyn,act,method,case):
    torch.manual_seed(20260906)
    anchor=torch.cat([ctx.flatten()+4375,dyn.flatten(),act[0]])[None].long()
    pending=anchor;cache=None;recent=[];generated=[];lengths=[]
    started=time.monotonic()
    for t in range(1,33):
        if method=='recap' and t>7:
            pending=torch.cat([anchor]+recent,1);cache=None
        lengths.append(1280+93*(1+(min(t-1,6) if method=='recap' else t-1)))
        frame=[]
        for _ in range(80):
            result=model(input_ids=pending,past_key_values=cache,use_cache=True)
            cache=result.past_key_values
            values,indices=torch.topk(result.logits[:,-1,:],100,dim=-1)
            choice=torch.multinomial(torch.softmax(values,dim=-1),1)
            pending=indices.gather(-1,choice);frame.append(pending)
        ids=torch.cat(frame,1);generated.append(ids)
        pending=torch.cat([pending,act[t][None]],1)
        recent.append(torch.cat([ids,act[t][None]],1));recent=recent[-6:]
        if t%8==0:print(f'PROGRESS case={case} method={method} frame={t}/32 seconds={time.monotonic()-started:.1f}',flush=True)
    raw=torch.stack(generated,1)
    bad=int(((raw<0)|(raw>=4375)).sum())
    images=[]
    for start in range(0,32,2):
        images.append(tok.detokenize(ctx,raw[:,start:start+2].clamp(0,4374))[:,1:].float().clamp(0,1))
    torch.cuda.synchronize()
    return torch.cat(images,1)[0],raw,{'out_of_range_tokens':bad,'seconds':time.monotonic()-started,'prompt_lengths':lengths}

@torch.inference_mode()
def metrics(pred,truth):
    values={'psnr':[],'ssim':[],'lpips':[]}
    for start in range(0,32,2):
        x=pred[start:start+2];y=truth[start:start+2]
        values['psnr']+=(-10*(x-y).square().mean((1,2,3)).clamp_min(1e-12).log10()).tolist()
        values['ssim']+=ssim(x,y).tolist()
        values['lpips']+=perc(x*2-1,y*2-1).flatten().tolist()
    return {str(h):{k:float(np.mean(v[:h])) for k,v in values.items()} for h in [8,16,32]}

def export(case,images,full,recap):
    arrays=[images,full,recap];frames=[]
    for t in range(33):
        canvas=Image.new('RGB',(720,224),'white');draw=ImageDraw.Draw(canvas)
        for col,(label,arr) in enumerate(zip(['Ground truth','100k Full history','100k ReCAP W=6'],arrays)):
            canvas.paste(Image.fromarray(arr[t]).resize((240,192),Image.Resampling.LANCZOS),(240*col,32))
            draw.text((240*col+5,4),label,fill='black');draw.text((240*col+5,17),f'frame {t}',fill='black')
        frames.append(canvas)
    frames[0].save(out/f'case{case}_comparison.gif',save_all=True,append_images=frames[1:],duration=160,loop=0,optimize=True)
    strip=Image.new('RGB',(720,224*5),'white')
    for row,t in enumerate([0,8,16,24,32]):strip.paste(frames[t],(0,row*224))
    strip.save(out/f'case{case}_keyframes.png')

for ci,(file,images,actions,instruction) in enumerate(selected):
    gt=torch.from_numpy(images).permute(0,3,1,2).float().to(device)/255
    ac=torch.from_numpy(actions).float().to(device)
    act=torch.floor(((ac-ranges[:,0])/(ranges[:,1]-ranges[:,0]+1e-8)).clamp(0,1)*256).clamp(0,255).long()+8750
    with torch.inference_mode():ctx,dyn=tok.tokenize(gt[:1].repeat(2,1,1,1)[None])
    full,full_ids,full_meta=rollout(ctx,dyn,act,'full_history',ci)
    recap,recap_ids,recap_meta=rollout(ctx,dyn,act,'recap',ci)
    assert torch.equal(full_ids[:,:7],recap_ids[:,:7]),'Pre-eviction predictions differ'
    with torch.inference_mode():
        true_ctx,true_dyn=tok.tokenize(gt[None]);parts=[]
        for start in range(0,32,2):parts.append(tok.detokenize(true_ctx,true_dyn[:,start:start+2])[:,1:].float().clamp(0,1))
        reconstruction=torch.cat(parts,1)[0]
    scores={name:metrics(pred,gt[1:]) for name,pred in [('full_history',full),('recap',recap),('repeat_first',gt[:1].expand(32,-1,-1,-1)),('tokenizer_reconstruction',reconstruction)]}
    case={'case':ci,'source':str(file),'instruction':instruction,'metrics':scores,
          'first_7_frames_identical':True,'full_history':full_meta,'recap':recap_meta}
    report['cases'].append(case)
    def array(pred):return (torch.cat([gt[:1],pred]).permute(0,2,3,1).cpu().numpy()*255).round().astype(np.uint8)
    full_array,recap_array=array(full),array(recap)
    np.savez_compressed(out/f'case{ci}_frames.npz',ground_truth=images,full_history=full_array,recap=recap_array,full_tokens=full_ids.cpu().numpy(),recap_tokens=recap_ids.cpu().numpy(),actions=actions)
    export(ci,images,full_array,recap_array)
    (out/'report.partial.json').write_text(json.dumps(report,indent=2))
    print('CASE_RESULT='+json.dumps(case),flush=True)
report['aggregate']={name:{str(h):{k:float(np.mean([c['metrics'][name][str(h)][k] for c in report['cases']])) for k in ['psnr','ssim','lpips']} for h in [8,16,32]} for name in ['full_history','recap','repeat_first','tokenizer_reconstruction']}
(out/'report.json').write_text(json.dumps(report,indent=2))
print('DONE='+json.dumps(report['aggregate']),flush=True)
