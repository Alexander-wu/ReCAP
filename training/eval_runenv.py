import copy
import os
import warnings

current_path = os.getcwd()
print("current_path is: ", current_path)

import numpy as np
import torch
from tqdm import tqdm, trange
import time
import sys

import argparse
import json
import logging
import math
import os
from pathlib import Path
import imageio

from vllm import LLM, SamplingParams

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from tqdm.auto import tqdm

from safetensors.torch import load_file
import transformers
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    SchedulerType,
    get_scheduler,
)
from transformers.utils.versions import require_version

from ivideogpt.utils.video_metric import Evaluator, FeatureStats
from ivideogpt.tokenizer import CNNFSQModel256
from ivideogpt.ctx_tokenizer import CompressiveVQModelFSQ
from ivideogpt.data import *
from ivideogpt.processor import *
from rt1_inference import RT1Inference

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.39.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())


def get_dataloaders(args):
    # DataLoaders creation:
    augmentation_args = {
        'brightness': [0.9, 1.1],
        'contrast': [0.9, 1.1],
        'saturation': [0.9, 1.1],
        'hue': [-0.05, 0.05],
        # 'random_resized_crop_scale': (0.8, 1.0),
        # 'random_resized_crop_ratio': (0.9, 1.1),
        'random_resized_crop_scale': (1.0, 1.0),
        'random_resized_crop_ratio': (1.25, 1.25),
        'no_aug': args.no_aug,
    }
    segment_args = {
        'random_selection': False,
        'segment_length': args.segment_length,
        'context_length': None,
        'stepsize': args.video_stepsize,
        'segment_horizon': None,
        'random_ctx_frame': 'ctx' in args.processor_type,
        'task_instruction': args.task_instruction,
        'from_start': True,
    }
    if args.resolution_width is None:
        resolution = [args.resolution, args.resolution]
    else:
        resolution = [args.resolution, args.resolution_width]
    train_dataloader = SimpleRoboticDataLoaderv2(
        parent_dir=args.dataset_path,
        datasets=DATASET_NAMED_MIXES[args.oxe_data_mixes_type],
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
        train=True,
        maxsize=args.dataset_size,
        image_size=resolution,
        **augmentation_args,
        **segment_args,
        load_action=True,
    )
    if args.use_eval_dataset:
        assert len(DATASET_NAMED_MIXES[args.oxe_data_mixes_type]) == 1
        eval_dataloader = EvalDataLoader(
            parent_dir=args.dataset_path,
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.dataloader_num_workers,
            image_size=resolution,
            segment_length=args.segment_length,
            load_action=True,
            random_ctx_frame='ctx' in args.processor_type,
            random_start_frame=True,
        )
    else:
        eval_dataloader = SimpleRoboticDataLoaderv2(
            parent_dir=args.dataset_path,
            datasets=DATASET_NAMED_MIXES[args.oxe_data_mixes_type],
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.dataloader_num_workers,
            train=False,
            image_size=resolution,
            **augmentation_args,
            **segment_args,
            load_action=True,
        )
    return train_dataloader, eval_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument("--config_name", type=str, default="configs/vgpt/llama_small.json",
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=1000000,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="constant_with_warmup",
                        help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=5000,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default="trm-eval", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default="pretrained_models/cnn_fsq12_frac_res320_500k")
    parser.add_argument('--pretrained_transformer_path', type=str,
                        default="pretrained_models/checkpoint_90000_unwrapped")
    parser.add_argument("--trust_remote_code", type=bool, default=False,
                        help=(
                            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                            "execute code present on the Hub on your local machine."
                        ),
                        )
    parser.add_argument("--checkpointing_steps", type=int, default=10000,
                        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="If the training should continue from a checkpoint folder.")
    parser.add_argument("--with_tracking", type=bool, default=True,
                        help="Whether to enable experiment trackers for logging.")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help=(
                            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
                            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
                            "Only applicable when `--with_tracking` is passed."
                        ),
                        )
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
                        help=(
                            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
                            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
                            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
    ),
    )
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--gradient_checkpointing', default=False, action='store_true')
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument('--start_completed_steps', default=None, type=int)

    # datasets
    parser.add_argument("--segment_length", type=int, default=5,
                        help="The length of the segmented trajectories to use for the training.")
    parser.add_argument('--video_stepsize', default=1, type=int)
    parser.add_argument('--dataset_path', default='/home/NAS/rl_data/frame_action_datasets/',
                        type=str, help='Path to the tensorflow datasets')
    parser.add_argument('--dataset_size', default=None, type=int)
    parser.add_argument('--resolution', default=256, type=int, nargs='+')
    parser.add_argument('--resolution_width', default=320, type=int, nargs='+')
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help=(
                            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
                        ),
                        )
    parser.add_argument('--strong_aug', default=False, action='store_true')
    parser.add_argument('--no_aug', default=True, action='store_true')
    parser.add_argument('--oxe_data_mixes_type', default='frac', type=str)

    parser.add_argument("--log_steps", type=int, default=100, help=("Print logs every X steps."))
    parser.add_argument("--validation_steps", type=int, default=5000)
    parser.add_argument('--skip_first_val', default=False, action='store_true')
    parser.add_argument('--latest_checkpoint_only', default=False, action='store_true')
    parser.add_argument('--action_dim', default=13, type=int, help='action dimension for the task')
    parser.add_argument('--action_bins', default=256, type=int)
    parser.add_argument('--action_ranges_path', default="configs/vgpt/frac_action_ranges.pth")
    parser.add_argument('--embed_no_wd', default=False, action='store_true')

    # evaluation
    parser.add_argument('--max_eval_iters', default=100, type=int)
    parser.add_argument('--use_eval_dataset', default=False, action='store_true')
    parser.add_argument('--i3d_path', default=None,
                        type=str, help='path to the i3d model')
    parser.add_argument('--use_frame_metrics', default=True, action='store_true')
    parser.add_argument('--eval_generate_times', default=1, type=int, help='for eval, fvd')
    parser.add_argument('--max_generate_batchsize', default=None, type=int)
    parser.add_argument('--max_decode_batchsize', default=None, type=int)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--log_gif_interval', default=1, type=int)
    parser.add_argument('--log_gif', default=True, action='store_true')
    parser.add_argument('--output_jsonl', default='eval.jsonl', type=str)
    parser.add_argument('--n_sample', default=1, type=int)

    # processor
    parser.add_argument('--visual_token_num', default=4375, type=int)
    parser.add_argument('--bos_token_id', default=4631, type=int)
    parser.add_argument('--eos_token_id', default=4632, type=int)
    parser.add_argument('--vid_multi', default=1, type=int)
    parser.add_argument("--context_length", type=int, default=4)
    parser.add_argument('--tokens_per_frame', default=320, type=int)

    parser.add_argument('--processor_type', default='ctx_msp')
    parser.add_argument('--tokenizer_micro_batch_size', default=2, type=int)

    # vllm
    parser.add_argument('--gpu_memory_utilization', default=0.75, type=float)
    parser.add_argument('--topk', default=100, type=int)
    parser.add_argument('--topp', default=1.0, type=float)
    parser.add_argument('--repetition_penalty', default=1.0, type=float)

    parser.add_argument('--reject_repeating', default=False, action='store_true')
    
    # policy evaluation
    parser.add_argument('--task_instruction', default='pick coke can', type=str)
    parser.add_argument('--policy_model_path', default='pretrained_models/rt_1_tf_trained_for_000400120', type=str)
    parser.add_argument('--max_round', default=16, type=int)

    args = parser.parse_args()
    
    assert args.processor_type == 'ctx_msp'
    
    if 'ctx' in args.processor_type:
        args.bos_token_id = 9006
        args.eos_token_id = 9007
        args.tokens_per_frame = 80

    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.per_device_train_batch_size

    return args



def batch_forward(batch_size, input, input2, forward):
    return torch.cat([forward(input[i: i + batch_size], input2[i: i + batch_size]) for i in range(0, input.shape[0], batch_size)], dim=0)


@torch.no_grad()
def evaluate(args, accelerator, processor, tokenizer, model, policy, eval_dataloader, evaluator, completed_steps=0):
    mse_values, psnr_values, ssim_values, lpips_values, = [], [], [], []
    eval_iters = min(len(eval_dataloader), args.max_eval_iters)
    bar = tqdm(range(eval_iters), desc="validation", disable=not accelerator.is_local_main_process)

    for i, batch in enumerate(eval_dataloader):
        if i == args.max_eval_iters:
            break

        pixel_values, actions = batch
        actions = actions.to(accelerator.device, non_blocking=True)
        pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
        batch_size = pixel_values.shape[0]
        with torch.no_grad():
            if 'ctx' in args.processor_type:
                model_input, pixel_values, ctx_tokens = processor(pixel_values, actions, return_interpolated=True, return_ctx_tokens=True)
            else:
                model_input, pixel_values = processor(pixel_values, actions, return_interpolated=True)
                
        def get_action(obs, t):
            # return actions[:, min(t, actions.shape[1])], False  # !for debug
            obs_numpy = (obs[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            # superresolution by 2x
            # obs_numpy = cv2.resize(obs_numpy, (obs_numpy.shape[1] * 2, obs_numpy.shape[0] * 2))
            cv2.imwrite('tmp.png', obs_numpy[:, :, ::-1])
            action, _ = policy.step(obs_numpy)
            keys = ['base_displacement_vector', 'base_displacement_vertical_rotation', 'gripper_closedness_action', 'rotation_delta', 'terminate_episode', 'world_vector']
            terminal = (action['terminate_episode'] == np.array([1, 0, 0], dtype=np.int32)).all()
            if 'base_displacement_vector' not in action:
                action['base_displacement_vector'] = np.zeros(2)
            if 'base_displacement_vertical_rotation' not in action:
                action['base_displacement_vertical_rotation'] = np.zeros(1)
            print("Raw action: ", {k: action[k] for k in keys})
            action = np.concatenate([action[k] for k in keys])
            # print("Action: ", action)
            action = torch.from_numpy(action)[None].float().to(accelerator.device)
            return action, terminal
        
        policy.reset(args.task_instruction)
        
        max_round = args.max_round
        sampling_params = SamplingParams(
            temperature=1.0, top_k=args.topk, top_p=args.topp,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.tokens_per_frame,
            seed=args.seed,
            ignore_eos=True,
        )
        all_recons, all_actions = [], []
        
        input_tokens = model_input["input_ids"][:, :args.tokens_per_frame * 16 + args.tokens_per_frame]
        input_tokens = input_tokens.detach().cpu().numpy().tolist()
        ctx_input_tokens = input_tokens[0][:args.tokens_per_frame * 16]
        for j in range(max_round):
            terminals = False
            for t in range(args.segment_length - 1):
                # get action
                frame_tokens = torch.tensor([[input_tokens[0][-args.tokens_per_frame:]]]).to(accelerator.device)  # [1, 1, 80]  
                recon = accelerator.unwrap_model(tokenizer).detokenize(ctx_tokens, frame_tokens)
                recon = recon[:, -1].clamp(0.0, 1.0)
                action, terminal = get_action(recon, t+1)
                action_tokens = processor._discretize_actions(action) + args.visual_token_num * 2
                input_tokens[0] += action_tokens[0].detach().cpu().numpy().tolist()                
                
                # generate frame
                outputs = model.generate(prompt_token_ids=input_tokens,
                                        sampling_params=sampling_params,
                                        use_tqdm=False)
                input_tokens[0] += outputs[0].outputs[0].token_ids
                
                all_recons.append(recon)
                all_actions.append(action)
                if terminal:
                    terminals = True
                    break
            
            if terminals:
                break
            last_frame_tokens = input_tokens[0][-args.tokens_per_frame:]
            input_tokens = [ctx_input_tokens + last_frame_tokens]
        
        all_recons = torch.stack(all_recons, dim=1)
        all_actions = torch.stack(all_actions, dim=1)

        # save predicted video
        recon_output = all_recons
        save_path = os.path.join(args.output_dir, "images", f"interact-samples-{completed_steps}")
        os.makedirs(save_path, exist_ok=True)
        recon_frames = [(recon_output[0, i].permute(1, 2, 0).detach().cpu().numpy() *
                            255).astype(np.uint8) for i in range(recon_output.shape[1])]
        imageio.mimsave(f"{save_path}/interact-samples-{completed_steps}-{i}.gif", recon_frames, fps=4, loop=0)
        
        # save gt video for reference
        recon_output = pixel_values
        recon_frames = [(recon_output[0, i].permute(1, 2, 0).detach().cpu().numpy() *
                            255).astype(np.uint8) for i in range(recon_output.shape[1])]
        imageio.mimsave(f"{save_path}/refernce-samples-{completed_steps}-{i}.gif", recon_frames, fps=4, loop=0)
        
        bar.update(1)


def plot_gif(x, postfix=''):
    # [B, T, C, H, W]
    frames = [(x[0, i].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8) for i in range(x.shape[1])]
    imageio.mimsave(f"tmp{postfix}.gif", frames, fps=4, loop=0)


def start_train():
    args = parse_args()
    args.output_dir = os.path.join(args.output_dir, time.strftime("%Y-%m-%d-%X", time.localtime()) + (
        "" if args.exp_name is None else f"-{args.exp_name}"))
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    logging_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logging_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

            with open(os.path.join(args.output_dir, "cmd.sh"), "w") as f:
                f.write("python " + " ".join(sys.argv))

            src_path = os.path.join(args.output_dir, 'src')
            os.makedirs(src_path, exist_ok=True)
            os.system(f"rsync -rv --exclude-from=.gitignore . {src_path}")

    accelerator.wait_for_everyone()

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    train_dataloader, eval_dataloader = get_dataloaders(args)

    if 'ctx' in args.processor_type:
        tokenizer = CompressiveVQModelFSQ.from_pretrained(args.pretrained_model_name_or_path).to(accelerator.device).eval()
    else:
        raise NotImplementedError

    evaluator = Evaluator(args.i3d_path, max_batchsize=args.max_decode_batchsize)

    # Prepare everything with our `accelerator`.
    # we do not need to prepare train dataloader
    evaluator, train_dataloader, eval_dataloader = accelerator.prepare(evaluator, train_dataloader, eval_dataloader)

    if args.processor_type == 'ctx_msp':
        processor = ContextMultiStepPredictionProcessor(args, tokenizer)
    else:
        raise NotImplementedError
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["num_processes"] = accelerator.num_processes
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    model_path = args.pretrained_transformer_path
    if os.path.exists(model_path) and not os.path.exists(os.path.join(model_path, 'config.json')):
        unwrapped_model_path = os.path.join(model_path, 'unwrapped_model')
        os.makedirs(unwrapped_model_path, exist_ok=True)
        cmd = f"python transform_vgpt_checkpoint.py --resume_from_checkpoint {model_path} --save_dir {unwrapped_model_path} --dataset_path {args.dataset_path} --pretrained_model_name_or_path {args.pretrained_model_name_or_path} --processor_type {args.processor_type} --config_name {args.config_name}"
        print(cmd)
        os.system(cmd)
    else:
        unwrapped_model_path = model_path
    
    model = LLM(model=unwrapped_model_path, gpu_memory_utilization=0.75)

    if args.start_completed_steps is not None:
        completed_steps = args.start_completed_steps
    else:
        completed_steps = os.path.basename(model_path).split('_')[1] if os.path.exists(model_path) else 0

    policy = RT1Inference(saved_model_path=args.policy_model_path)
    policy.reset(args.task_instruction)
    # policy = None
        
    evaluate(args, accelerator, processor, tokenizer, model, policy,
            train_dataloader, 
        #  eval_dataloader, 
            evaluator, completed_steps=completed_steps)
    
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    start_train()
