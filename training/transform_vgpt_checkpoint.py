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
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None,
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
    parser.add_argument("--output_dir", type=str, default="trm-debug", help="Where to store the final model.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default="pretrained_models/checkpoint-tokenizer100000")
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
    parser.add_argument('--no_aug', default=False, action='store_true')
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
    parser.add_argument('--use_frame_metrics', default=False, action='store_true')
    parser.add_argument('--eval_generate_times', default=1, type=int, help='for eval, fvd')
    parser.add_argument('--max_generate_batchsize', default=None, type=int)
    parser.add_argument('--max_decode_batchsize', default=None, type=int)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument('--log_gif_interval', default=10, type=int)
    parser.add_argument('--log_gif', default=False, action='store_true')

    # processor
    parser.add_argument('--visual_token_num', default=4375, type=int)
    parser.add_argument('--bos_token_id', default=4631, type=int)
    parser.add_argument('--eos_token_id', default=4632, type=int)
    parser.add_argument('--vid_multi', default=1, type=int)
    parser.add_argument("--context_length", type=int, default=4)
    parser.add_argument('--tokens_per_frame', default=320, type=int)

    parser.add_argument('--processor_type', default='simple')

    args = parser.parse_args()

    if args.per_device_eval_batch_size is None:
        args.per_device_eval_batch_size = args.per_device_train_batch_size

    return args



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
    # if accelerator.is_main_process:
    #     if args.output_dir is not None:
    #         os.makedirs(args.output_dir, exist_ok=True)

    #         with open(os.path.join(args.output_dir, "cmd.sh"), "w") as f:
    #             f.write("python " + " ".join(sys.argv))

    #         src_path = os.path.join(args.output_dir, 'src')
    #         os.makedirs(src_path, exist_ok=True)
    #         os.system(f"rsync -rv --exclude-from=.gitignore . {src_path}")

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
        tokenizer = CNNFSQModel256.from_pretrained(args.pretrained_model_name_or_path).to(accelerator.device)

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        assert False

    model = AutoModelForCausalLM.from_config(config,
                                             torch_dtype=torch.float32,
                                             attn_implementation='flash_attention_2',
                                             trust_remote_code=args.trust_remote_code)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("gradient checkpointing enabled")

    # if args.pretrained_transformer_path is not None:
    #     state_dict = load_file(os.path.join(args.pretrained_transformer_path, 'model.safetensors'))
    #     if args.load_internal_llm:
    #         model.llm.load_state_dict(state_dict, strict=True)
    #     else:
    #         model.load_state_dict(state_dict, strict=True)
    #     logger.info("Finetuning the model from " + args.pretrained_transformer_path)
    # else:
    logger.info("Training new model from scratch")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    # no_decay = ["bias", "layer_norm.weight"]
    no_decay = []
    if args.embed_no_wd:
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                if pn.endswith('bias') or \
                    (pn.endswith('weight') and isinstance(m, torch.nn.Embedding)) or \
                    (pn.endswith('weight') and isinstance(m, torch.nn.LayerNorm)) or \
                        (pn.endswith('weight') and isinstance(m, LlamaRMSNorm)):
                    fpn = '%s.%s' % (mn, pn) if mn else pn
                    no_decay.append(fpn)
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    evaluator = Evaluator(args.i3d_path, max_batchsize=args.max_decode_batchsize)

    # Prepare everything with our `accelerator`.
    # we do not need to prepare train dataloader
    model, evaluator, optimizer, lr_scheduler, eval_dataloader = accelerator.prepare(
        model, evaluator, optimizer, lr_scheduler, eval_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    # if accelerator.distributed_type == DistributedType.TPU:
    #     model.tie_weights()

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    if args.start_completed_steps is not None:
        completed_steps = args.start_completed_steps
        progress_bar.update(completed_steps)
    starting_epoch = 0
    end = time.time()

    lastest_output_dir, lastest_completed_steps = None, None
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            raise NotImplementedError
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            # resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            resume_step = int(training_difference.replace("checkpoint_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

        lastest_output_dir, lastest_completed_steps = args.resume_from_checkpoint, completed_steps

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        args.save_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )
    
    cmd = f"ln -s pretrained_models/dummy_tokenizer/tokenizer_config.json {args.save_dir}"
    os.system(cmd)
    cmd = f"ln -s pretrained_models/dummy_tokenizer/tokenizer.json {args.save_dir}"
    os.system(cmd)

if __name__ == "__main__":
    start_train()
