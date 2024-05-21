#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -- OLCF specific imports
from maxie.plugins.olcf import init_dist_env_on_summit

# -- Basic imports
import os
import yaml
import tqdm
import signal
import argparse
import logging
import traceback

from functools  import partial
from contextlib import nullcontext
from datetime   import timedelta

# -- maxie specific imports
from maxie.datasets.ipc_segmented_dataset_dist import IPCDistributedSegmentedDatasetConfig, IPCDistributedSegmentedDataset, IPCDatasetConfig, IPCDataset
from maxie.utils.seed           import set_seed
from maxie.utils.misc           import is_action_due
from maxie.perf                 import Timer
from maxie.tensor_transforms    import Pad, DownscaleLocalMean, RandomPatch, RandomRotate, RandomShift, Patchify, Norm

# -- Torch specific imports
import torch
import torch.nn as nn
import torch.optim as optim

from packaging import version

# --- Distributed library
import torch.distributed as dist

# -- Debug
torch.autograd.set_detect_anomaly(False)    # [WARNING] Making it True may throw errors when using bfloat16

# -- Reporting specific imports
import colorama
colorama.init(autoreset=True)

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- #
#  COMMAND LINE INTERFACE
# ----------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description="Load training configuration from a YAML file to a dictionary.")
parser.add_argument("yaml_file", help="Path to the YAML file")
args = parser.parse_args()

# ----------------------------------------------------------------------- #
#  CONFIGURATION
# ----------------------------------------------------------------------- #
# Load CONFIG from YAML
fl_yaml = args.yaml_file
with open(fl_yaml, 'r') as fh:
    config = yaml.safe_load(fh)

# -- Checkpoint
chkpt_config        = config.get("checkpoint")
dir_root_chkpt      = chkpt_config.get("directory")
fl_chkpt_prefix     = chkpt_config.get("filename_prefix")
dir_chkpt_prefix    = chkpt_config.get("dir_chkpt_prefix")
path_chkpt_prev     = chkpt_config.get("path_chkpt_prev")
chkpt_saving_period = chkpt_config.get("chkpt_saving_period")

# -- Dataset
dataset_config       = config.get("dataset")
path_train_json      = dataset_config.get("path_train")
path_eval_json       = dataset_config.get("path_eval")
batch_size           = dataset_config.get("batch_size")
num_workers          = dataset_config.get("num_workers")
seg_size             = dataset_config.get("seg_size")
server_address       = dataset_config.get("server_address")
transforms_config    = dataset_config.get("transforms")
num_patch            = transforms_config.get("num_patch")
size_patch           = transforms_config.get("size_patch")
frac_shift_max       = transforms_config.get("frac_shift_max")
angle_max            = transforms_config.get("angle_max")
var_size_patch       = transforms_config.get("var_size_patch")
downscale_factors    = transforms_config.get("downscale_factors")
H_pad                = transforms_config.get("H_pad")
W_pad                = transforms_config.get("W_pad")
patch_size           = transforms_config.get("patch_size")
stride               = transforms_config.get("stride")
detector_norm_params = transforms_config.get("norm")

# -- Model
model_params = config.get("model")
model_name   = model_params.get("name")

# -- Loss
loss_config      = config.get("loss")
grad_accum_steps = max(int(loss_config.get("grad_accum_steps")), 1)

# -- Optimizer
optim_config = config.get("optim")
lr           = float(optim_config.get("lr"))
weight_decay = float(optim_config.get("weight_decay"))
grad_clip    = float(optim_config.get("grad_clip"))

# -- Scheduler
lr_scheduler_config = config.get("lr_scheduler")
patience            = lr_scheduler_config.get("patience")
warmup_iterations   = lr_scheduler_config.get("warmup_iterations")
total_iterations    = lr_scheduler_config.get("total_iterations")
uses_prev_scheduler = lr_scheduler_config.get("uses_prev")
min_lr              = float(lr_scheduler_config.get("min_lr"))

# -- Distributed envs
dist_config            = config.get("dist")
dist_backend           = dist_config.get("backend")
uses_unique_world_seed = dist_config.get("uses_unique_world_seed")
dist_dtype             = dist_config.get("dtype")

# -- Logging
logging_config = config.get("logging")
drc_log       = logging_config.get("directory")
fl_log_prefix = logging_config.get("filename_prefix")

# -- Misc
misc_config = config.get("misc")
max_epochs           = misc_config.get("max_epochs")
max_eval_iter        = misc_config.get("max_eval_iter")
num_gpus             = misc_config.get("num_gpus")
compiles_model       = misc_config.get("compiles_model")
data_dump_on         = misc_config.get("data_dump_on", False)

# ----------------------------------------------------------------------- #
#  MISC FEATURES
# ----------------------------------------------------------------------- #
def signal_handler(signal, frame):
    # Emit Ctrl+C like signal which is then caught by our try/except block
    raise KeyboardInterrupt

# Register the signal handler
signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ----------------------------------------------------------------------- #
#  DIST SETUP
# ----------------------------------------------------------------------- #
# -- DIST init
# --- OLCF specific env
# torchrun doesn't work well on OLCF.  Refer to https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html#torchrun
# Thanks to the suggestion by @frobnitzem
torchrun_exists = int(os.environ.get("RANK", -1)) != -1
if not torchrun_exists: init_dist_env_on_summit()

# --- Initialize distributed environment
uses_dist = int(os.environ.get("RANK", -1)) != -1
if uses_dist:
    dist_rank       = int(os.environ["RANK"      ])
    dist_local_rank = int(os.environ["LOCAL_RANK"])
    dist_world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend     = dist_backend,
                            rank        = dist_rank,
                            world_size  = dist_world_size,
                            timeout     = timedelta(seconds=1800),
                            init_method = "env://",)
    print(f"RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")
else:
    dist_rank       = 0
    dist_local_rank = 0
    dist_world_size = 1
    print(f"NO FSDP is used.  RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")

# --- Set up GPU device
gpu_idx = dist_local_rank % torch.cuda.device_count()    # dist_local_rank is node-centric, whereas torch.cuda.device_count() is resource-centeric (on LSF)
device = f'cuda:{gpu_idx}' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)
seed_offset = dist_rank if uses_unique_world_seed else 0

# ----------------------------------------------------------------------- #
#  DATASET
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring dataset...')
# -- Seeding
base_seed  = 0
world_seed = base_seed + seed_offset
set_seed(world_seed)

# -- Set up transformation
transforms = (
    Norm(detector_norm_params),
    Pad(H_pad, W_pad),
    ## DownscaleLocalMean(factors = downscale_factors),
    ## RandomPatch(num_patch = num_patch, H_patch = size_patch, W_patch = size_patch, var_H_patch = var_size_patch, var_W_patch = var_size_patch, returns_mask = False),
    ## RandomRotate(angle_max),
    RandomShift(frac_y_shift_max=frac_shift_max, frac_x_shift_max=frac_shift_max),
    Patchify(patch_size, stride),
)

# -- Set up training set
ipc_dataset_train_config = IPCDistributedSegmentedDatasetConfig(
    path_json             = path_train_json,
    seg_size              = seg_size,
    world_size            = dist_world_size,
    transforms            = transforms,
    is_perf               = True,
    server_address        = tuple(server_address),
    loads_segment_in_init = False,
)
dataset_train = IPCDistributedSegmentedDataset(ipc_dataset_train_config)

def custom_collate(batch):
    batch_filtered = [x for x in batch if x is not None]
    return torch.cat(batch_filtered, dim=0) if len(batch_filtered) else None

from_resume = path_chkpt_prev is not None
batch_idx   = 0
try:
    # Reset everything for a new epoch if not from a resume
    if not from_resume:
        dataset_train.reset()
    while dataset_train.end_idx < dataset_train.total_size:
        # -- Prepare training on one micro batch (iteration)
        # Set next micro batch
        dataset_train.set_start_idx(dataset_train.end_idx)

        if dist_rank == 0:
            print(f"Working on segment: {dataset_train.start_idx}-{dataset_train.end_idx}; Total size: {dataset_train.total_size}")

        # Split sampler across ranks
        sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
        dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers = num_workers, collate_fn=custom_collate)

        # Shuffle the training example
        sampler.set_epoch(batch_idx)

        dataloader_iter = iter(dataloader)
        while True:
            try:
                with Timer(tag = None, is_on = True) as t:
                    batch_data = next(dataloader_iter)
                loading_time_in_sec = t.duration

                print(f"RANK {dist_rank} - Batch idx: {batch_idx:d}, Total time: {loading_time_in_sec:.2f} s, Average time: {loading_time_in_sec / len(batch_data) * 1e3:.2f} ms/event, Batch shape: {batch_data.shape}")

                batch_idx += 1
            except StopIteration:
                break
except KeyboardInterrupt:
    print(f"FSDP RANK {dist_rank}: Training was interrupted!")
except Exception as e:
    tb = traceback.format_exc()
    print(f"FSDP RANK {dist_rank}: Error occurred: {e}\nTraceback: {tb}")
finally:
    # Ensure that the process group is always destroyed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
