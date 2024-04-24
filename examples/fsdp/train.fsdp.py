#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -- Basic imports
import os
import yaml
import tqdm
import signal
import argparse
import logging

from functools import partial
from contextlib import nullcontext
from datetime import timedelta

# -- maxie specific imports
from maxie.datasets.ipc_dataset_dist import IPCDistributedSegmentedDatasetConfig, IPCDistributedSegmentedDataset, IPCDatasetConfig, IPCDataset
from maxie.modeling.adapted_mae      import AdaptedViTMAEForPreTrainingConfig, AdaptedViTMAEForPreTraining
from maxie.utils.logger              import init_logger
from maxie.utils.seed                import set_seed
from maxie.utils.misc                import is_action_due
from maxie.lr_scheduler              import CosineLRScheduler
from maxie.perf                      import Timer
from maxie.tensor_transforms         import Pad, DownscaleLocalMean, RandomPatch, RandomRotate, RandomShift, Patchify
from maxie.utils_fsdp                import (
    MemoryMaximizer,
    verify_bfloat_support,
    TrainingStateDictConfig,
    ShardedStateDictCheckpointConfig,
    ShardedStateDictCheckpoint,
    broadcast_dict,
)

# -- Torch specific imports
import torch
import torch.nn as nn
import torch.optim as optim

# -- Fully Sharded Data Parallel (FSDP)
# --- Main
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)

# --- Policy wrapper
from torch.distributed.fsdp.wrap                import size_based_auto_wrap_policy, enable_wrap, wrap

# --- Scaler for float16
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

# --- Activation checkpointing
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
)

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
dataset_config            = config.get("dataset")
path_train_json           = dataset_config.get("path_train")
path_eval_json            = dataset_config.get("path_eval")
batch_size                = dataset_config.get("batch_size")
num_workers               = dataset_config.get("num_workers")
micro_batch_size_per_rank = dataset_config.get("micro_batch_size_per_rank")
server_address            = dataset_config.get("server_address")
transforms_config         = dataset_config.get("transforms")
num_patch                 = transforms_config.get("num_patch")
size_patch                = transforms_config.get("size_patch")
frac_shift_max            = transforms_config.get("frac_shift_max")
angle_max                 = transforms_config.get("angle_max")
var_size_patch            = transforms_config.get("var_size_patch")
downscale_factors         = transforms_config.get("downscale_factors")
H_pad                     = transforms_config.get("H_pad")
W_pad                     = transforms_config.get("W_pad")
patch_size                = transforms_config.get("patch_size")
stride                    = transforms_config.get("stride")

# -- Model
model_params = config.get("model")
model_name   = model_params.get("name")

# -- Loss
loss_config      = config.get("loss")
grad_accum_steps = min(loss_config.get("grad_accum_steps"), 1)

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
uses_mixed_precision = misc_config.get("uses_mixed_precision")
max_epochs           = misc_config.get("max_epochs")
max_eval_iter        = misc_config.get("max_eval_iter")
num_gpus             = misc_config.get("num_gpus")
compiles_model       = misc_config.get("compiles_model")

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
# --- Initialize distributed environment
uses_dist = int(os.environ.get("RANK", -1)) != -1
if uses_dist:
    dist_rank       = int(os.environ["RANK"      ])
    dist_local_rank = int(os.environ["LOCAL_RANK"])
    dist_world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend     = dist_backend,
                            rank        = dist_rank,
                            world_size  = dist_world_size,
                            timeout     = timedelta(seconds=900),
                            init_method = "env://",)
    print(f"RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")
else:
    dist_rank       = 0
    dist_local_rank = 0
    dist_world_size = 1
    print(f"NO FSDP is used.  RANK:{dist_rank},LOCAL_RANK:{dist_local_rank},WORLD_SIZE:{dist_world_size}")

# --- Set up GPU device
device = f'cuda:{dist_local_rank}' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)
seed_offset = dist_rank if uses_unique_world_seed else 0

# --- Set up performance utility
memmax = MemoryMaximizer() if dist_local_rank == 0 else None


# ----------------------------------------------------------------------- #
#  FSDP SETUP
# ----------------------------------------------------------------------- #
# -- FSDP policy
# --- Mixed precision
mixed_precision = None
if verify_bfloat_support:
    dist_dtype = 'bfloat16'
    mixed_precision = MixedPrecision(
        param_dtype  = torch.bfloat16,
        reduce_dtype = torch.bfloat16,
        buffer_dtype = torch.bfloat16,
    )
else:
    dist_dtype = 'float16'
    mixed_precision = MixedPrecision(
        param_dtype  = torch.float16,
        reduce_dtype = torch.float16,
        buffer_dtype = torch.float16,
    )

# --- Sharding strategy
sharding_strategy = ShardingStrategy.FULL_SHARD

# --- Wrapping strategy
min_num_params   = 500_000
auto_wrap_policy = partial(size_based_auto_wrap_policy,
                           min_num_params = min_num_params,)

# --- Activation checkpointing
non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    offload_to_cpu  = False,
    checkpoint_impl = CheckpointImpl.NO_REENTRANT,
)

# --- Backward prefetch policy
backward_prefetch = BackwardPrefetch.BACKWARD_PRE


# ----------------------------------------------------------------------- #
#  LOGGING
# ----------------------------------------------------------------------- #
timestamp = None
if dist_rank == 0:
    # Fetch the current timestamp...
    timestamp = init_logger(fl_prefix = fl_log_prefix, drc_log = drc_log, returns_timestamp = True)

    # Convert dictionary to yaml formatted string...
    config_yaml = yaml.dump(config)

    # Log the config...
    logger.info(config_yaml)
timestamp = broadcast_dict(dict(timestamp=timestamp), src = 0, device = device).get('timestamp')


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
    Pad(H_pad, W_pad),
    ## DownscaleLocalMean(factors = downscale_factors),
    ## RandomPatch(num_patch = num_patch, H_patch = size_patch, W_patch = size_patch, var_H_patch = var_size_patch, var_W_patch = var_size_patch, returns_mask = False),
    ## RandomRotate(angle_max),
    RandomShift(frac_y_shift_max=frac_shift_max, frac_x_shift_max=frac_shift_max),
    Patchify(patch_size, stride),
)

# -- Set up training set
ipc_dataset_train_config = IPCDistributedSegmentedDatasetConfig(
    path_json                 = path_train_json,
    micro_batch_size_per_rank = micro_batch_size_per_rank,
    world_size                = dist_world_size,
    transforms                = transforms,
    is_perf                   = True,
    server_address            = server_address,
    loads_segment_in_init     = False,
)
dataset_train = IPCDistributedSegmentedDataset(ipc_dataset_train_config)

# -- Set up eval set
# --- For training loss
dataset_eval_train = IPCDistributedSegmentedDataset(ipc_dataset_train_config)

# --- For val loss
ipc_dataset_eval_config = IPCDistributedSegmentedDatasetConfig(
    path_json                 = path_eval_json,
    micro_batch_size_per_rank = micro_batch_size_per_rank,
    world_size                = dist_world_size,
    transforms                = transforms,
    is_perf                   = True,
    server_address            = server_address,
    loads_segment_in_init     = False,
)
dataset_eval_val = IPCDistributedSegmentedDataset(ipc_dataset_eval_config)

# ----------------------------------------------------------------------- #
#  MODEL
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring model...')
# -- Config the model
model_config = AdaptedViTMAEForPreTrainingConfig(model_name = model_name)
model = AdaptedViTMAEForPreTraining(model_config)

if dist_rank == 0:
    print(f"{sum(p.numel() for p in model.parameters())/1e6} M pamameters.")

# -- Mixed precision
mixed_precision_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dist_dtype]

# --- Autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_context = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=mixed_precision_dtype)

# --- GradScaler
# If enabled=False scaler is a no-op
scaler = ShardedGradScaler(enabled=(dist_dtype == 'float16'))

# -- Compile the model
if compiles_model:
    print("Compiling the model...")
    model = torch.compile(model) # requires PyTorch 2.0

# -- Wrapping the model in FSDP...
if uses_dist:
    # Convert BatchNorm to SyncBatchNorm...
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Wrap it up using FSDP...
    model = FSDP(
        model,
        auto_wrap_policy  = auto_wrap_policy,
        mixed_precision   = mixed_precision,
        backward_prefetch = backward_prefetch,
        forward_prefetch  = True,
        sharding_strategy = sharding_strategy,
        limit_all_gathers = True,
        use_orig_params   = True,
        device_id         = device,
    )

    sharded_param_count = sum(p.numel() for p in model.module.parameters())
    print(f"RANK {dist_rank} - sharded parameter count: {sharded_param_count*1e-6} M.")

    dist.barrier()

# -- Optional grad sync off
no_grad_sync_context = model.no_sync() if grad_accum_steps > 1 else nullcontext()

# -- [TODO] Apply activation checkpointing
ac_layer = None
if ac_layer is not None:
    check_fn = lambda submodule: isinstance(submodule, ac_layer)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn = non_reentrant_wrapper,
        check_fn              = check_fn
    )

if dist_rank == 0:
    print(f"Current timestamp: {timestamp}")


# ----------------------------------------------------------------------- #
#  CRITERION (LOSS)
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring criterion (Skip, it's configured in the model)...')


# ----------------------------------------------------------------------- #
#  Optimizer
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring optimizer...')
param_iter = model.parameters()
optimizer = optim.AdamW(param_iter,
                        lr = lr,
                        weight_decay = weight_decay)
scheduler = CosineLRScheduler(optimizer         = optimizer,
                              warmup_iterations = warmup_iterations,
                              total_iterations  = total_iterations,
                              min_lr            = min_lr)


# ----------------------------------------------------------------------- #
#  CHECKPOINT (SHARDED STATE DICT)
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Confguring optimizer...')
# -- Set init training state dict
loss_min = float('inf')
training_state_dict_config = TrainingStateDictConfig(
    epoch     = 0,
    start_idx = dataset_train.start_idx,
    end_idx   = dataset_train.end_idx,
    loss_min  = loss_min,
)

# -- Sharded state dict
chkpt_config = ShardedStateDictCheckpointConfig(
    model           = model,
    optimizer       = optimizer,
    lr_scheduler    = scheduler,
    training_state  = training_state_dict_config,
    rank            = dist_rank,
    device          = device,
    path_checkpoint = path_chkpt_prev,
)
checkpointer = ShardedStateDictCheckpoint(config = chkpt_config)

# -- Optional resumption
epoch_min = 0
from_resume = path_chkpt_prev is not None
if from_resume:
    if isinstance(checkpointer, ShardedStateDictCheckpoint):
        checkpointer.load()

        training_state = checkpointer.config.training_state
        epoch_min      = training_state.epoch
        start_idx_prev = training_state.start_idx
        end_idx_prev   = training_state.end_idx
        loss_min       = training_state.loss_min

        scheduler.step()    # scheduler state loading is handled inside checkpoint.load()
        logger.info(f"PREV - epoch_min = {epoch_min}, loss_min = {loss_min}")


# ----------------------------------------------------------------------- #
#  HELPER
# ----------------------------------------------------------------------- #
@torch.no_grad()
def estimate_loss(dataloader, model, autocast_context, max_iter = None, desc = '', device = 'cpu'):
    ''' Estimate loss.
        The dataloader should be wrapped with Dataloader class or
        DistributedSampler class, best with shuffle being true.  The shuffle
        takes place before batching.
    '''
    model.eval()

    if max_iter is None:
        max_iter = len(dataloader)

    losses      = torch.zeros(len(dataloader), device = device)
    num_samples = torch.zeros(len(dataloader), device = device)
    for enum_idx, batch_data in tqdm.tqdm(enumerate(dataloader), total = max_iter, desc = f'[RANK {dist_rank}] Eval{desc}'):    # (B, C, H, W)
        if enum_idx + 1 > max_iter: break

        ## print("Pre fetching")

        batch_input = batch_data
        batch_input = batch_input.to(device, non_blocking = True)

        ## print("Post fetching")

        with autocast_context:
            ## print("Forwarding")
            batch_output = model(batch_input)

            ## print("Loss")
            loss = batch_output.loss

        losses[enum_idx]      = loss
        num_samples[enum_idx] = len(batch_input)

    losses_sum      = torch.dot(losses, num_samples)
    num_samples_sum = losses_sum.sum()

    world_losses_sum      = [ torch.tensor(0.0).to(device) for _ in range(dist_world_size) ]
    world_num_samples_sum = [ torch.tensor(0.0).to(device) for _ in range(dist_world_size) ]
    dist.all_gather(world_losses_sum, losses_sum)
    dist.all_gather(world_num_samples_sum, num_samples_sum)

    world_losses_mean = torch.tensor(world_losses_sum).sum() / torch.tensor(world_num_samples_sum).sum()

    model.train()

    return world_losses_mean

def is_last_batch(batch_idx, num_batches):
    return batch_idx + 1 == num_batches

# ----------------------------------------------------------------------- #
#  TRAINING LOOP
# ----------------------------------------------------------------------- #
print(f'[RANK {dist_rank}] Ready for training loop...')
try:
    # -- Train one epoch
    for epoch in tqdm.tqdm(range(max_epochs), desc = f'[RANK {dist_rank}] Epoch'):
        # -- Restore epoch and starting micro batch index
        epoch += epoch_min

        # -- Train on one segment
        seg_pbar = tqdm.tqdm(total = dataset_train.num_seg, initial = 0, desc = f'[RANK {dist_rank}] Segment')
        micro_batch = 0
        batch_idx = 0

        # Reset everything for a new epoch if not from a resume
        if not from_resume:
            dataset_train.reset()
        while dataset_train.end_idx < dataset_train.total_size:
            # [PERFORMANCE]
            if dist_local_rank == 0:
                memmax.start()

            # -- Switch to training state
            model.train()

            # -- Prepare training on one micro batch (iteration)
            # Set next micro batch
            dataset_train.set_start_idx(dataset_train.end_idx)

            if dist_rank == 0:
                print(f"Working on segment: {dataset_train.start_idx}:{dataset_train.end_idx}; Total size: {dataset_train.total_size}")

            # Split sampler across ranks
            sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
            dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers = num_workers)

            # Shuffle the training example
            sampler.set_epoch(epoch)

            # -- Train one mini batch
            grad_accum_counter = 0
            for batch_idx, batch_data in tqdm.tqdm(enumerate(dataloader), total = len(dataloader), desc = f'[RANK {dist_rank}] Mini batch'):    # (B, C, H, W)
                batch_input = batch_data
                batch_input = batch_input.to(device, non_blocking = True)

                # Forward
                with autocast_context:
                    batch_output = model(batch_input)
                    loss = batch_output.loss  # Refer to https://github.com/huggingface/transformers/blob/e34da3ee3c9d2d628fdbeb60cee45c4f8f32945a/src/transformers/models/vit_mae/modeling_vit_mae.py#L1001
                    loss = loss / grad_accum_steps  # scale the loss to account for gradient accumulation

                # Backward
                # Turn off grad sync for every batch to simulate a larger batch size
                with no_grad_sync_context:
                    scaler.scale(loss).backward()

                # Increment the grad accum counter
                grad_accum_counter += 1

                # Conditional parameter updates either when it's due or the last batch
                if is_action_due(grad_accum_counter, grad_accum_steps) or is_last_batch(batch_idx, len(dataloader)):
                    # Grad clipping
                    if grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()

                    # Flush the gradients
                    optimizer.zero_grad(set_to_none=True)

                    # Reset grad accum counter
                    grad_accum_counter = 0

            # Track and update micro batch for conditional logging and eval
            micro_batch += 1
            seg_pbar.update(1)

            # -- Eval and checkpointing
            # Rank0 performs evaluation and decide if a sharded state dict should be saved
            if is_action_due(micro_batch, chkpt_saving_period):
                print(f'[RANK {dist_rank}] Start evaluation...')

                # -- Eval
                # --- Train
                # Get a random subset of the training set
                dataset_eval_train.reset()
                high_seg_idx = dataset_eval_train.total_size - micro_batch_size_per_rank * world_size
                rand_start_idx = torch.randint(low = 0, high = high_seg_idx, size = (1,))
                dataset_eval_train.set_start_idx(rand_start_idx)

                sampler_eval = torch.utils.data.DistributedSampler(dataset_eval_train, shuffle=True)
                dataloader_eval = torch.utils.data.DataLoader(dataset_eval_train, batch_size=batch_size, sampler = sampler_eval, num_workers = num_workers, shuffle = False)

                # Shuffle the training example
                sampler_eval.set_epoch(0)

                # Get loss
                train_loss = estimate_loss(dataloader_eval, model, autocast_context, max_iter = max_eval_iter, desc = '(training set)', device = device)


                # --- Validation
                # Get a random subset of the validation set
                dataset_eval_val.reset()
                high_seg_idx = dataset_eval_val.total_size - micro_batch_size_per_rank * world_size
                rand_start_idx = torch.randint(low = 0, high = high_seg_idx, size = (1,))
                dataset_eval_val.set_start_idx(rand_start_idx)

                sampler_eval = torch.utils.data.DistributedSampler(dataset_eval_val, shuffle=True)
                dataloader_eval = torch.utils.data.DataLoader(dataset_eval_val, batch_size=batch_size, sampler = sampler_eval, num_workers = num_workers, shuffle = False)
                validate_loss = estimate_loss(dataloader_eval, model, autocast_context, max_iter = max_eval_iter, desc = '(validation set)', device = device)

                # Shuffle the training example
                sampler_eval.set_epoch(0)

                # -- Save checkpoint
                if validate_loss < loss_min:
                    loss_min = validate_loss.item()

                    dir_chkpt = f"{timestamp}.epoch_{epoch}.end_idx_{end_idx}"
                    if dir_chkpt_prefix is not None: dir_chkpt = f"{dir_chkpt_prefix}.{dir_chkpt}"
                    path_chkpt = os.path.join(dir_root_chkpt, dir_chkpt)
                    checkpointer.save(model, optimizer, scheduler, training_state, path_chkpt)

                # All ranks wait until the end of evaluation by rank 0
                # [WARNING] Expecting NCCL TIMEOUT ERROR if the evaluation takes too long
                dist.barrier()
                print(f'[RANK {dist_rank}] Done evaluation...')

            # -- Update lr after one iteration
            scheduler.step()

            # [PERFORMANCE]
            if dist_local_rank == 0:
                memmax.update()

            # [PERFORMANCE]
            if dist_local_rank == 0:
                memmax.stop()

        # Reset the from_resume flag
        from_resume = False

        # Close seg pbar
        seg_pbar.close()

except KeyboardInterrupt:
    print(f"FSDP RANK {dist_rank}: Training was interrupted!")
except Exception as e:
    print(f"FSDP RANK {dist_rank}: Error occurred: {e}")
finally:
    # Ensure that the process group is always destroyed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
