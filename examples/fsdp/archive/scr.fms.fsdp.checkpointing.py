from packaging import version

import os
import functools

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)

from poorman_transformer.modeling.transformer import Transformer, TransformerBlock

import copy

CURRENT_PYTORCH_VERSION = torch.__version__
print(f"PyTorch version: {CURRENT_PYTORCH_VERSION}")

# ___/ DDP INIT \___
## # Set environment variables
## os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
## os.environ['MASTER_PORT'] = '12355'      # an unused port on the machine
## os.environ['RANK'] = '0'                 # the rank of the current process
## os.environ['WORLD_SIZE'] = '1'           # the total number of processes
## os.environ['LOCAL_RANK'] = '0'           # the rank of the current process

fsdp_rank       = int(os.environ["RANK"      ])
fsdp_local_rank = int(os.environ["LOCAL_RANK"])
fsdp_world_size = int(os.environ["WORLD_SIZE"])
fsdp_backend    = 'nccl'
dist.init_process_group(backend=fsdp_backend,
                        rank = fsdp_rank,
                        world_size = fsdp_world_size,
                        init_method = "env://",)
print(f"RANK:{fsdp_rank},LOCAL_RANK:{fsdp_local_rank},WORLD_SIZE:{fsdp_world_size}")

# ___/ DEVICE \___
device = f'cuda:{fsdp_local_rank}' if torch.cuda.is_available() else 'cpu'
if device != 'cpu': torch.cuda.set_device(device)

# ___/ MODEL \___
# Define model
token_lib_size = 33_000
embd_size      = 128
num_blocks     = 28
head_size      = 4
context_length = 2000
num_heads      = embd_size // head_size
model = Transformer(token_lib_size, embd_size, context_length, num_blocks, num_heads).to(device)
if fsdp_rank == 0: print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# ___/ POLICY \___
# == Mixed precision ==
# Create a policy - one for Bfloat16 is shown
bfloatPolicy = MixedPrecision(
    param_dtype  = torch.bfloat16, # Param precision
    reduce_dtype = torch.bfloat16, # Gradient communication precision.
)
## comboPolicy = MixedPrecision(
##     param_dtype  = torch.bfloat16, # Param precision
##     reduce_dtype = torch.float32,  # Gradient communication precision.
##     buffer_dtype = torch.float32,  # Buffer precision.
## )

# == Sharding strategy ==
# # Three available sharding strategies - tradeoff memory size vs communication overhead:
# ShardingStrategy.FULL_SHARD # default!  Model, optimizer and gradient are all sharded (communicated) ... max model size support
# ShardingStrategy.SHARD_GRAD_OP # Zero2 mode - model parameters are not freed after forward pass, reducing communication needs
# ShardingStrategy.NO_SHARD  # DDP mode - each GPU keeps a full copy of the model, optimizer and gradients
#                            # only grad synch needed
# ShardingStrategy.HYBRID_SHARD   #FSDP Full shard within each node, but No Shard (DDP) between each nodes. 
model_sharding_strategy = ShardingStrategy.FULL_SHARD

# == Auto wrapper ==
transformer_auto_wrapper_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        TransformerBlock,
    },
)

# ___/ FSDP WRAP \___
model_orig = copy.deepcopy(model)
sharded_model = FSDP(
    model,
    auto_wrap_policy  = transformer_auto_wrapper_policy,
    mixed_precision   = bfloatPolicy,
    sharding_strategy = model_sharding_strategy,
    device_id         = torch.cuda.current_device(),
)

# Count parameters in sharded and original model
unsharded_param_count = sum(p.numel() for p in model_orig.parameters())
sharded_param_count = sum(p.numel() for p in sharded_model.module.parameters())    # ...`.module` accesses the original unwrapped model
print(f"Unsharded parameter count: {unsharded_param_count}")
print(f"Sharded parameter count: {sharded_param_count}")

# ___/ OPTIMIZER \___
optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=8e-4, weight_decay=0.005)

# ___/ CHECKPOINT \___
# == Full state dict ==
# [SCENARIO] We simulate a model saving and loading.
uses_full_state_dict = False
if uses_full_state_dict and version.parse(CURRENT_PYTORCH_VERSION) < version.parse('2.1.0'):
    # == Saving ==
    # === Model ===
    fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(sharded_model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        cpu_state = sharded_model.state_dict()
    if fsdp_rank == 0:
        path_chkpt_model = 'dummy_chkpt.model.pt'
        checkpoint_type = StateDictType.FULL_STATE_DICT
        torch.save(cpu_state, path_chkpt_model)
        print(f"{path_chkpt_model} is saved.")

    # === Optimizer ===
    optim_state = FSDP.full_optim_state_dict(sharded_model, optimizer)
    if fsdp_rank == 0:
        path_chkpt_optim = 'dummy_chkpt.optim.pt'
        torch.save(optim_state, path_chkpt_optim)
        print(f"{path_chkpt_optim} is saved.")

    dist.barrier()

    # == Loading ==
    # === Model ===
    # I believe this is not the right way to load a full state dict, becuase the model is already sharded...
    if fsdp_rank == 0:
        chkpt_model = torch.load(path_chkpt_model, map_location = 'cpu')
        ## sharded_model.module.load_state_dict(chkpt_model)
        model = Transformer(token_lib_size, embd_size, context_length, num_blocks, num_heads)
        model.load_state_dict(chkpt_model)
        sharded_model = FSDP(
            model,
            auto_wrap_policy  = transformer_auto_wrapper_policy,
            mixed_precision   = bfloatPolicy,
            sharding_strategy = model_sharding_strategy,
            device_id         = torch.cuda.current_device(),
        )
        print(f"{path_chkpt_model} is loaded to model.")

    dist.barrier()

    # === Optimizer ===
    chkpt_optim = None
    if fsdp_rank == 0:
        chkpt_optim = torch.load(path_chkpt_optim)
        print(f"[RANK {fsdp_rank}] {path_chkpt_optim} is loaded.")
    sharded_optim = FSDP.scatter_full_optim_state_dict(chkpt_optim, sharded_model)
## else:
##     # == Saving ==
##     # === Model ===
##     if fsdp_rank == 0:
##         path_chkpt_model = 'dummy_chkpt.model.pt'
##         checkpoint_type = StateDictType.FULL_STATE_DICT
##         fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
##         with FSDP.state_dict_type(sharded_model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
##             cpu_state = sharded_model.state_dict()
##         torch.save(cpu_state, path_chkpt_model)
##         print(f"{path_chkpt_model} is saved.")
## 
##     # === Optimizer ===
##     if fsdp_rank == 0:
##         path_chkpt_optim = 'dummy_chkpt.optim.pt'
##         optim_state = FSDP.full_optim_state_dict(sharded_model, optimizer)
##         torch.save(optim_state, path_chkpt_optim)
##         print(f"{path_chkpt_optim} is saved.")
## 
##     # == Loading ==
##     # === Model ===
##     if fsdp_rank == 0:
##         chkpt_model = torch.load(path_chkpt_model)
##         sharded_model.load_state_dict(chkpt_model)
##         print(f"{path_chkpt_model} is loaded.")
## 
##     # === Optimizer ===
##     chkpt_optim = None
##     if fsdp_rank == 0:
##         chkpt_optim = torch.load(path_chkpt_optim)
##     ## sharded_optim = FSDP.scatter_full_optim_state_dict(chkpt_optim, sharded_model)
##     FSDP.optim_state_dict_to_load(chkpt_optim, sharded_model)
##     print(f"[RANK {fsdp_rank}] {path_chkpt_optim} is loaded.")

if not uses_full_state_dict and version.parse(CURRENT_PYTORCH_VERSION) < version.parse('2.1.0'):
    # == Saver ==
    path_root_chkpt_dir = 'dummy_sharded_chkpts'
    os.makedirs(path_root_chkpt_dir, exist_ok = True)
    distributed_writer = FileSystemWriter(
        path_root_chkpt_dir,
        single_file_per_rank=True,
        ## thread_count=cfg.save_using_num_threads,
        sync_files=False,
        # per_thread_copy_ahead=20_000_000,
    )
    state_dict = {}
    with FSDP.state_dict_type(sharded_model, StateDictType.SHARDED_STATE_DICT):
        checkpoint  = sharded_model.state_dict()
        optim_state = FSDP.sharded_optim_state_dict(sharded_model, optimizer)
        state_dict = {
            'model_state_dict' : checkpoint,
            'optim_state_dict' : optim_state,
        }

        save_state_dict(
            state_dict     = state_dict,
            storage_writer = distributed_writer,
            ## planner        = DefaultSavePlanner(),
        )
    dist.barrier()

    if fsdp_rank == 0:
        print(f"Sharded state checkpoint saved to {path_root_chkpt_dir}.")

    # == Loader ==
    model = Transformer(token_lib_size, embd_size, context_length, num_blocks, num_heads)
    sharded_model = FSDP(
        model,
        auto_wrap_policy  = transformer_auto_wrapper_policy,
        mixed_precision   = bfloatPolicy,
        sharding_strategy = model_sharding_strategy,
        device_id         = torch.cuda.current_device(),
    )
    reader = FileSystemReader(path_root_chkpt_dir)
    with FSDP.state_dict_type(sharded_model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model_state_dict": sharded_model.state_dict()}
        load_state_dict(
            state_dict     = checkpoint,
            storage_reader = reader,
            ## planner        = DefaultLoadPlanner(),
        )
        sharded_model.load_state_dict(checkpoint["model_state_dict"])
    if fsdp_rank == 0:
        print(f"Sharded state checkpoint loaded from {path_root_chkpt_dir}.")
