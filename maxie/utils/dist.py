# ====================================================================================================
# Distributed Environment Initialization Process:
# - mpi4py Usage:
#   - mpi4py is used to obtain rank and size information for distributed processing.
#   - It provides a standardized way to get this information across different systems.
#
# - Environment Variable Setup:
#   - We set environment variables like WORLD_SIZE, RANK, and LOCAL_RANK.
#   - This adheres to the torchrun convention, which uses these environment variables.
#   - PyTorch's distributed module can then use these variables for its setup.
#
# ====================================================================================================

import os
import socket
import torch
import torch.distributed as dist
from datetime import timedelta
from omegaconf import OmegaConf
import sys

def init_dist_env_with_mpi():
    """Initialize distributed environment using MPI."""
    try:
        from mpi4py import MPI
    except ImportError:
        raise RuntimeError("mpi4py is not found!!!")

    # Use mpi4py to get rank and size information
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # Calculate local rank based on the available GPUs
    mpi_local_rank = mpi_rank % torch.cuda.device_count()

    # Are we using multiple ranks?
    uses_dist = mpi_size > 1

    # Set basic environment variables
    os.environ["WORLD_SIZE"] = str(mpi_size) if uses_dist else "1"
    os.environ["RANK"] = str(mpi_rank) if uses_dist else "0"
    os.environ["LOCAL_RANK"] = str(mpi_local_rank) if uses_dist else "0"
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")

    # Only handle MASTER_ADDR if it's not already set in environment
    if "MASTER_ADDR" not in os.environ:
        if uses_dist:
            MAIN_RANK = 0
            # Try to determine the master address and broadcast it to every rank
            master_addr = socket.gethostbyname(socket.gethostname()) if mpi_rank == MAIN_RANK else None
            master_addr = mpi_comm.bcast(master_addr, root=MAIN_RANK)
            os.environ["MASTER_ADDR"] = master_addr
        else:
            # Single rank case
            os.environ["MASTER_ADDR"] = "127.0.0.1"

    print(f"Environment setup for distributed computation: "
          f"WORLD_SIZE={os.environ['WORLD_SIZE']}, "
          f"RANK={os.environ['RANK']}, "
          f"LOCAL_RANK={os.environ['LOCAL_RANK']}, "
          f"MASTER_ADDR={os.environ['MASTER_ADDR']}, "
          f"MASTER_PORT={os.environ['MASTER_PORT']}")

def init_dist_env_with_srun():
    uses_dist = int(os.environ.get('SLURM_NTASKS',1)) > 1
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS'] if uses_dist else "1"
    os.environ['RANK'] = os.environ['SLURM_PROCID'] if uses_dist else "0"
    os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID'] if uses_dist else "0"
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")
    if "MASTER_ADDR" not in os.environ:
        if uses_dist:
            raise RuntimeError(
                "Error: MASTER_ADDR environment variable is not set.\n"
                "Please set it before launching with srun using:\n"
                "    export MASTER_ADDR=$(hostname)"
            )
        else:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
    print(f"Environment setup for distributed computation: "
          f"WORLD_SIZE={os.environ['WORLD_SIZE']}, "
          f"RANK={os.environ['RANK']}, "
          f"LOCAL_RANK={os.environ['LOCAL_RANK']}, "
          f"MASTER_ADDR={os.environ['MASTER_ADDR']}, "
          f"MASTER_PORT={os.environ['MASTER_PORT']}")

def dist_setup(cpu_only, dist_backend='nccl'):
    # -- DIST init
    # --- OLCF specific env
    # torchrun doesn't work well on OLCF.  Refer to https://docs.olcf.ornl.gov/software/python/pytorch_frontier.html#torchrun
    # Thanks to the suggestion by @frobnitzem
    is_rank_setup = int(os.environ.get("RANK", -1)) != -1
    if not is_rank_setup:
        is_srun_used = int(os.environ.get('SLURM_NTASKS',-1)) != -1
        if is_srun_used:
            init_dist_env_with_srun()
        else:
            init_dist_env_with_mpi()

    # --- Initialize distributed environment
    uses_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if uses_dist:
        rank       = int(os.environ["RANK"      ])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend     = dist_backend,
                                rank        = rank,
                                world_size  = world_size,
                                timeout     = timedelta(seconds = 1800),
                                init_method = "env://",)
        print(f"RANK:{rank},LOCAL_RANK:{local_rank},WORLD_SIZE:{world_size}")
    else:
        rank       = 0
        local_rank = 0
        world_size = 1
        print(f"NO distributed environment is required.  RANK:{rank},LOCAL_RANK:{local_rank},WORLD_SIZE:{world_size}")

    # --- Set up GPU device
    gpu_idx = local_rank % torch.cuda.device_count()    # local_rank is node-centric, whereas torch.cuda.device_count() is resource-centeric (on LSF)
    device = f'cuda:{gpu_idx}' if not cpu_only and torch.cuda.is_available() else 'cpu'
    if device != 'cpu': torch.cuda.set_device(device)
    return OmegaConf.create(
        dict(
            uses_dist=uses_dist,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )
    )

def setup_distributed(config):
    dist_config = config.dist
    dist_backend = dist_config.backend
    uses_unique_world_seed = dist_config.uses_unique_world_seed
    dist_dtype = dist_config.dtype

    init_dist_env()
    uses_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if uses_dist:
        dist_rank = int(os.environ["RANK"])
        dist_local_rank = int(os.environ["LOCAL_RANK"])
        dist_world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend=dist_backend,
            rank=dist_rank,
            world_size=dist_world_size,
            timeout=timedelta(seconds=1800),
            init_method="env://",
        )
    else:
        dist_rank = 0
        dist_local_rank = 0
        dist_world_size = 1

    return {
        'uses_dist': uses_dist,
        'rank': dist_rank,
        'local_rank': dist_local_rank,
        'world_size': dist_world_size,
        'backend': dist_backend,
        'dtype': dist_dtype,
        'device': f'cuda:{dist_local_rank}' if not config.misc.cpu_only and torch.cuda.is_available() else 'cpu'
    }
