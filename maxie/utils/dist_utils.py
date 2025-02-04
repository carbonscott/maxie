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
from mpi4py import MPI

def init_dist_env():
    """Initialize distributed environment using MPI."""
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

def setup_distributed(config):
    dist_config = config.get("dist")
    dist_backend = dist_config.get("backend")
    uses_unique_world_seed = dist_config.get("uses_unique_world_seed")
    dist_dtype = dist_config.get("dtype")

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
        'dtype': dist_dtype
    }
