import os
import socket
import torch
import torch.distributed as dist

def init_dist_env_on_nersc():
    srun_rank = int(os.environ.get('SLURM_PROCID',0))
    srun_local_rank = int(os.environ.get('SLURM_LOCALID',0))
    srun_size = int(os.environ.get('SLURM_NTASKS',1))

    # Are we using multiple ranks?
    uses_dist = srun_size > 1

    if uses_dist:
        MAIN_RANK = 0

        # MPI environment variables detected (e.g., Summit)
        os.environ["WORLD_SIZE"] = str(srun_size)
        os.environ["RANK"]       = str(srun_rank)
        os.environ["LOCAL_RANK"] = str(srun_local_rank)

        # Set the default master address and port, prioritizing definition in the job script
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "29500")

        master_addr = os.getenv("MASTER_ADDR", None)
        print(f"Environment setup for distributed computation: WORLD_SIZE={os.environ['WORLD_SIZE']}, RANK={os.environ['RANK']}, LOCAL_RANK={os.environ['LOCAL_RANK']}, MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
