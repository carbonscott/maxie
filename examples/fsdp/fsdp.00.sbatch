#!/bin/bash
#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lcls:prjdat21         # Check it in your Iris portal: https://iris.nersc.gov
#SBATCH --partition=ampere
#SBATCH --time 12:00:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --exclusive           # Exclusive mode
#SBATCH --job-name=fsdp.02
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2    # In this case, 1 torchrun per node
#!SBATCH --cpus-per-task=10
#SBATCH --mincpus=21

cd /sdf/data/lcls/ds/prj/prjcwang31/results/proj.maxie

echo "sbatch experiments/sbatch/fsdp.02.sbatch"

# ----------------------------------------------------------------------- #
#  Debug
# ----------------------------------------------------------------------- #
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

# ----------------------------------------------------------------------- #
#  Nodes
# ----------------------------------------------------------------------- #
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip=$(echo "$head_node_ip" | awk '{print $1}')

echo "Waiting for the server to fully start..."
sleep 10

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

CPUS_FOR_SERVER=5
CPUS_FOR_CLIENT=16
srun --ntasks=2 --cpus-per-task=$CPUS_FOR_SERVER --gpus=0 --ntasks-per-node=1 --exclusive python server.ipc.py &

sleep 10

## srun --ntasks=2 --cpus-per-task=$CPUS_FOR_CLIENT --gpus-per-task=4 --ntasks-per-node=1 --exclusive python scr.test.py
srun --ntasks=2 --cpus-per-task=$CPUS_FOR_CLIENT --gpus-per-task=4 --ntasks-per-node=1 --exclusive \
torchrun                    \
--nnodes 2                  \
--nproc_per_node 4          \
--rdzv_id $RANDOM           \
--rdzv_backend c10d         \
--rdzv_endpoint $head_node_ip:29500 \
train.fsdp.py experiments/yaml/fsdp.00.yaml
