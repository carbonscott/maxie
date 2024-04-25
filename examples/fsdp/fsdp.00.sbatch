#!/bin/bash
#SBATCH --output=slurm/%j.log    # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=slurm/%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --account lcls:prjdat21         # Check it in your Iris portal: https://iris.nersc.gov
#!SBATCH --constraint gpu         # Use GPU 
#SBATCH --partition=ampere
#!SBATCH --qos=debug              # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#!SBATCH --time 00:29:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#!SBATCH --qos=regular        # See details: https://docs.nersc.gov/policies/resource-usage/#intended-purpose-of-available-qoss
#SBATCH --time 12:00:00          # Regular only allows a max of 12 hours.  See https://docs.nersc.gov/jobs/policy/
#SBATCH --exclusive           # Exclusive mode
#SBATCH --job-name=fsdp.00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    # In this case, 1 torchrun per node
#SBATCH --cpus-per-task=64

cd /sdf/data/lcls/ds/prj/prjcwang31/results/proj.maxie

echo "sbatch experiments/sbatch/fsdp.00.sbatch"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
head_node_ip=$(echo "$head_node_ip" | awk '{print $1}')

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

# Function to start ipc server
export head_node_ip    # Expose the variable to the following process
start_process() {
    echo "Starting server on $(hostname)..."
    python server.ipc.py &

    sleep 10

    echo "Running client on $(hostname)..."
    torchrun                    \
    --nnodes 1                  \
    --nproc_per_node 4          \
    --rdzv_id $RANDOM           \
    --rdzv_backend c10d         \
    --rdzv_endpoint $head_node_ip:29500 \
    train.fsdp.py experiments/yaml/fsdp.00.yaml
}

export -f start_process
srun --export=ALL bash -c start_process
