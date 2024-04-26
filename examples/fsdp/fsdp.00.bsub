#!/bin/bash
#BSUB -o lsf/%J.log
#BSUB -e lsf/%J.err
#!BSUB -q batch
#!BSUB -q batch-hm
#BSUB -q debug
#BSUB -W 2:00
#BSUB -P LRN044
#BSUB -J fsdp2.00
#!BSUB -alloc_flags gpudefault
#BSUB -nnodes 1

cd /gpfs/alpine2/proj-shared/lrn044/foundation_models/results/cwang31

echo "sbatch experiments/bsub/fsdp.00.bsub"

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

# Fetch all nodes and output a whole string of concatenated host nodes
# $LSB_MCPU_HOSTS gives something like "batch02 1 a09n03 42 a10n04 42".
# I need just "a09n03 a10n04" to set up a head node.
nodelist=$(echo $LSB_MCPU_HOSTS | awk '{for (i=3; i<=NF; i+=2) print $i}' | sort | uniq)    # "a09n03 a10n04"
read -r -a nodes <<< "$nodelist"
head_node=${nodes[0]}
head_node_ip=$(ssh "$head_node" hostname --ip-address)
head_node_ip=$(echo "$head_node_ip" | awk '{print $1}')

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

## CPUS_FOR_SERVER=5
## CPUS_FOR_CLIENT=16

echo "Starting server..."
jsrun \
--nrs 1 \
--tasks_per_rs 1 \
--cpu_per_rs 5 \
--gpu_per_rs 0 \
--rs_per_host 1 \
--latency_priority cpu-cpu \
--launch_distribution packed \
python server.ipc.py &

sleep 10

echo "Running client..."
jsrun \
--nrs 1 \
--tasks_per_rs 1 \
--cpu_per_rs 36 \
--gpu_per_rs 6 \
--rs_per_host 1 \
--latency_priority gpu-gpu \
--launch_distribution packed \
torchrun                    \
--nnodes 1                  \
--nproc_per_node 6          \
--rdzv_id $RANDOM           \
--rdzv_backend c10d         \
--rdzv_endpoint $head_node_ip:29500 \
train.fsdp.py experiments/yaml/fsdp.00.yaml
