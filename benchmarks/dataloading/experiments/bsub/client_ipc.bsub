#!/bin/bash
#BSUB -o lsf/%J.log
#BSUB -e lsf/%J.err
#BSUB -q debug
#BSUB -W 2:00
#BSUB -P LRN044
#BSUB -J client_ipc
#BSUB -nnodes 10

export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/

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

echo "Starting server..."
jsrun \
--nrs 10 \
--tasks_per_rs 1 \
--cpu_per_rs 6 \
--gpu_per_rs 0 \
--rs_per_host 1 \
--latency_priority cpu-cpu \
--launch_distribution packed \
python server.ipc.py --num_workers 2 &

sleep 10

echo "Running client..."
jsrun \
--rs_per_host 6 \
--tasks_per_rs 1 \
--cpu_per_rs 4 \
--gpu_per_rs 1 \
--latency_priority gpu-gpu \
--launch_distribution packed \
python client.ipc.py experiments/yaml/client_ipc.yaml
