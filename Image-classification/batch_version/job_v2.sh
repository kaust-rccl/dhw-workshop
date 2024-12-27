#!/bin/bash -l

#SBATCH -J bench-tinyIM-ds
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH -n 1
#SBATCH --tasks-per-node=1
#SBATCH -c 6
#SBATCH --mem=100G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH -t 04:0:0
#SBATCH -A c2227
#SBATCH -C rtx2080ti

export EPOCHS=3

scontrol show job ${SLURM_JOBID}
#rm -rf /home/shaima0d/.cache/torch_extensions/*
source /ibex/user/shaima0d/miniconda3/bin/activate /ibex/ai/home/shaima0d/KSL_Trainings/hpc-saudi-2024/ds-env
export CUDA_HOME=${CONDA_PREFIX}

export TORCH_EXTENSIONS_DIR=${PWD}/torch_extension_${SLURM_JOB_NAME}
mkdir -p $TORCH_EXTENSIONS_DIR

export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0 
export NCCL_SOCKET_IFNAME=ib0
export NCCL_NET_GDR_LEVEL=4


export OMP_NUM_THREADS=1
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Node IDs of participating nodes ${nodes_array[*]}"


# Get the IP address and set port for MASTER node
head_node="${nodes_array[0]}"
echo "Getting the IP address of the head node ${head_node}"
export master_ip=$(srun -n 1 -N 1 --gpus=1 -w ${head_node} /bin/hostname -I | cut -d " " -f 2)
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo "head node is ${master_ip}:${MASTER_PORT}"

workers=${SLURM_CPUS_PER_TASK}

echo "Hostname: $(/bin/hostname)"
echo "CPU workers: $workers"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

for ((i = 0; i < ${SLURM_NNODES}; i++)); do
   node_i=${nodes_array[$i]}
   echo $node_i slots=${SLURM_GPUS_PER_NODE} >> hostfile
done
start=$(date +%s)
deepspeed --num_nodes=${SLURM_NNODES} --num_gpus=${SLURM_GPUS} \
          --master_addr=${master_ip} \
          --hostfile ./hostfile --no_ssh \
           ../scripts/train_resnet50_ds.py \
	--deepspeed --deepspeed_config /ibex/ai/home/shaima0d/KSL_Trainings/hpc-saudi-2024/tiny_imagenet_ds/scripts/ds_config.json \
        --epochs ${EPOCHS} --num-workers=${SLURM_CPUS_PER_TASK}\
        --log-interval 50
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"
