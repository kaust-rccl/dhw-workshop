#!/bin/bash

#SBATCH -p batch
#SBATCH -C t4g4
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:20:00
#SBATCH -c 48

module load python

nvidia-smi --query-gpu=pci.bus_id,utilization.gpu,utilization.memory --format=csv -l 5 -i $CUDA_VISIBLE_DEVICES &>  mon.log &

torchrun --nproc-per-node=${SLURM_GPUS_PER_NODE} --nnodes=${SLURM_NNODES} cifar10_ddp.py --num-workers=${SLURM_CPUS_PER_TASK}

kill $(pidof nvidia-smi)
