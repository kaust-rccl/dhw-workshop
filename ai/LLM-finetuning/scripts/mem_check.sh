#!/bin/bash

#SBATCH --job-name=memcheck
#SBATCH --tasks-per-node=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --account=c2227
#SBATCH --constraint=rtx2080ti

scontrol show job ${SLURM_JOBID}
rm -rf /home/shaima0d/.cache/torch_extensions/*
source /ibex/user/shaima0d/miniconda3/bin/activate /ibex/ai/home/shaima0d/KSL_Trainings/hpc-saudi-2024/ds-env
export CUDA_HOME=${CONDA_PREFIX}
export OMP_NUM_THREADS=1
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
export PYTHONWARNINGS=ignore

export master_ip=$(/bin/hostname -I | cut -d " " -f 2 )
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

export HF_HOME=$PWD/cache
export HF_METRICS_CACHE=$PWD/cache
mkdir -p $HF_HOME


python ./scripts/memory_req.py
