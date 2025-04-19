#!/bin/bash

#SBATCH --job-name=finetune
#SBATCH --tasks-per-node=1
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:10:00
#SBATCH --account=c2227
#SBATCH --constraint=rtx2080ti

scontrol show job ${SLURM_JOBID}
source /ibex/user/shaima0d/miniconda3/bin/activate /ibex/ai/home/shaima0d/KSL_Trainings/hpc-saudi-2024/ds-env
export CUDA_HOME=${CONDA_PREFIX}
export OMP_NUM_THREADS=1
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
export PYTHONWARNINGS=ignore

export master_ip=$(/bin/hostname -I | cut -d " " -f 2 )
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export nvport=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

nvdashboard $nvport &
echo "ssh -L $nvport:$(/bin/hostname -s):$nvport $USER@glogin.ibex.kaust.edu.sa"

export BASEDIR=$(echo ${PWD}/cache)
export HF_HOME=${BASEDIR}
export HF_METRICS_CACHE=${BASEDIR}/metric
export TRITON_CACHE_DIR=${BASEDIR}/triton
export TORCH_EXTENSIONS_DIR=${BASEDIR}
mkdir -p ${HF_METRICS_CACHE} ${TRITON_CACHE_DIR} ${HF_HOME}
export HUGGINGFACE_HUB_VERBOSITY=debug
export TOKENIZERS_PARALLELISM=false
rm -rf ${TORCH_EXTENSIONS_DIR}/*
cd ../scripts

export DS_CONFIG='ds_config_nozero.json'
start=$(date +%s)
srun -l ./wrapper.sh
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"

cd ..
