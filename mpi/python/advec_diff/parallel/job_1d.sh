#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --hint=nomultithread

module load python

time srun -n ${SLURM_NTASKS} -N ${SLURM_NNODES} --hint=nomultithread python3 1D_advection_diffusion_mpi.py

