#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -n 125
#SBATCH -N 1
#SBATCH --hint=nomultithread

module load python

#time srun -n ${SLURM_NTASKS} -N ${SLURM_NNODES} --hint=nomultithread python3 1D_advection_diffusion_mpi.py

#time srun -n ${SLURM_NTASKS} -N ${SLURM_NNODES} --hint=nomultithread python3 2D_advection_diffusion_mpi.py

time srun -n ${SLURM_NTASKS} -N ${SLURM_NNODES} --hint=nomultithread python3 3D_advection_diffusion_mpi.py
