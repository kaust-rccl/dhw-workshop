#!/bin/bash
#SBATCH -N 2
#SBATCH -t 00:30:00
#SBATCH -J petsc_ex50

module load cray-petsc
time srun -n 4 --hint=nomultithread ./ex50 -da_grid_x 16 -da_grid_y 16 -pc_type mg -da_refine 10 -ksp_monitor -ksp_view -log_view

