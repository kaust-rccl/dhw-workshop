#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --hint=nomultithread

module load python

time python3 2D_advection_diffusion.py

