# Poisson Equation Solver using PETSc

## Overview

This repository contains an implementation of a Poisson equation solver using PETSc. The implementation files include source code (`ex50.c`), a `makefile` for building the code, and a `job.sh` script for running the code on a cluster using SLURM scheduling. This README provides instructions for building, running, and using the provided scripts.

## Prerequisites

- PETSc library installed on your system.
- A compatible MPI installation.
- SLURM workload manager (for using the `job.sh` script).

## Files in this Repository

- **`ex50.c`**: The main source file implementing the Poisson equation solver.
- **`makefile`**: Build file to compile the `ex50.c` source code using PETSc.
- **`job.sh`**: SLURM script for running the executable on a compute cluster.

## Instructions

### 1. Building the Code

To compile the code, you can use the provided `makefile`. Make sure you have PETSc correctly configured and loaded (if using a module system).

1. Open a terminal and navigate to the directory containing the files.
2. Compile the code using:
   ```bash
   make ex50
   ```
3. Ensure there are no compilation errors before proceeding.

### 2. Running the Code Locally

After compiling, you can run the `ex50` executable locally using MPI:

1. Run the code with MPI:
   ```bash
   mpirun -n 4 ./ex50 -da_grid_x 16 -da_grid_y 16 -pc_type mg -da_refine 10 -ksp_monitor -ksp_view -log_view
   ```
   - Adjust the number of processes (`-n 4`) and other parameters as needed.

### 3. Running the Code on a Cluster

The provided `job.sh` script is configured for running the code on a cluster using SLURM.

1. Ensure the `job.sh` file is executable:
   ```bash
   chmod +x job.sh
   ```
2. Submit the job using SLURM:
   ```bash
   sbatch job.sh
   ```
3. The `job.sh` script contains the following settings:
   - **`-N 2`**: Number of nodes.
   - **`-t 00:30:00`**: Maximum run time.
   - **`-J petsc_ex50`**: Job name.
   - **Modules and Run Command**: Loads `cray-petsc` module and runs `ex50` using `srun` with specified grid and solver options.

### Customization

You can modify the parameters in the `job.sh` script and the run command options in `mpirun` as per your computational needs and available resources.
