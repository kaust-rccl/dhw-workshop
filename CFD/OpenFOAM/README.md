# Diffusion and Convection-Diffusion Foam Codes

## Overview

This repository contains implementations of Diffusion and Convection-Diffusion codes using OpenFOAM. The code includes examples demonstrating its functionality, which can be accessed in the `testcase` directory. This README provides instructions to ensure the code is functioning correctly and explains how to run the provided examples.

## Prerequisites

Make sure you have OpenFOAM installed and properly set up on your system. 

## Instructions

### 1. Code Compilation

First, ensure that the code is compiled correctly:

1. Navigate to the `code` directory:
   ```bash
   cd code
   ```
2. Clean any previous compilation:
   ```bash
   wclean ./all
   ```
3. Compile the code:
   ```bash
   wmake
   ```
   Ensure there are no compilation errors before proceeding.

### 2. Running the Test Cases

Navigate to the `testcase` directory to find the example cases.

#### General Steps:

1. Go to the specific example directory:
   ```bash
   cd testcase/example1  # or example2 for the second case
   ```
2. Generate the mesh:
   ```bash
   blockMesh
   ```
3. Check the mesh quality:
   ```bash
   checkMesh
   ```
   This command will evaluate the mesh quality and notify you of any potential issues.

4. **For Example 2 Only**: Set complex boundary conditions using:
   ```bash
   funkySetFields -time 0
   ```

### 3. Parallel Simulation

1. Decompose the domain for parallel processing. Modify the number of processors as needed in `decomposeParDict`:
   ```bash
   decomposePar
   ```
2. Run the simulation using MPI:
   ```bash
   mpirun -np n myDiffusionFoam/myConvectionDiffusionFoam
   ```
   Replace `n` with the number of processors specified in `decomposeParDict`.

### 4. Bash Automation

For a smoother workflow, a bash script is provided for each simulation. This script allows you to run the entire process with a single command.

