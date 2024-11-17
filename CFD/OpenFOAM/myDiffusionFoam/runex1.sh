#!/bin/bash

# Navigate to the code directory and compile the code
cd code || { echo "Error: 'code' directory not found!"; exit 1; }
echo "Cleaning previous compilations..."
wclean ./all > ../log.wclean 2>&1
echo "Compiling the code..."
wmake > ../log.wmake 2>&1

# Navigate to the testcase directory and run the selected test case
cd ../testcase/3DSimple || { echo "Error: 'testcase/3DSimple' directory not found!"; exit 1; }

# Delete existing log files
echo "Deleting existing log files..."
rm -f log.*  # Removes all log files (e.g., log.blockMesh, log.simulation, etc.)
echo "Existing log files deleted."

# Clean previous simulation files and processor directories
echo "Cleaning previous simulation files..."
find . -maxdepth 1 -type d -regex './[1-5]+' -not -name '0.org' -exec rm -rf {} +  # Removes directories like 1, 2, ..., 5 but keeps 0.org
find . -maxdepth 1 -type d -regex './[0-9]+\.[0-9]+' -exec rm -rf {} +             # Removes directories like 0.1, 0.2, etc.
rm -rf processor*    # Removes all processor directories from previous parallel runs

# Copy 0.org to 0 directory
if [ -d "0.org" ]; then
    echo "Copying '0.org' to '0'..."
    rm -rf 0       # Remove existing '0' directory if it exists
    cp -r 0.org 0  # Copy '0.org' to '0'
    echo "Copy complete. Original files are intact in '0.org'."
else
    echo "Error: '0.org' directory not found! Ensure it exists before running this script."
    exit 1
fi

# Running the test case
echo "Generating the mesh..."
blockMesh > log.blockMesh 2>&1
if [ $? -ne 0 ]; then
    echo "Error: blockMesh failed. Check log.blockMesh for details."
    exit 1
fi

echo "Checking mesh quality..."
checkMesh > log.checkMesh 2>&1
if [ $? -ne 0 ]; then
    echo "Warning: Mesh quality check reported issues. Check log.checkMesh for details."
fi

# Decompose the domain for parallel processing
echo "Decomposing the domain..."
decomposePar > log.decomposePar 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Domain decomposition failed. Check log.decomposePar for details."
    exit 1
fi

# Replace 'n' with the desired number of processors from decomposeParDict
NUM_PROCESSORS=$(grep -oP '(?<=numberOfSubdomains\s+)\d+' system/decomposeParDict)
echo "Running the simulation with $NUM_PROCESSORS processors..."
mpirun -np "$NUM_PROCESSORS" myDiffusionFoam > log.simulation 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Simulation failed. Check log.simulation for details."
    exit 1
fi

echo "Simulation completed successfully!"
