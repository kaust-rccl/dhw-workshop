# Solving 2D Heat eaquation with MPI 

## Compilation of C version

For compiling, the following are required:
- a C compiler (e.g. GCC/Intel/clang etc)
- MPI library (e.g. OpenMPI, IntelMPI etc)

A conda environment file is provided for convinience to create the software environment. 

`conda env create -f software_stack.yaml -p $PWD/install`

`conda activate ./install`

`cd C/`

`make VERBOSE=1`

## Running C version

To execute the serial version please use the following command line:

`./heat_serial`

And for distributed memory runs:
 
`mpirun -np <num_mpi_ranks> ./heat_mpi_2d`


The following conditions shall be adhered when setting parameters:

- Parameter N controls the mesh or grid points. It should a perfect square.
- Parameter MAX_ITER is the maximum number of iterations the solver should run. The loop break if either Tolerance TOL is reached or MAX_ITER.
- Parameter TOL controls the precision of the error that needs to be minimized by the solver
- Parameter MAX_TEM is the maximum temprature at set as Dirichlet boundary condition

Note that the larger the N, the more memory will be consumed.

With default settings of N=64, MAX_ITER=4000, TOL=1e-4 and MAX_TEMP=100.0, the solution converges in 4000 iterations with an error of 0.0002

