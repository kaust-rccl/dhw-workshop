from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
my_hostname=MPI.Get_processor_name()

X = np.empty(4,dtype='i')
x = np.empty(1,dtype='i')

if rank == 0:
    X = np.random.randint(low=0,high=10,size=4,dtype='i')
    print(f"MPI rank {rank} of {size} processes on machine {my_hostname} has X: {X}")
    
comm.Scatter(X,x, root=0)
print(f"MPI rank {rank} of {size} processes on machine {my_hostname} has x: {x}")

