# usage: mpirun -np 2 python advection_diffusion_mpi.py

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
L = 1.0           # Length of the domain
Nx = 10           # Number of spatial points
dx = L / (Nx - 1) # Spatial step size
v = 0.0           # Advection velocity
D = 1.0           # Diffusion coefficient
dt = 0.001        # Time step size
Nt = 10           # Number of time steps

# Boundary conditions
u_0 = 0.0        # Boundary value at x = 0
u_L = 1.0        # Boundary value at x = L

# Initial condition function
def initial_condition(x):
    return np.zeros(x.shape[0]) #np.sin(np.pi * x / L)

# Divide the domain among processes
local_Nx = Nx // size  # Number of points per process
x_start = rank * local_Nx
x_end = (rank + 1) * local_Nx if rank != size - 1 else Nx
x_local = np.linspace(x_start * dx, x_end * dx, local_Nx)

# Initial state on each process
u_local = initial_condition(x_local)

#print(rank, u_local)

# Time-stepping loop
for n in range(1, Nt):
    u_new_local = np.copy(u_local)
    
    # Interior points update
    for i in range(1, local_Nx - 1):
        u_new_local[i] = u_local[i] - v * dt / (2 * dx) * (u_local[i+1] - u_local[i-1]) \
                         + D * dt / dx**2 * (u_local[i+1] - 2*u_local[i] + u_local[i-1])
    
    # Exchange boundary data between processes
    if rank > 0:
        comm.send(u_local[1], dest=rank-1)
        u_new_local[0] = comm.recv(source=rank-1)
    if rank < size - 1:
        comm.send(u_local[-2], dest=rank+1)
        u_new_local[-1] = comm.recv(source=rank+1)
    
    # Apply boundary conditions
    if rank == 0:
        u_new_local[0] = u_0
    if rank == size - 1:
        u_new_local[-1] = u_L
    
    # Update the solution
    u_local = np.copy(u_new_local)

#print(rank, u_local)

# Gather the data on the root process
u_global = None
if rank == 0:
    u_global = np.empty(Nx, dtype=np.float64)

comm.Gather(u_local, u_global, root=0)


# Plot the final solution on the root process
if rank == 0:
    x_global = np.linspace(0, L, Nx)
    plt.plot(x_global, u_global, label=f't = {Nt*dt:.2f}')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('1D Advection-Diffusion Equation with MPI')
    plt.legend()
    plt.savefig('solution_1d.pdf')
    #print(rank, u_global)
