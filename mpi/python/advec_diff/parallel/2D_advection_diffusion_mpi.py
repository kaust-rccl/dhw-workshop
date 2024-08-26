from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
Lx, Ly = 1.0, 1.0          # Length of the domain in x and y
Nx, Ny = 100, 100          # Number of spatial points in x and y
dx = Lx / (Nx - 1)         # Spatial step size in x
dy = Ly / (Ny - 1)         # Spatial step size in y
vx, vy = 0.0, 0.0          # Advection velocities in x and y
D = 1.0                   # Diffusion coefficient
dt = 0.001                 # Time step size
Nt = 5000                  # Number of time steps

# Boundary conditions
u_x0, u_xL = 0.0, 1.0      # Boundary values at x = 0 and x = Lx
u_y0, u_yL = 0.0, 1.0      # Boundary values at y = 0 and y = Ly

# Initial condition function
def initial_condition(x, y):
    return np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly)

# Determine local grid size (assuming perfect square grid decomposition)
local_Nx = Nx // int(np.sqrt(size))
local_Ny = Ny // int(np.sqrt(size))
x_start = (rank % int(np.sqrt(size))) * local_Nx
y_start = (rank // int(np.sqrt(size))) * local_Ny

# Local grid
x_local = np.linspace(x_start * dx, (x_start + local_Nx) * dx, local_Nx)
y_local = np.linspace(y_start * dy, (y_start + local_Ny) * dy, local_Ny)
X_local, Y_local = np.meshgrid(x_local, y_local)

# Initial state on each process
u_local = initial_condition(X_local, Y_local)
#print(rank, u_local)
#print(rank,u_local.shape[0],u_local.shape[1])
#for i in range(0, u_local.shape[0]):
#    for j in range(0, u_local.shape[1]):
#        print(rank,i,j,u_local[i,j])


# Time-stepping loop
for n in range(1, Nt):
    u_new_local = np.copy(u_local)

    # Interior points update
    for i in range(1, local_Nx - 1):
        for j in range(1, local_Ny - 1):
            u_new_local[i, j] = u_local[i, j] \
                                - vx * dt / (2 * dx) * (u_local[i+1, j] - u_local[i-1, j]) \
                                - vy * dt / (2 * dy) * (u_local[i, j+1] - u_local[i, j-1]) \
                                + D * dt * ((u_local[i+1, j] - 2*u_local[i, j] + u_local[i-1, j]) / dx**2 \
                                           + (u_local[i, j+1] - 2*u_local[i, j] + u_local[i, j-1]) / dy**2)
    # Exchange boundary data between processes
    # x-direction
    if rank % int(np.sqrt(size)) > 0:  # Not left boundary
        comm.send(u_local[1, :], dest=rank-1)
        u_new_local[0, :] = comm.recv(source=rank-1)
    if rank % int(np.sqrt(size)) < int(np.sqrt(size)) - 1:  # Not right boundary
        comm.send(u_local[-2, :], dest=rank+1)
        u_new_local[-1, :] = comm.recv(source=rank+1)
    
    # y-direction
    if rank // int(np.sqrt(size)) > 0:  # Not bottom boundary
        comm.send(u_local[:, 1], dest=rank-int(np.sqrt(size)))
        u_new_local[:, 0] = comm.recv(source=rank-int(np.sqrt(size)))
    if rank // int(np.sqrt(size)) < int(np.sqrt(size)) - 1:  # Not top boundary
        comm.send(u_local[:, -2], dest=rank+int(np.sqrt(size)))
        u_new_local[:, -1] = comm.recv(source=rank+int(np.sqrt(size)))
    
    # Apply boundary conditions
    if x_start == 0:
        u_new_local[:, 0] = u_x0
    if x_start + local_Nx == Nx:
        u_new_local[:, -1] = u_xL
    if y_start == 0:
        u_new_local[0, :] = u_y0
    if y_start + local_Ny == Ny:
        u_new_local[-1, :] = u_yL
    
    # Update the solution
    u_local = np.copy(u_new_local)

# Gather the data on the root process
u_global = None
if rank == 0:
    u_global = np.empty((Nx, Ny), dtype=np.float64)

comm.Gather(u_local, u_global, root=0)

# Plot the final solution on the root process
if rank == 0:
    print(n)
    X_global, Y_global = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
    plt.contourf(X_global, Y_global, u_global, 20, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Advection-Diffusion Equation with MPI')
    #plt.show()
    plt.savefig('solution_2d.pdf')

