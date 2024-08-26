from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
Lx, Ly, Lz = 1.0, 1.0, 1.0   # Length of the domain in x, y, and z
Nx, Ny, Nz = 50, 50, 50      # Number of spatial points in x, y, and z
dx = Lx / (Nx - 1)           # Spatial step size in x
dy = Ly / (Ny - 1)           # Spatial step size in y
dz = Lz / (Nz - 1)           # Spatial step size in z
vx, vy, vz = 1.0, 1.0, 1.0   # Advection velocities in x, y, and z
D = 0.01                     # Diffusion coefficient
dt = 0.001                   # Time step size
Nt = 1000                    # Number of time steps

# Boundary conditions
u_x0, u_xL = 0.0, 1.0        # Boundary values at x = 0 and x = Lx
u_y0, u_yL = 0.0, 1.0        # Boundary values at y = 0 and y = Ly
u_z0, u_zL = 0.0, 1.0        # Boundary values at z = 0 and z = Lz

# Initial condition function
def initial_condition(x, y, z):
    return np.sin(np.pi * x / Lx) * np.sin(np.pi * y / Ly) * np.sin(np.pi * z / Lz)

# Determine local grid size (assuming perfect cubic grid decomposition)
local_Nx = Nx // int(np.cbrt(size))
local_Ny = Ny // int(np.cbrt(size))
local_Nz = Nz // int(np.cbrt(size))
x_start = (rank % int(np.cbrt(size))) * local_Nx
y_start = ((rank // int(np.cbrt(size))) % int(np.cbrt(size))) * local_Ny
z_start = (rank // (int(np.cbrt(size))**2)) * local_Nz

# Local grid
x_local = np.linspace(x_start * dx, (x_start + local_Nx) * dx, local_Nx)
y_local = np.linspace(y_start * dy, (y_start + local_Ny) * dy, local_Ny)
z_local = np.linspace(z_start * dz, (z_start + local_Nz) * dz, local_Nz)
X_local, Y_local, Z_local = np.meshgrid(x_local, y_local, z_local, indexing='ij')

# Initial state on each process
u_local = initial_condition(X_local, Y_local, Z_local)

# Time-stepping loop
for n in range(1, Nt):
    u_new_local = np.copy(u_local)
    
    # Interior points update
    for i in range(1, local_Nx - 1):
        for j in range(1, local_Ny - 1):
            for k in range(1, local_Nz - 1):
                u_new_local[i, j, k] = u_local[i, j, k] \
                    - vx * dt / (2 * dx) * (u_local[i+1, j, k] - u_local[i-1, j, k]) \
                    - vy * dt / (2 * dy) * (u_local[i, j+1, k] - u_local[i, j-1, k]) \
                    - vz * dt / (2 * dz) * (u_local[i, j, k+1] - u_local[i, j, k-1]) \
                    + D * dt * ((u_local[i+1, j, k] - 2*u_local[i, j, k] + u_local[i-1, j, k]) / dx**2 \
                              + (u_local[i, j+1, k] - 2*u_local[i, j, k] + u_local[i, j-1, k]) / dy**2 \
                              + (u_local[i, j, k+1] - 2*u_local[i, j, k] + u_local[i, j, k-1]) / dz**2)
    # Exchange boundary data between processes
    # x-direction
    if rank % int(np.cbrt(size)) > 0:  # Not left boundary
        comm.send(u_local[1, :, :], dest=rank-1)
        u_new_local[0, :, :] = comm.recv(source=rank-1)
    if rank % int(np.cbrt(size)) < int(np.cbrt(size)) - 1:  # Not right boundary
        comm.send(u_local[-2, :, :], dest=rank+1)
        u_new_local[-1, :, :] = comm.recv(source=rank+1)
    
    # y-direction
    if (rank // int(np.cbrt(size))) % int(np.cbrt(size)) > 0:  # Not front boundary
        comm.send(u_local[:, 1, :], dest=rank-int(np.cbrt(size)))
        u_new_local[:, 0, :] = comm.recv(source=rank-int(np.cbrt(size)))
    if (rank // int(np.cbrt(size))) % int(np.cbrt(size)) < int(np.cbrt(size)) - 1:  # Not back boundary
        comm.send(u_local[:, -2, :], dest=rank+int(np.cbrt(size)))
        u_new_local[:, -1, :] = comm.recv(source=rank+int(np.cbrt(size)))
    
    # z-direction
    if rank // (int(np.cbrt(size))**2) > 0:  # Not bottom boundary
        comm.send(u_local[:, :, 1], dest=rank-int(np.cbrt(size))**2)
        u_new_local[:, :, 0] = comm.recv(source=rank-int(np.cbrt(size))**2)
    if rank // (int(np.cbrt(size))**2) < int(np.cbrt(size)) - 1:  # Not top boundary
        comm.send(u_local[:, :, -2], dest=rank+int(np.cbrt(size))**2)
        u_new_local[:, :, -1] = comm.recv(source=rank+int(np.cbrt(size))**2)
    
    # Apply boundary conditions
    if x_start == 0:
        u_new_local[0, :, :] = u_x0
    if x_start + local_Nx == Nx:
        u_new_local[-1, :, :] = u_xL
    if y_start == 0:
        u_new_local[:, 0, :] = u_y0
    if y_start + local_Ny == Ny:
        u_new_local[:, -1, :] = u_yL
    if z_start == 0:
        u_new_local[:, :, 0] = u_z0
    if z_start + local_Nz == Nz:
        u_new_local[:, :, -1] = u_zL
    
    # Update the solution
    u_local = np.copy(u_new_local)

# Gather the data on the root process
u_global = None
if rank == 0:
    u_global = np.empty((Nx, Ny, Nz), dtype=np.float64)

comm.Gather(u_local, u_global, root=0)

# Plot the final solution on the root process (2D slice for visualization)
if rank == 0:
    slice_z = Nz // 2
    X_global, Y_global = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
    plt.contourf(X_global, Y_global, u_global[:, :, slice_z], 20, cmap='viridis')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Slice of 3D Advection-Diffusion Equation at z={}'.format(slice_z * dz))
    #plt.show()
    plt.savefig('solution_3d.pdf')
