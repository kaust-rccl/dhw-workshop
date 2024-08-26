import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly, Lz = 1.0, 1.0, 1.0  # Domain length in x, y, z directions
T = 1.0                      # Total time
Nx, Ny, Nz = 50, 50, 50      # Number of grid points in x, y, z directions
Nt = 200                     # Number of time steps
cx, cy, cz = 0.0, 0.0, 0.0   # Advection velocities in x, y, z directions
D = 0.01                     # Diffusion coefficient

# Discretization
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dz = Lz / (Nz - 1)
dt = T / Nt
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)
u = np.zeros((Nx, Ny, Nz))  # Initial condition: u(x, y, z, 0) = 0
u_new = np.zeros((Nx, Ny, Nz))

# Dirichlet Boundary Conditions
u[0, :, :] = 0.0   # Left boundary (x=0)
u[:, 0, :] = 0.0   # Bottom boundary (y=0)
u[:, :, 0] = 0.0   # Front boundary (z=0)
u[-1, :, :] = 1.0  # Right boundary (x=Lx)
u[:, -1, :] = 1.0  # Top boundary (y=Ly)
u[:, :, -1] = 1.0  # Back boundary (z=Lz)

# Stability conditions
alpha_x = cx * dt / dx
alpha_y = cy * dt / dy
alpha_z = cz * dt / dz
beta_x = D * dt / dx**2
beta_y = D * dt / dy**2
beta_z = D * dt / dz**2

# Time-stepping loop
for n in range(1, Nt):
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                # Explicit finite difference scheme
                u_new[i, j, k] = (u[i, j, k]
                                  - alpha_x * (u[i, j, k] - u[i-1, j, k])
                                  - alpha_y * (u[i, j, k] - u[i, j-1, k])
                                  - alpha_z * (u[i, j, k] - u[i, j, k-1])
                                  + beta_x * (u[i+1, j, k] - 2*u[i, j, k] + u[i-1, j, k])
                                  + beta_y * (u[i, j+1, k] - 2*u[i, j, k] + u[i, j-1, k])
                                  + beta_z * (u[i, j, k+1] - 2*u[i, j, k] + u[i, j, k-1]))

    # Update the solution for the next time step
    u = u_new.copy()

    # Reapply Dirichlet boundary conditions: only needed if DBC ar etime dependent
    u[0, :, :] = 0.0
    u[:, 0, :] = 0.0
    u[:, :, 0] = 1.0
    u[-1, :, :] = 1.0
    u[:, -1, :] = 1.0
    u[:, :, -1] = 1.0

# Visualizing the results: a slice through the middle of the domain
mid_x = Nx // 2
mid_y = Ny // 2
mid_z = Nz // 2

fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131)
ax1.contourf(y, z, u[mid_x, :, :].T, 20, cmap='viridis')
ax1.set_title('Slice at x = Lx/2')
ax1.set_xlabel('y')
ax1.set_ylabel('z')

ax2 = fig.add_subplot(132)
ax2.contourf(x, z, u[:, mid_y, :].T, 20, cmap='viridis')
ax2.set_title('Slice at y = Ly/2')
ax2.set_xlabel('x')
ax2.set_ylabel('z')

ax3 = fig.add_subplot(133)
ax3.contourf(x, y, u[:, :, mid_z].T, 20, cmap='viridis')
ax3.set_title('Slice at z = Lz/2')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
#plt.show()
plt.savefig('solution_3d.pdf')

