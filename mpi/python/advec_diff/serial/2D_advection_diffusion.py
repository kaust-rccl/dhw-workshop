import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 1.0, 1.0  # Domain length in x and y directions
T = 1.0            # Total time
Nx, Ny = 50, 50    # Number of grid points in x and y directions
Nt = 200           # Number of time steps
cx, cy = 0.0, 0.0  # Advection velocities in x and y directions
D = 0.01           # Diffusion coefficient

# Discretization
dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)
dt = T / Nt
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
u = np.zeros((Nx, Ny))  # Initial condition: u(x, y, 0) = 0
u_new = np.zeros((Nx, Ny))

# Dirichlet Boundary Conditions
u[0, :] = 0.0   # Left boundary (x=0)
u[:, 0] = 0.0   # Bottom boundary (y=0)
u[-1, :] = 1.0  # Right boundary (x=Lx)
u[:, -1] = 1.0  # Top boundary (y=Ly)

# Stability conditions
alpha_x = cx * dt / dx
alpha_y = cy * dt / dy
beta_x = D * dt / dx**2
beta_y = D * dt / dy**2

# Time-stepping loop
for n in range(1, Nt):
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            # Explicit finite difference scheme
            u_new[i, j] = (u[i, j] 
                           - alpha_x * (u[i, j] - u[i-1, j])
                           - alpha_y * (u[i, j] - u[i, j-1])
                           + beta_x * (u[i+1, j] - 2*u[i, j] + u[i-1, j])
                           + beta_y * (u[i, j+1] - 2*u[i, j] + u[i, j-1]))

    # Update the solution for the next time step
    u = u_new.copy()

    # Apply Dirichlet boundary conditions again: : only needed if DBC ar etime dependent
    u[0, :] = 0.0
    u[:, 0] = 0.0
    u[-1, :] = 1.0
    u[:, -1] = 1.0

# Plotting the results
X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, u.T, 20, cmap='viridis')
plt.colorbar(label='u(x, y, t)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('2D Advection-Diffusion Equation')
#plt.show()
plt.savefig('solution_2d.pdf')

