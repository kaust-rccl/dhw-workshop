import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0        # Length of the domain
T = 1.0        # Total time
Nx = 50        # Number of spatial grid points
Nt = 200       # Number of time steps
c = 0.0        # Advection speed
D = 0.01       # Diffusion coefficient

# Discretization
dx = L / (Nx - 1)
dt = T / Nt
x = np.linspace(0, L, Nx)
u = np.zeros(Nx)  # Initial condition: u(x, 0) = 0
u_new = np.zeros(Nx)

# Dirichlet Boundary Conditions
u[0] = 0.0  # Left boundary condition
u[-1] = 1.0  # Right boundary condition

# Stability condition (for explicit methods)
alpha = c * dt / dx
beta = D * dt / (dx ** 2)

# Time-stepping loop
for n in range(1, Nt):
    for i in range(1, Nx-1):
        # Explicit finite difference scheme
        u_new[i] = u[i] - alpha * (u[i] - u[i-1]) + beta * (u[i+1] - 2*u[i] + u[i-1])
    
    # Update the solution for the next time step
    u = u_new.copy()
    
    # Apply Dirichlet boundary conditions again: only needed if DBC ar etime dependent
    u[0] = 0.0
    u[-1] = 1.0

# Plotting the results
plt.plot(x, u, label=f't = {T}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('1D Advection-Diffusion Equation')
plt.legend()
#plt.show()
plt.savefig('solution_1d.pdf')

