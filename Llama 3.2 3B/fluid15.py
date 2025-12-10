# filename: legacy_fenics.py

import numpy as np
from fenics import *

# Define the mesh and domain
mesh = UnitSquareMesh(64, 64)
cell_centered = False

# Create a function space for the velocity and pressure fields
V = FunctionSpace(mesh, 'CG', 2)
P = FunctionSpace(mesh, 'CG', 1)

# Define the boundary conditions
def periodic_boundary(x):
    return x[0] % 1, x[1] % 1

bc_u = DirichletBC(V, Constant(0), 'on_boundary')
bc_v = DirichletBC(V, Constant(0), 'on_boundary')

# Define the initial condition for velocity
def u_initial(x):
    return sin(2 * np.pi * x[0]) * cos(2 * np.pi * x[1])

u0 = Function(V)
u0.interpolate(u_initial)

# Define the time-dependent function for pressure
def p_initial(t):
    return 1 + t

p0 = Function(P)
p0.interpolate(p_initial)

# Set parameters
rho = 1.0
nu = 1e-3
t_final = 1.0
dt = min(nu / 10, 1 / (np.sqrt(2) * np.pi)) # choose a small enough time step

# Create a time-stepping loop
t = 0
while t < t_final:
    # Solve the Navier-Stokes equations
    u, p = solve(NavierStokes(rho, nu, V, P, bc_u, bc_v), (u0, p0))

    # Update the initial conditions for velocity and pressure
    u0.assign(u)
    p0.assign(p)

    # Increment time
    t += dt

    # Save the velocity field at specific times
    if t == 0 or t == 0.25 * dt or t == 0.5 * dt or t == t_final:
        file_name = f"velocity_{t:.2f}.xdmf"
        with XDMFFile(file_name, mode='w') as xf:
            xf.write_mesh(mesh)
            for i in range(1, len(V.subcells())):
                v = Function(V)
                v.interpolate(u, 0)
                xf.append_entry(mesh, v, 'velocity', 0)

# Print the final velocity field
print("Final velocity field:")
print(u)