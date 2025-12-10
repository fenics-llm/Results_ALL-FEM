# filename: transport_chemical.py

import numpy as np
from fenics import *
from ufl_legacy import inner, dot

# Constants
R0 = 0.05  # Initial radius (m)
s = 1e-4   # Constant rate of expansion (m/s)
D = 1e-5   # Diffusivity (m² s⁻¹)
kappa = 1e-4  # Decay rate (s⁻¹)
dt = 0.01  # Time step (s)
t_final = 10  # Final time (s)

# Create mesh
mesh = UnitDiscMesh unit_sphere(1, 'unit_sphere.h5')
cell_volume = mesh.cell_volume()

# Define function spaces
V = FunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Initialize solution and test functions
c = Function(V)
w = Function(Q)

# Define boundary conditions
bc_left = DirichletBC(V, Constant(0), 'left')
bc_right = DirichletBC(V, Constant(0), 'right')

# Define initial condition
def init_condition(x):
    return 1.0

c.set_initial_value(init_condition, 0)

# Compute time-dependent radius
R = R0 + s * dt

# Create boundary function for moving boundary
def boundary(x):
    if x[0] == 0:
        return 0
    else:
        return w(x) / np.linalg.norm(x)

# Define boundary condition for moving boundary
bc_move = DirichletBC(V, boundary, 'left')

# Assemble and solve the equation
ufl_c = c
ufl_w = w

def assemble_and_solve():
    global ufl_c, ufl_w
    
    # Compute time-dependent diffusion coefficient
    D_t = D * (1 + s * dt)

    # Create variational problem
    F = inner(D_t * ufl_c, ufl_c) / cell_volume + kappa * ufl_c
    L = dot(w, c)
    
    # Assemble and solve the equation
    a = inner(ufl_c, ufl_w) / cell_volume
    b = inner(D_t * ufl_c, w) / cell_volume + kappa * ufl_w
    
    F = assemble(F)
    L = assemble(L)
    a = assemble(a)
    b = assemble(b)

    solve(F == L, ufl_c, bc_left + bc_move + bc_right)

# Time-stepping loop
for n in range(int(t_final / dt)):
    assemble_and_solve()
    
    # Save output concentration as function of time
    if n % 100 == 0:
        xdmf_file = File('output.xdmf')
        xdmf_file << Function(c)
        
    # Report total concentration after every 100 time steps
    if n % 100 == 0:
        print(f'Total concentration at t = {n * dt} s: {c.integrate(cell_volume)}')

# Save final output concentration as function of time
xdmf_file = File('output.xdmf')
xdmf_file << Function(c)
