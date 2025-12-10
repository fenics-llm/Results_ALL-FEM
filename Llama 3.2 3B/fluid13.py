# filename: transport_chemical.py

import numpy as np
from fenics import *

# Define constants
L = 1.0  # length of the channel (m)
H = 0.10  # height of the channel (m)
U_max = 0.75  # maximum velocity (m/s)
D = 1.0e-5  # diffusivity of the chemical (m² s⁻¹)

# Create a mesh
mesh = UnitSquareMesh(100, 10)

# Define the function space for the solution and test functions
V = FunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define the boundary conditions
bc_inlet = DirichletBC(V, Constant(0.0), 'left')  # Use a valid subdomain identifier
bc_outlet = DirichletBC(Q, Constant(1.0), 'right')  # Use a valid subdomain identifier
bc_walls = NeumannBC(Q, Constant(0.0), 'bottom') + \
           NeumannBC(Q, Constant(0.0), 'top')

# Define the velocity field
u_x = Expression('4*y*(H-y)/H**2', y=interpolate(Constant(H/2), Q))
u = VectorField((u_x, 0))

# Define the advection-diffusion equation
def advection_diffusion(v):
    return inner(u*v, dx) + D*inner(grad(v), dx)

# Solve the problem
u = Function(V)
bc_inlet.apply(u)
bc_outlet.apply(u)
bc_walls.apply(u)
solve(advection_diffusion(u), u, 'EXACT', bc_outlet)

# Save the solution in xdmf format
file = File('transport_chemical.xdmf')
file << u