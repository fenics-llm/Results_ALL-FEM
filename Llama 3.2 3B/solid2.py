# filename: legacy_fenics.py

import numpy as np
from fenics import *

# Define the mesh parameters
mesh_size_x = 1.0 / 40
mesh_size_y = 0.20 / 8
num_cells_x = int(1 / mesh_size_x)
num_cells_y = int(0.2 / mesh_size_y)

# Create the mesh
mesh = UnitSquareMesh(num_cells_x, num_cells_y)

# Define the boundary conditions
def left_boundary(x):
    return x[0] == 0

def top_boundary(x):
    return x[1] == 0.20

def right_boundary(x):
    return x[0] == 1.0

def bottom_boundary(x):
    return x[1] == 0

# Create the boundary conditions
bc_left = DirichletBC(mesh, Constant((0.0, 0.0)), left_boundary)
bc_top = DirichletBC(mesh, Constant((-2000.0, 0.0)), top_boundary)
bc_right = NeumannBC(mesh, Constant((0.0, 0.0)), right_boundary)
bc_bottom = NeumannBC(mesh, Constant((0.0, 0.0)), bottom_boundary)

# Define the parameters
E = 200e9  # Young's modulus in Pa
nu = 0.3   # Poisson's ratio

# Create the function space
V = FunctionSpace(mesh, 'P', 1)
Q = FunctionSpace(mesh, 'P', 1)

# Define the variational problem
u = TrialFunction(V)
v = TestFunction(V)
sigma = Expression('200e9 * epsilon(u)', epsilon=2*nu*Identity(2))
f = Constant((0.0, 0.0)) # Modified right-hand side

a = inner(sigma, v) * dx
L = inner(f, v) * dx

# Assemble the system
u = Function(V)
solve(a == L, u, bc_left + bc_top + bc_right + bc_bottom)

# Compute the vertical displacement
uy = u[1]

# Save the color map of the vertical displacement
from matplotlib import pyplot as plt
plt.imshow(uy, cmap='RdYlGn', extent=(0, 1.0, 0, 0.20), origin='lower')
plt.savefig('q2_uy.png')

# Save the resulting displacement field in XDMF format
file = File("displacement.xdmf")
file << (u,)