# Import necessary modules
from fenics import UnitSquareMesh, DirichletBC, NeumannBC, solve_stokes, solve
import numpy as np

# Define the dimensions of the rectangular channel
L = 2.0  # length in meters
H = 0.20  # height in meters

# Create a mesh for the x dimension
x_mesh = UnitSquareMesh(L, H)

# Create a mesh for the y dimension
y_mesh = UnitSquareMesh(H, L)

# Define the parameters
mu = 1.0  # dynamic viscosity in Pa·s
rho = 1.0  # density in kg·m⁻³

# Set up the Stokes equations
u, p = solve_stokes(x_mesh, y_mesh, mu, rho)

# Define the boundary conditions
def wall_bc(x):
    return (x[0], 0) if x[0] < L/2 else (L-x[0], 0)

wall_bc = DirichletBC(x_mesh, wall_bc, "on_boundary")

inlet_bc = NeumannBC(y_mesh, -1.0, "on_boundary")
outlet_bc = NeumannBC(y_mesh, -0.0, "on_boundary")

# Solve the system of equations
solve(u, p, u_bc=wall_bc, p_bc=inlet_bc, p_out_bc=outlet_bc)