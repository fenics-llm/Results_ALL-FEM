# filename: mesh.py

from fenics import UnitSquareMesh
import numpy as np

mesh = UnitSquareMesh(128, 128)

# Define the boundary conditions
def left_boundary(x):
    return 0

def right_boundary(x):
    return 1

def bottom_boundary(x):
    return 0

def top_boundary(x):
    return 1

bc_left = DirichletBC(mesh, Constant(0), Interval(0, 0))
bc_right = DirichletBC(mesh, Constant(1), Interval(1, 1))
bc_bottom = DirichletBC(mesh, Constant(0), Interval(0, 0))
bc_top = DirichletBC(mesh, Constant((1, 0)), Interval(0, 1))

# Define the parameters
rho = 1.0  # density
mu = 0.01  # dynamic viscosity

# Create a function space for the velocity and pressure fields
V = FunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 2)

# Define the Navier-Stokes equations
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
f = Constant((0, 0))  # no source term

# Define the boundary conditions
bc_left(u) == 0
bc_right(u) == 1
bc_bottom(u) == 0
bc_top(u) == (1, 0)

bc_left(p) == 0
bc_right(p) == 0
bc_bottom(p) == 0

# Define the variational form of the Navier-Stokes equations
F_u = rho * inner(grad(u), grad(v)) + mu * div(u) * v
F_p = -div(u) * p + f * p

# Assemble the matrices and vectors
M_u, b_u = assemble_and_assemble(F_u)
M_p, b_p = assemble_and_assemble(F_p)

# Solve for u and p
u_solution = Function(V)
p_solution = Function(Q)

solve(b_u == M_u, u_solution, bc_left)
solve(b_p == M_p, p_solution, bc_left)

# Save the velocity field and pressure field to files
vtk_file = File('q5_soln.xdmf')
vtk_file << (u_solution, p_solution)

# Create a color map of speed |u| over Ω
speed_field = Function(V)
speed_field.assign(u_solution)

vtk_file = File('q5_speed.png')
vtk_file << speed_field
