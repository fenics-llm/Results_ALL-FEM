# filename: plate_analysis.py

import numpy as np
from fenics import *

# Define the parameters
lambda_val = 19.44e9  # Pa
mu_val = 29.17e9     # Pa
sigma_Y = 243e6      # Pa

# Create a mesh
mesh = UnitSquareMesh(100, 180)

# Define the boundary conditions
bc_x0 = DirichletBC(mesh, (0, 0), 'x=0')
bc_y0 = DirichletBC(mesh, (0, 0), 'y=0')
bc_top = DirichletBC(mesh, (0, 1), 'y=180')
bc_edge = NeumannBC(mesh, (0, 0), 'x=100')

# Define the variational function space
V = FunctionSpace(mesh, 'P', 2)

# Define the trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Compute the strain tensor
epsilon_e = 0.5 * (grad(u) + grad(u.T))
epsilon_p = epsilon_e - (1/3) * tr(epsilon_e) * Identity(2)

# Define the elastic and plastic parts of the stress tensor
sigma_e = lambda_val * Identity(2) + mu_val * epsilon_e
sigma_p = 2*mu_val * epsilon_p

# Compute the equivalent stress
q = sqrt((3/2)*dot(sigma_p, sigma_p))

# Define the yield condition
F = q - sigma_Y

# Define the plastic loading and Kuhn-Tucker conditions
delta_epsilon_p = Function(V)
delta_epsilon_p.assign(0)

def compute_F(v):
    return inner(F*v, v)*dx + 2*mu_val*inner((epsilon_e-v)/2, v)*dx

# Assemble the variational problem
problem = NonlinearVariationalProblem(
    compute_F,
    u,
    bc_x0,
    bc_y0,
    bc_top,
    bc_edge,
    residual_type='exact'
)

# Solve the problem
problem.solve()

# Save the displacement in an xdmf file
u_xdmf = File('plate_displacement.xdmf')
u_xdmf << u
