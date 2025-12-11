# filename: lid_driven_cavity.py
from fenics import *

# Create mesh
mesh = UnitSquareMesh(128, 128)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define parameters
rho = 1.0
mu = 0.01

# Define boundary conditions
def lid(x, on_boundary):
    return on_boundary and near(x[1], 1)

def walls(x, on_boundary):
    return on_boundary and (near(x[0], 0) or near(x[0], 1) or near(x[1], 0))

bc_lid = DirichletBC(V, Constant((1, 0)), lid)
bc_walls = DirichletBC(V, Constant((0, 0)), walls)

bcs_u = [bc_lid, bc_walls]
bc_p = DirichletBC(Q, Constant(0), 'on_boundary')

# Define variational problem
F = rho * inner(dot(u, nabla_grad(u)), v) * dx + \
    mu * inner(grad(u), grad(v)) * dx - \
    inner(p, div(v)) * dx + \
    inner(div(u), q) * dx

a = lhs(F)
L = rhs(F)

# Compute solution
u_sol = Function(V)
p_sol = Function(Q)

solve(a == L, [u_sol, p_sol], bcs=[bc_lid, bc_walls, bc_p])

# Save solution to file
vtkfile_u = File('q5_soln.xdmf')
vtkfile_p = File('q5_soln.xdmf')

u_sol.rename("u", "velocity")
p_sol.rename("p", "pressure")

vtkfile_u << u_sol
vtkfile_p << p_sol

# Save speed to file
speed = sqrt(u_sol[0]**2 + u_sol[1]**2)
speed_f = project(speed, Q)

import matplotlib.pyplot as plt
import numpy as np

speed_values = speed_f.compute_vertex_values(mesh)
plt.figure()
plt.imshow(np.reshape(speed_values, (128, 128)), cmap='viridis', origin='lower')
plt.colorbar()
plt.savefig('q5_speed.png')
