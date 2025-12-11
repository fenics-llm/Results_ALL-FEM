from fenics import *

# Create mesh
mesh = UnitSquareMesh(96, 96)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
W = V * Q

# Define boundary conditions
u_lid = Constant((1, 0))
u_walls = Constant((0, 0))

def boundary(x, on_boundary):
    return on_boundary

bc_lid = DirichletBC(W.sub(0), u_lid, lambda x, on_boundary: on_boundary and near(x[1], 1))
bc_walls = DirichletBC(W.sub(0), u_walls, lambda x, on_boundary: on_boundary and (near(x[0], 0) or near(x[0], 1) or near(x[1], 0)))

bcs = [bc_lid, bc_walls]

# Define variational problem
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

rho = 1.0
mu = 1.0

a = mu * inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx + inner(q, div(u)) * dx

L = Constant(0) * v[0] * dx + Constant(0) * q * dx

# Compute solution
w = Function(W)
solve(a == L, w, bcs=bcs)

u_sol, p_sol = w.split()

# Save solution to file
vtkfile_u = File('q3_soln.xdmf')
vtkfile_p = File('q3_soln.xdmf')

u_sol.rename("velocity", "velocity")
p_sol.rename("pressure", "pressure")

vtkfile_u << u_sol
vtkfile_p << p_sol

# Plot speed
speed = sqrt(u_sol[0]**2 + u_sol[1]**2)
plot(speed, title='Speed')
interactive()

import matplotlib.pyplot as plt
import numpy as np

speed_values = speed.compute_vertex_values(mesh)
plt.imshow(np.reshape(speed_values, (97, 97)), cmap='viridis', extent=(0, 1, 0, 1))
plt.colorbar(label='Speed')
plt.savefig('q3_speed.png')
