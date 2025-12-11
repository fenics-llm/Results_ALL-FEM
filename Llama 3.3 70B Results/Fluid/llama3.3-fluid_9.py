# filename: navier_stokes.py
from fenics import *

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(2.0, 0.20), 200, 20)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
C = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions for flow
u_inlet = Expression(('6 * U_bar * (x[1]/H) * (1 - x[1]/H)', '0'), degree=2, U_bar=0.1, H=0.20)
u_walls = Constant((0, 0))
p_outlet = Constant(0)

# Define boundary conditions for concentration
c_inlet = Constant(0)
c_outlet = Constant(1)

# Define parameters
rho = 1
mu = 0.01
kappa = 1e-3

# Define variational problem for flow
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

a_flow = rho * inner(dot(u, grad(u)), v) * dx + mu * inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx
L_flow = Constant(0) * v[0] * dx

# Define variational problem for concentration
c = TrialFunction(C)
d = TestFunction(C)

a_conc = kappa * inner(grad(c), grad(d)) * dx + inner(dot(u, grad(c)), d) * dx
L_conc = Constant(0) * d * dx

# Assemble and solve flow problem
u_bc = DirichletBC(V, u_inlet, 'on_boundary && near(x[0], 0)')
v_bc = DirichletBC(V, u_walls, 'on_boundary && near(x[1], 0) || on_boundary && near(x[1], 0.20)')
p_bc = DirichletBC(Q, p_outlet, 'on_boundary && near(x[0], 2.0)')

u_sol = Function(V)
p_sol = Function(Q)

F_flow = a_flow - L_flow
solve(F_flow == 0, u_sol, bcs=[u_bc, v_bc])
solve(inner(grad(p_sol), grad(q)) * dx == inner(div(u_sol), q) * dx, p_sol, bcs=[p_bc])

# Assemble and solve concentration problem
c_bc_inlet = DirichletBC(C, c_inlet, 'on_boundary && near(x[0], 0)')
c_bc_outlet = DirichletBC(C, c_outlet, 'on_boundary && near(x[0], 2.0)')

c_sol = Function(C)

F_conc = a_conc - L_conc
solve(F_conc == 0, c_sol, bcs=[c_bc_inlet, c_bc_outlet])

# Save solutions to file
vtkfile_u = File('q10_solution.xdmf')
vtkfile_p = File('q10_solution.xdmf')
vtkfile_c = File('q10_solution.xdmf')

vtkfile_u << u_sol
vtkfile_p << p_sol
vtkfile_c << c_sol

# Save concentration field as color map
import matplotlib.pyplot as plt
import numpy as np

c_values = c_sol.compute_vertex_values(mesh)
plt.figure(figsize=(8, 2))
plt.imshow(c_values.reshape(20, 200), cmap='viridis', origin='lower')
plt.colorbar()
plt.savefig('q10_conc.png')

TERMINATE