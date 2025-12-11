from fenics import *

# Define parameters
rho = 1.0  # density
U_bar = 1.0  # mean inlet speed
mu_ref = 0.02  # reference viscosity
beta = 0.05  # temperature-dependent viscosity parameter
T_ref = 300  # reference temperature
kappa = 1e-3  # thermal diffusivity

# Define geometry and mesh
L = 2.0  # length
H = 0.20  # height
mesh = RectangleMesh(Point(0, 0), Point(L, H), 100, 20)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)  # velocity space
Q = FunctionSpace(mesh, 'P', 1)  # pressure space
T_space = FunctionSpace(mesh, 'P', 2)  # temperature space

# Define boundary conditions
u_inlet = Expression('6 * U_bar * x[1] * (H - x[1]) / pow(H, 2)', degree=2, U_bar=U_bar, H=H)
u_bc = DirichletBC(V, u_inlet, 'on_boundary && near(x[0], 0)')
no_slip_bc = DirichletBC(V, Constant((0, 0)), 'on_boundary && (near(x[1], 0) || near(x[1], H))')
T_ref_bc = DirichletBC(T_space, T_ref + 10, 'on_boundary && near(x[1], 0)')
T_neumann_bc = NeumannBC(T_space, Constant(0), 'on_boundary && near(x[1], H)')

# Define variational problem
u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)
T, S = TrialFunction(T_space), TestFunction(T_space)

mu = mu_ref * exp(-beta * (T - T_ref))
F_u = rho * inner(dot(u, grad(u)), v) * dx + inner(mu * sym(grad(u)), grad(v)) * dx - inner(p, div(v)) * dx
F_p = inner(div(u), q) * dx
F_T = inner(dot(u, grad(T)), S) * dx + kappa * inner(grad(T), grad(S)) * dx

# Solve variational problem
u_sol = Function(V)
p_sol = Function(Q)
T_sol = Function(T_space)

solve(F_u == 0, u_sol, [u_bc, no_slip_bc])
solve(F_p == 0, p_sol)
solve(F_T == 0, T_sol, [T_ref_bc, T_neumann_bc])

# Compute viscosity
mu_sol = mu_ref * exp(-beta * (T_sol - T_ref))

# Save viscosity as color map image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = mesh.coordinates()[:, 0]
y = mesh.coordinates()[:, 1]
mu_values = mu_sol.compute_vertex_values(mesh)

plt.figure(figsize=(8, 6))
plt.imshow(mu_values.reshape((20, 100)), extent=(0, L, 0, H), cmap='viridis', origin='lower')
plt.colorbar(label=r'$\mu$ (Pa$\cdot$s)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title(r'Viscosity $\mu(x, y)$')
plt.savefig('q13_mu.png')

# Extract streamwise velocity profile
ux_values = u_sol.compute_vertex_values(mesh)
y_values = mesh.coordinates()[:, 1]

import pandas as pd

df = pd.DataFrame({'y': y_values, 'ux': ux_values})
df.to_csv('q13_profile.csv', index=False)

# Export solution fields in XDMF format
u_file = XDMFFile('q13_solution_u.xdmf')
p_file = XDMFFile('q13_solution_p.xdmf')
T_file = XDMFFile('q13_solution_T.xdmf')
mu_file = XDMFFile('q13_solution_mu.xdmf')

u_file.write(u_sol)
p_file.write(p_sol)
T_file.write(T_sol)
mu_file.write(mu_sol)

u_file.close()
p_file.close()
T_file.close()
mu_file.close()

# filename: q13.py