from fenics import *

# Parameters
rho = 1.0         # kg/m^3
mu = 1.5e-5       # Pa·s
alpha = 2.1e-5    # m² s⁻¹
g_beta = 3.15e-5  # m s-2 K-1
T_ref = 0.5       # K

# Create mesh and define function spaces
mesh = UnitSquareMesh(100, 100)
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
W = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
u0 = Constant((0, 0))
bc_u_left = DirichletBC(V, u0, 'on_boundary && near(x[0], 0)')
bc_u_right = DirichletBC(V, u0, 'on_boundary && near(x[0], 1)')
bc_u_top_bottom = DirichletBC(V, u0, 'on_boundary && (near(x[1], 0) || near(x[1], 1))')

T_left = Constant(1.0)
T_right = Constant(0.0)
bc_T_left = DirichletBC(W, T_left, 'on_boundary && near(x[0], 0)')
bc_T_right = DirichletBC(W, T_right, 'on_boundary && near(x[0], 1)')

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)
T = TrialFunction(W)
w = TestFunction(W)

a_u = rho * inner(grad(u), grad(v)) * dx + mu * inner(div(u), div(v)) * dx
L_u = -g_beta * (T_ref) * v[1] * dx

a_p = inner(p, q) * dx
L_p = inner(div(u), q) * dx

a_T = alpha * inner(grad(T), grad(w)) * dx + inner(dot(u, grad(T)), w) * dx
L_T = Constant(0) * w * dx

# Solve variational problem for temperature first
T_sol = Function(W)
solve(a_T == L_T, T_sol, [bc_T_left, bc_T_right])

# Now solve for velocity and pressure using the solved temperature field
a_u = rho * inner(grad(u), grad(v)) * dx + mu * inner(div(u), div(v)) * dx
L_u = -g_beta * (T_sol - T_ref) * v[1] * dx

u_sol = Function(V)
p_sol = Function(Q)

F = rho * inner(grad(u), grad(v)) * dx + mu * inner(div(u), div(v)) * dx - g_beta * (T_sol - T_ref) * v[1] * dx + p * div(v) * dx + q * div(u) * dx
solve(F == 0, [u_sol, p_sol], [bc_u_left, bc_u_right, bc_u_top_bottom])

# Save solution to file
vtkfile_u = File('q11_solution_u.pvd')
vtkfile_p = File('q11_solution_p.pvd')
vtkfile_T = File('q11_solution_T.pvd')

vtkfile_u << u_sol
vtkfile_p << p_sol
vtkfile_T << T_sol

# Save temperature field as color map
import matplotlib.pyplot as plt
import numpy as np

T_values = T_sol.compute_vertex_values(mesh)
x = mesh.coordinates()[:, 0]
y = mesh.coordinates()[:, 1]

plt.figure(figsize=(8, 8))
plt.scatter(x, y, c=T_values, cmap='viridis')
plt.colorbar(label='Temperature (K)')
plt.savefig('q11_T.png')

# Calculate average Nusselt number at left wall
n = FacetNormal(mesh)
h = CellSize(mesh)

Nu_avg = assemble((T_sol / (1 - T_ref)) * n[0] * ds(1)) / assemble(ds(1))
print(f'Average Nusselt number at left wall: {Nu_avg}')

# Save solution to xdmf file
xdmffile = XDMFFile('q11_solution.xdmf')
xdmffile.write(u_sol, 0)
xdmffile.write(p_sol, 0)
xdmffile.write(T_sol, 0)
