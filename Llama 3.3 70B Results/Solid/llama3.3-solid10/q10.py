# filename: elasticity.py
from fenics import *

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(1.0, 0.20), 100, 20)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
W = V * Q

# Define trial and test functions
w = TrialFunction(W)
v = TestFunction(W)

u, p = split(w)
v_u, v_p = split(v)

# Material parameters
E = 5.0e6  # Young's modulus (Pa)
nu = 0.49  # Poisson's ratio

# Lame parameters
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

# Stress tensor
def sigma(u):
    return lmbda * div(u) * Identity(2) + 2 * mu * sym(grad(u))

# Variational formulation
a_uu = inner(sigma(u), grad(v_u)) * dx
a_up = - p * div(v_u) * dx
a_pu = - v_p * div(u) * dx
a_pp = inner(grad(p), grad(v_p)) * dx

F = a_uu + a_up + a_pu + a_pp

# Boundary conditions
def left_boundary(x, on_boundary):
    return near(x[0], 0)

def right_boundary(x, on_boundary):
    return near(x[0], 1.0)

bc_left_u = DirichletBC(W.sub(0), Constant((0, 0)), left_boundary)
bc_right_ux = DirichletBC(W.sub(0).sub(0), Constant(0.03), right_boundary)
bc_right_uy = DirichletBC(W.sub(0).sub(1), Constant(0), right_boundary)

bcs = [bc_left_u, bc_right_ux, bc_right_uy]

# Solve the system
w_sol = Function(W)
solve(F == 0, w_sol, bcs=bcs)

u_sol, p_sol = w_sol.split()

# Save results to file
vtkfile_u = File('displacement.pvd')
vtkfile_p = File('pressure.pvd')

vtkfile_u << u_sol
vtkfile_p << p_sol

# Save color map of displacement magnitude
import matplotlib.pyplot as plt
import numpy as np

u_array_x = u_sol.compute_vertex_values(mesh)
u_array_y = u_sol.compute_vertex_values(mesh, 1)

u_magnitude = np.sqrt(u_array_x**2 + u_array_y**2)

plt.figure(figsize=(10, 4))
plt.imshow(u_magnitude.reshape((20, 100)), cmap='viridis', origin='lower')
plt.colorbar(label='Displacement magnitude (m)')
plt.title('Displacement Magnitude')
plt.savefig('q10_disp.png')

# Save resulting displacement field in XDMF format
xdmf_file = XDMFFile("displacement.xdmf")
xdmf_file.write(u_sol, 0)