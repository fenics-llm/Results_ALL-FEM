# filename: plane_stress.py
from fenics import *

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(1.0, 0.20), 40, 8)

# Define material parameters
E = 200e9  # Young's modulus
nu = 0.30  # Poisson's ratio

# Define boundary conditions
u_L = Constant((0, 0))
t_T = Constant((0, -2000))

# Define function space
V = VectorFunctionSpace(mesh, 'P', 1)

# Define stress and strain tensors
def sigma(u, E, nu):
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(2)

def epsilon(v):
    return sym(grad(v))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u, E, nu), epsilon(v)) * dx
L = dot(t_T, v) * ds(4)  # Top edge

# Define boundary conditions
def left_edge(x, on_boundary):
    return near(x[0], 0) and on_boundary

bc = DirichletBC(V, u_L, left_edge)

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save vertical displacement to file
uy = u.sub(1)
vtkfile = File('q2_uy.pvd')
vtkfile << uy

# Save color map of vertical displacement
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1.0, 100)
y = np.linspace(0, 0.20, 100)
X, Y = np.meshgrid(x, y)

points = [(x_, y_) for x_ in x for y_ in y]
uy_values = uy.compute_vertex_values(mesh)

Z = np.zeros_like(X)
for i, point in enumerate(points):
    x_, y_ = point
    idx = mesh.bounding_box_tree().compute_first_collision(Point(x_, y_))
    if idx < len(uy_values):
        Z[i // len(y), i % len(y)] = uy_values[idx]

plt.contourf(X, Y, Z, 50)
plt.colorbar(label='Vertical displacement (m)')
plt.savefig('q2_uy.png')

# Save resulting displacement field to XDMF format
xdmf_file = XDMFFile("displacement.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True
xdmf_file.write(u, 0)