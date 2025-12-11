# filename: elasticity.py
from fenics import *

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(1.0, 0.20), 100, 20)

# Define material parameters
nu = 0.30

# Define variational problem
V = VectorFunctionSpace(mesh, 'P', 2)
u = TrialFunction(V)
v = TestFunction(V)

# Define Young's modulus as a function of y
x, y = SpatialCoordinate(mesh)

# Define stress tensor
def sigma(u):
    E = 100e9 + 100e9 * (y / 0.20)
    return E * (nu / (1 - nu**2)) * (div(u) * Identity(2)) + E * (1 / (1 + nu)) * sym(grad(u))

# Define variational form
a = inner(sigma(TrialFunction(V)), grad(TestFunction(V))) * dx

# Define boundary conditions
def left_boundary(x, on_boundary):
    return near(x[0], 0)

def right_boundary(x, on_boundary):
    return near(x[0], 1.0)

bc_left = DirichletBC(V, Constant((0, 0)), left_boundary)
bc_right = []

# Define traction force
T = Expression(('2e6', '0'), degree=1)

# Add boundary conditions and traction force to variational form
L = inner(T, v) * ds(4)

# Solve the system
u_sol = Function(V)
solve(a == L, u_sol, bcs=[bc_left])

# Save displacement field in XDMF format
file = XDMFFile('displacement.xdmx')
file.write(u_sol)

# Save color map of displacement magnitude as PNG image
import matplotlib.pyplot as plt
import numpy as np

u_values = u_sol.compute_vertex_values(mesh)
u_magnitude = np.sqrt(u_values[::2]**2 + u_values[1::2]**2)
plt.imshow(np.reshape(u_magnitude, (21, 101)), cmap='viridis', origin='lower')
plt.colorbar(label='Displacement magnitude (m)')
plt.savefig('q9_disp.png')