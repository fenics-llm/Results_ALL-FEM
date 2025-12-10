import numpy as np
from fenics import *

# Define the mesh
L = 2.0
H = 0.20
nx = 120
ny = 12

# Create the top and bottom boundaries
x_top = np.linspace(0, L, nx)
y_top = np.zeros(ny)
top_boundary = Point(x_top, y_top)

x_bottom = np.linspace(0, L, nx)
y_bottom = np.linspace(0, H, ny)
bottom_boundary = Point(x_bottom, y_bottom)

# Create the mesh
mesh = RectangleMesh(bottom_boundary, top_boundary, L/120, H/12)

# Create a finite element function space
V = FunctionSpace(mesh, 'P', 1)
Q = FunctionSpace(mesh, 'P', 1)

# Define the boundary conditions
def wall(x):
    return on_boundary(x)

def inlet_outlet(x):
    return x[0] == 0 or x[1] == H

bc_walls = DirichletBC(V, Constant(0), wall)
bc_inout = DirichletBC(Q, Constant(0), inlet_outlet)

# Define the body force
f = Expression('1.0', degree=2)

# Define the velocity and pressure
u = Function(V)
p = Function(Q)

# Assemble the linear system
u_l = Expression('x[0]', degree=2) + f*x[1]
p_l = Constant(0)

a_u = inner(u, u) * dx
L_u = u_l * dx

a_p = inner(p, p) * dx
L_p = -inner(grad(u), grad(p)) * dx

F_u = L_u - a_u
F_p = L_p + a_p

# Solve the system
solve(F_u == 0, u, bc_walls)
solve(F_p == 0, p, bc_inout)

# Compute the velocity and pressure fields
u_n = u.compute_vertex_values(mesh.vertices())
p_n = p.compute_vertex_values(mesh.vertices())

# Save the results
np.save('q2_speed.npy', np.array(u_n))
np.save('q2_solution.xdmf', p_n)