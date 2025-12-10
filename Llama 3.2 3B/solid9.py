# filename: fem.py

import numpy as np
from fenics import *

# Define the mesh
mesh = RectangleMesh(Point(0, 0), Point(1, 0), Point(1, 0.2), Point(0, 0.2), diagonal='right', nx=100, ny=20)

# Define the boundary conditions
def left_bc(x):
    return DirichletBC(V, Constant(0), x[0])

def right_bc(x):
    return DirichletBC(V, Constant((2000000, 0)), x[1])

def top_bottom_bc(x):
    return DirichletBC(V, Constant((0, 0)), x[2])

bc_left = left_bc()
bc_right = right_bc()
bc_top_bottom = top_bottom_bc()

# Define the material properties
nu = 0.3

def E(y):
    return 100e9 + 100e9 * (y / 0.2)

V = FunctionSpace(mesh, 'P', 1)
u = TrialFunction(V)
v = TestFunction(V)

sigma_x = Symmetric(2*Epsilon(u))
sigma_y = Symmetric(2*Epsilon(u))

a = inner(sigma_x, sigma_x) + inner(sigma_y, sigma_y)
L = inner((0, 0), v)

# Assemble the stiffness matrix and load vector
M = assemble(a)
b = assemble(L)

# Define the boundary conditions
bc_left.apply(M)
bc_right.apply(M)
bc_top_bottom.apply(M)

# Solve the system
u_expr = Function(V)
solve(M, u_expr, b)

# Compute the displacement field
displacement_x = u_expr[0]
displacement_y = u_expr[1]

# Save the color map of displacement magnitude |u|
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list("", ["blue", "red"])
plt.imshow(displacement_x, cmap=cmap, extent=[0, 1, 0, 0.2], origin='lower')
plt.colorbar(label="Displacement X")
plt.savefig("q9_disp.png")

# Save the resulting displacement field in XDMF format
import xdmf

output_file = "displacement.xdmf"
with xdmf.open(output_file, "w") as output:
    root_group = output.create_root_group()
    mesh_group = root_group.create_mesh_group(name="mesh")
    mesh_group.set_mesh(mesh)
    displacement_group = root_group.create_function_group(name="displacement")
    displacement_group.append(displacement_x)
    displacement_group.append(displacement_y)

# Print the final answer
print("Displacement X:", np.max(displacement_x))
print("Displacement Y:", np.max(displacement_y))