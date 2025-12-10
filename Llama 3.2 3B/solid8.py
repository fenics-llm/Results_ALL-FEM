
# filename: q8_fenics.py

import numpy as np
from fenics import *

# Define constants
Lx, Ly = 1.0, 0.2
num_x, num_y = 50, 25
theta = 30 * pi / 180

# Create mesh and finite element function space
mesh = RectangleMesh(Point(0, 0), Point(Lx, 0), Point(Lx, Ly), Point(0, Ly))
x, y = SpatialCoordinate(mesh)
u = Function('Displacement', (x, y))

# Define material properties
E1, E2, G12, nu12 = 40e9, 10e9, 5e9, 0.25

# Create stiffness matrix and apply boundary conditions
Q = [[E1, -nu12*E1], [-nu12*E1, E2]]
Q_rotated = rotate_matrix(Q, theta)
K = assemble(linear_elasticity_stiffness(Q_rotated))
bc_ux = DirichletBC(mesh, Constant(0), 'on_boundary')
bc_uy = DirichletBC(mesh, Constant(0), 'on_boundary')
bc_sigma_y = DirichletBC(mesh, Expression(-(10e6 * np.sin(theta)), degree=2), 'y=0.20')
bc_sigma_x = NeumannBC(mesh, Constant(0), 'x=0', 'x=1')

# Assemble and solve the linear system
u_sol = Function('Displacement', (x, y))
K.bind(u_sol)
bc_ux.bind(u_sol)
bc_uy.bind(u_sol)
bc_sigma_y.bind(u_sol)
bc_sigma_x.bind(u_sol)

solve(K, u_sol, bc_ux, bc_uy, bc_sigma_y, bc_sigma_x)

# Calculate von Mises stress
sigma = linear_elasticity_stress(Q_rotated, u_sol)
vm_stress = calculate_von_mises_stress(sigma)

# Save fields to XDMF file
xdmf_file = File("q8_solution.xdmf")
xdmf_file << (u_sol, vm_stress)

# Save displacement and stress fields as images
import matplotlib.pyplot as plt

plt.imshow(u_sol.x, cmap='RdYlGn', origin='lower')
plt.savefig('q8_ux.png')

plt.imshow(vm_stress.x, cmap='RdYlGn', origin='lower')
plt.savefig('q8_vm.png')

# Print final result
print("Displacement field saved as q8_ux.png")
print("Von Mises stress field saved as q8_vm.png")