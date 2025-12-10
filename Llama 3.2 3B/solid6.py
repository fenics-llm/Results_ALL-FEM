# filename: legacy_fenics.py

import numpy as np
from fenics import *

# Define the problem domain
mesh = UnitSquareMesh(100, 100)

# Create a function space for the displacement field
V = FunctionSpace(mesh, 'P', 1)

# Define the boundary conditions
bc_bottom = CompiledSubDomain("0 <= x[0] && 0 <= x[1] && x[1] <= 0.2", mesh)
bc_top = CompiledSubDomain("(x[0], x[1]) >= (0.5, 0.2) && (x[0], x[1]) <= (0.55, 0.2)", mesh)

# Create a function space for the stress field
W = FunctionSpace(mesh, 'P', 1)

# Define the boundary conditions for the stress field
sigma_n = Constant((0.0, -10.0))

bc_stress_top = DirichletBC(W, sigma_n, bc_top)
bc_stress_notch_arc = DirichletBC(W, sigma_n, bc_bottom)

# Create a linear system of equations
u = Function(V)
v = TestFunction(V)

f = Constant((0.0, 0.0))

a = inner(grad(u), grad(v)) * dx
L = dot(f, v) * dx

bc_stress_top.apply(a)
bc_stress_notch_arc.apply(a)

# Solve the system of equations
u.solve(L-u, u)

# Compute the von Mises equivalent stress
sigma_x = u.x[0]
sigma_y = u.x[1]

stress_tensor = Matrix2x2(sigma_x, sigma_y)
von_mises_stress = np.sqrt(stress_tensor[0, 0]**2 + stress_tensor[1, 1]**2 + 
                            2*stress_tensor[0, 0]*stress_tensor[1, 1])

# Create a color map
import matplotlib.pyplot as plt

plt.imshow(von_mises_stress, cmap='hot', extent=[0, 1, 0, 0.2])
plt.colorbar()
plt.savefig('q6_vm.png')

# Save the resulting displacement field in XDMF format
u_xdmf = File('displacement.xdmf')
u_xdmf << u

# Compute the von Mises equivalent stress and save it to a file
np.savetxt('von_mises_stress.txt', von_mises_stress)
