# filename: geometry.py

import numpy as np
from fenics import Circle, Point, solve, VectorFunctionFunctionSpace, integrate


def boundary(x):
    if x[0] == 0:
        return 'inlet'
    elif x[1] == 0 or x[1] == 0.41:
        return 'wall'
    elif np.sqrt((x[0]-0.20)**2 + (x[1]-0.20)**2) <= 0.05:
        return 'circular'
    else:
        return 'outlet'

def boundary_function(x):
    if x[0] == 0:
        return 6 * 0.2 * x[1] * (0.41 - x[1]) / (0.41 ** 2)
    elif x[1] == 0 or x[1] == 0.41:
        return 0
    elif np.sqrt((x[0]-0.20)**2 + (x[1]-0.20)**2) <= 0.05:
        return 0
    else:
        return 0

# Define the geometry of the channel with a circular hole
Omega = ((0, 2.2), (0, 0.41))
hole = Circle(Point(0.20, 0.20), 0.05)

# Define the velocity and pressure fields
V = VectorFunctionSpace(Omega[0] * Omega[1], 'P', 1)
Q = FunctionSpace(Omega[0] * Omega[1], 'P', 1)

u = Function(V)
p = Function(Q)

# Define the problem
problem = (div(u) == 0, u.dual() == 0, p == 0, 0 == 0)

# Set up the boundary conditions
bc_inlet = DirichletBC(V.subset((0, 0)), boundary_function, 'inlet')
bc_wall = DirichletBC(V.subset(((0, 0.41) + (2.2, 0))), 0, 'wall')
bc_circular = DirichletBC(hole, 0, 'circular')
bc_outlet = NeumannBC(V.subset((2.2, 0)), 0, 'outlet')

# Solve the problem
u_sol, p_sol = solve(problem, bc_inlet + bc_wall + bc_circular + bc_outlet)

# Define the velocity field at the outlet
u_outlet = u_sol.subset((2.2, 0))

# Compute the drag force on the circle
F_D = 1e-3 * integrate(u_outlet[1] ** 2, hole)

# Compute the drag coefficient C_D
C_D = 2 * F_D / (1 * 0.2 ** 2 * 4 * 3.14159)

# Save the velocity field and pressure field
File("q7_soln.xdmf") << u_sol, p_sol

# Save the color map of speed
import matplotlib.pyplot as plt
plt.imshow(u_sol[1], cmap='RdYlGn')
plt.savefig("q7_speed.png")
