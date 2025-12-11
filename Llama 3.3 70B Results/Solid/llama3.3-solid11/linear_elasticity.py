# filename: linear_elasticity.py
from fenics import *
import numpy as np

# Define the geometry and mesh
L = 1.0
W = 0.20
a = 0.04
center = (0.50, 0.10)

mesh = RectangleMesh(Point(0, 0), Point(L, W), 100, 20)

# Define the material properties
E = 5e6  # Young's modulus (Pa)
nu = 0.49  # Poisson's ratio

# Define the boundary conditions and loading
u_L = Constant((0, 0))  # Left edge displacement
u_R = Constant((0.001, 0))  # Right edge displacement

def left_boundary(x):
    return near(x[0], 0)

def right_boundary(x):
    return near(x[0], L)

# Define the mixed displacement-pressure formulation
V = VectorFunctionSpace(mesh, 'P', 2)  # Displacement space
Q = FunctionSpace(mesh, 'P', 1)  # Pressure space

u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Define the stress and strain tensors
def sigma(u, p):
    return 2 * mu * epsilon(u) + lambda_ * div(u) * Identity(2)

def epsilon(u):
    return sym(grad(u))

mu = E / (2 * (1 + nu))
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))

# Define the bilinear form and linear form for the displacement equation
a_displacement = inner(sigma(u, p), epsilon(v)) * dx

# Define the bilinear form and linear form for the pressure equation
a_pressure = -p * q * dx

# Assemble the system of equations
u_u = Function(V)
p_p = Function(Q)

F = a_displacement + a_pressure

bc1 = DirichletBC(V, u_L, left_boundary)
bc2 = DirichletBC(V, u_R, right_boundary)
bcs = [bc1, bc2]

solve(F == 0, [u_u, p_p], bcs=bcs)

# Compute von Mises equivalent stress
sigma_vm = sqrt((sigma(u_u, p_p)[0, 0] - sigma(u_u, p_p)[1, 1])**2 + 3 * sigma(u_u, p_p)[0, 1]**2 + 3 * sigma(u_u, p_p)[1, 0]**2)

# Save color maps and displacement field
File('q11_vm.pvd') << project(sigma_vm)
File('q11_ux.pvd') << u_u[0]
XDMFFile('displacement_field.xdmf').write(u_u)