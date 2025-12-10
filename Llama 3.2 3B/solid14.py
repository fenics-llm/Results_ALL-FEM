# filename: q14.py
import fenics as fe
import numpy as np

# Define the geometry
Lx = 1.0  # length of the rectangle
Ly = 0.20  # height of the rectangle
a = 0.04  # radius of the circular holes

# Create a mesh for the rectangular strip with two circular holes
mesh = fe.UnitSquareMesh(Ly, Lx)

# Define the material properties
E = 5e6  # Young's modulus (MPa)
nu = 0.49  # Poisson's ratio

# Create a function space for the displacement field
V = fe.FunctionSpace(mesh, 'P', 1)

# Define the boundary conditions
u_D = fe.Constant((0.060, 0))  # prescribed displacement on the right edge
u_bc_left = fe.DirichletBC(V, u_D, 'left')
u_bc_right = fe.DirichletBC(V, u_D, 'right')

# Create a function space for the pressure field
Q = fe.FunctionSpace(mesh, 'P', 1)

# Define the boundary conditions for the pressure field
def boundary_value(x):
    return -0.10e6

p_bc_top_bottom = fe.DirichletBC(Q, boundary_value, 'top')
p_bc_left = fe.DirichletBC(Q, boundary_value, 'left')
p_bc_right = fe.DirichletBC(Q, boundary_value, 'right')

# Define the weak form of the equations
u, p = fe.TrialFunctions(V, Q)
w = fe.TestFunction(V)

# Define the bilinear form for the displacement equation
F_u = fe.inner(fe.grad(u), fe.grad(w)) * fe.dx

# Define the linear form for the pressure equation
F_p = -fe.inner(fe.div(p, 'dx'), w) * fe.dx

# Assemble the weak form of the equations
F = F_u + F_p

# Define the boundary conditions for the displacement equation
bc_u = u_bc_left + u_bc_right

# Assemble the linear system
A, b = fe.assemble(F), fe.assemble(boundary_value * fe.dx)

# Solve the system
u_sol = fe.solve(A, u, bc=bc_u)
p_sol = fe.solve(A, p, bc=p_bc_top_bottom + p_bc_left + p_bc_right)

# Plot the pressure field
fe.plot(p_sol, title='Pressure Field')

# Save the pressure field as a PNG file
fe.save('q14_p.png', p_sol)

# Define the von Mises stress tensor
sigma = fe.ElasticityTensor(E, nu)
stress = sigma * fe.div(fe.grad(u_sol), 'dx')

# Plot the von Mises stress field
fe.plot(stress, title='Von Mises Stress Field')

# Save the von Mises stress field as a PNG file
fe.save('q14_vm.png', stress)

# Save the displacement field in XDMF format
fe.save('displacement.xdmf', u_sol)
