from fenics import *
import numpy as np

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(1.0, 0.20), 50, 25)

# Define material properties
E1 = 40e9  # Young's modulus along the principal 'grain' direction (Pa)
E2 = 10e9  # Young's modulus across the 'grain' (Pa)
G12 = 5e9  # Shear modulus in the 1-2 plane (Pa)
nu12 = 0.25  # Poisson's ratio, for strain in direction 2 from a load in direction 1

# Define rotation angle
theta = np.deg2rad(30)

# Calculate stiffness matrix in local coordinates
Q11 = E1 / (1 - nu12**2)
Q22 = E2 / (1 - nu12**2)
Q12 = nu12 * E2 / (1 - nu12**2)
Q66 = G12

# Assemble the stiffness matrix
Q_local = np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])

# Rotate stiffness matrix to global coordinates
R = np.array([[np.cos(theta)**2, np.sin(theta)**2, 2*np.sin(theta)*np.cos(theta)],
              [np.sin(theta)**2, np.cos(theta)**2, -2*np.sin(theta)*np.cos(theta)],
              [-np.sin(theta)*np.cos(theta), np.sin(theta)*np.cos(theta), np.cos(theta)**2 - np.sin(theta)**2]])
Q_global = np.dot(R.T, np.dot(Q_local, R))

# Define variational problem
V = VectorFunctionSpace(mesh, 'P', 1)
u = TrialFunction(V)
v = TestFunction(V)

# Define boundary conditions
bc_bottom = DirichletBC(V, Constant((0, 0)), 'on_boundary && near(x[1], 0)')

# Define bilinear form
a = inner(grad(u), grad(v))*dx

# Define linear form
L = -10e6*v[1]*ds(4)

# Solve variational problem
u_sol = Function(V)
solve(a == L, u_sol, [bc_bottom])

# Save solution to file
vtkfile_u = File('q8_solution.pvd')
vtkfile_u << u_sol

# Compute von Mises stress
sigma_xx = Q_global[0, 0]*u_sol[0].dx(0) + Q_global[0, 1]*u_sol[1].dx(0)
sigma_yy = Q_global[1, 0]*u_sol[0].dx(0) + Q_global[1, 1]*u_sol[1].dx(0)
sigma_xy = Q_global[2, 2]*(u_sol[0].dx(1) + u_sol[1].dx(0))
von_mises = sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3*sigma_xy**2)

# Save von Mises stress to file
vtkfile_vm = File('q8_vm.pvd')
vtkfile_vm << project(von_mises, FunctionSpace(mesh, 'P', 1))

# Save horizontal displacement to file
ux = u_sol[0]
vtkfile_ux = File('q8_ux.pvd')
vtkfile_ux << project(ux, FunctionSpace(mesh, 'P', 1))

# Write fields to XDMF file
xdmffile = XDMFFile('q8_solution.xdmf')
xdmffile.write(u_sol, 0)
xdmffile.close()

# Save color maps as images
import matplotlib.pyplot as plt

ux_plot = plot(project(ux, FunctionSpace(mesh, 'P', 1)), title='Horizontal Displacement')
plt.savefig('q8_ux.png')

vm_plot = plot(project(von_mises, FunctionSpace(mesh, 'P', 1)), title='Von Mises Stress')
plt.savefig('q8_vm.png')