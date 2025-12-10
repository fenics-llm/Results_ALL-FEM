from fenics import *

# Define the problem domain
Omega = Rectangle(0, 2.0, 0, 0.20)

# Create a mesh with 200x20 elements
mesh = UnitSquareMesh(200, 20)
cell_volume = mesh.cell_volume()

# Define the boundary conditions
bc_inlet_u_x = DirichletBC(mesh, lambda x: 6 * 0.1 * (x[1] / 0.2) * (1 - x[1] / 0.2), 'inner')
bc_outlet_traction_free = NeumannBC(mesh, lambda x: np.zeros(3), 'outer')
bc_walls_no_slip = DirichletBC(mesh, lambda x: np.zeros(3), 'left', 'bottom')
bc_inlet_concentration = DirichletBC(mesh, 0.0, 'inner')
bc_outlet_concentration = DirichletBC(mesh, 1.0, 'outer')

# Define the parameters
rho = 1.0
mu = 0.01
epsilon = 1e-6
D = 1e-3

# Create a function space for the velocity and pressure
V = FunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 2)

# Define the variational problem for the fluid flow
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
w = TestFunction(Q)

# Define the weak form of the Navier-Stokes equations
F_u = rho * inner(u, v) * dx - mu * inner(grad(u), grad(v)) * dx + p * div(v) * dx
F_p = -rho * u * v * dx

# Assemble the bilinear forms and linear forms
F = F_u + F_p

# Define the boundary conditions for the weak form
bc_u = bc_inlet_u_x + bc_outlet_traction_free + bc_walls_no_slip
bc_p = DirichletBC(Q, 0.0, 'inner')

# Solve the system of equations
solve(F == 0, (u, p), solver_parameters={'linear_solver': 'gmres', 'max_iter': 100})

# Define the variational problem for the advection-diffusion equation
c = TrialFunction(V)
w = TestFunction(Q)

# Define the weak form of the advection-diffusion equation
F_c = D * inner(grad(c), grad(w)) * dx + u * w * dx

# Assemble the bilinear forms and linear forms
F_c = F_c

# Solve the system of equations
solve(F_c == 0, c, solver_parameters={'linear_solver': 'gmres', 'max_iter': 100})

# Save the solution to files
vtk_file = File('q10_conc.pvd')
vtk_file << (u, p, c)

# Save the solution as a color map
import matplotlib.pyplot as plt
plt.imshow(c[0].get_array(), cmap='viridis', origin='lower')
plt.colorbar()
plt.savefig('q10_conc.png')

# Save the solution to xdmf file
xdmf_file = File('q10_solution.xdmf')
xdmf_file << (u, p, c)
