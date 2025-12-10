# filename: q13.py

import numpy as np
from fenics import *

# Define the geometry
class Rectangle:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

L = 1.0
H = 1.0
Omega = Rectangle(0, 0, L, H)

# Define the parameters
rho = 1.0  # density
Ubar = 1.0  # mean inlet speed
mu_ref = 0.02  # reference viscosity
beta = 0.05  # temperature-dependent viscosity coefficient
T_ref = 300  # reference temperature
kappa = 1.0e-3  # thermal diffusivity

# Define the boundary conditions
def u_x(y):
    return 6 * Ubar * y * (H - y) / H**2

def u_y(x, y):
    return 0

def T_bottom_wall(y):
    return T_ref + 10

def T_top_wall(y):
    return T_ref

# Define the boundary conditions
bc_u_x = DirichletBC(Omega, u_x, 'x=0')
bc_u_y = DirichletBC(Omega, u_y, 'y=0') and DirichletBC(Omega, u_y, 'y=H')
bc_T_bottom_wall = DirichletBC(Omega, T_bottom_wall, 'y=0')
bc_T_top_wall = DirichletBC(Omega, T_top_wall, 'y=H')

### Step 3: Create a mesh for the domain
# Create a mesh for the domain
mesh = RectangleMesh(Omega, 100, 100)


### Step 4: Assemble the matrices and vectors representing the Navier-Stokes equations and the advection-diffusion equation
# Define the function spaces
V_u = FunctionSpace(mesh, 'P', 2)
V_p = FunctionSpace(mesh, 'P', 2)
V_T = FunctionSpace(mesh, 'P', 2)

# Define the test functions
u, p, T = TrialFunctions(V_u, V_p, V_T)

# Define the variational forms of the Navier-Stokes equations and the advection-diffusion equation
F_u = rho * inner(u * grad(u), dx) + inner(grad(p), dx) - 2 * mu_ref * exp(-beta * (T - T_ref)) * inner(grad(u), grad(u)) / 2
F_p = -inner(grad(u), dx)
F_T = u * grad(T), dx) - kappa * inner(grad(T), dx)

# Assemble the matrices and vectors representing the Navier-Stokes equations and the advection-diffusion equation
M_u, b_u = assemble(F_u, 'u')
M_p, b_p = assemble(F_p, 'p')
M_T, b_T = assemble(F_T[0], 'T')

# Define the boundary conditions for the matrices and vectors representing the Navier-Stokes equations and the advection-diffusion equation
bc_M_u = DirichletBC(V_u, 0, bc_u_x)
bc_M_p = DirichletBC(V_p, 0, bc_u_y)
bc_M_T = DirichletBC(V_T, T_bottom_wall, bc_T_bottom_wall) and DirichletBC(V_T, 0, bc_T_top_wall)

# Assemble the matrices and vectors representing the Navier-Stokes equations and the advection-diffusion equation
M_u_bc = assemble(bc_M_u.M * M_u, 'u')
b_u_bc = assemble(bc_M_u.b * b_u, 'u')

M_p_bc = assemble(bc_M_p.M * M_p, 'p')
b_p_bc = assemble(bc_M_p.b * b_p, 'p')

M_T_bc = assemble(bc_M_T.M * M_T, 'T')
b_T_bc = assemble(bc_M_T.b * b_T, 'T')

### Step 5: Solve the system of equations
# Define the solver parameters
solver_parameters = {
    'linear_solver': 'lu',
    'preconditioner': 'sor',
    'max_iter': 1000,
    'tolerance': 1e-6
}

# Create a solver object
u, p, T = Function(V_u)
M_u, b_u = M_u_bc
M_p, b_p = M_p_bc
M_T, b_T = M_T_bc

solver = LinearVariationalSolver(M_u, b_u, M_p, b_p, M_T, b_T, solver_parameters=solver_parameters)

# Solve the system of equations
u_solution = solver.solve(u)
p_solution = solver.solve(p)
T_solution = solver.solve(T)

# Save μ(x, y) as a color map image q13_mu.png
mu = mu_ref * exp(-beta * (T_solution - T_ref))
plot(mu, title='μ(x,y)')
save('q13_mu.png')

# Extract the streamwise velocity profile u_x(y) along the mid-length line (x = 1.0) and save it to a CSV file named q13_profile.csv with columns: y, ux
ux = u_solution[0].eval_form(1)
y_values = np.linspace(0, H, 100)
np.savetxt('q13_profile.csv', np.column_stack((y_values, ux)), delimiter=',')

### Step 6: Export solution fields (u, p, T, μ) in XDMF format for post-processing as q1 _solution.xdmf
# Create an XDMF file object
xdmf_file = XDMFFile('q13_solution.xdmf')

# Add the solution fields to the XDMF file
xdmf_file.append_solution((u, p, T, mu), 0)

# Write the XDMF file
xdmf_file.write()
