# Import necessary modules from FEniCS
from fenics import *

# Define constants
L = 2.0  # length of the channel (m)
H = 0.20  # height of the channel (m)
N_x = 160  # number of elements in x-direction
N_y = 16   # number of elements in y-direction

# Create a mesh of the rectangular channel
mesh = RectangleMesh(Point(0, 0), Point(L, H), N_x, N_y)

# Define boundary conditions for velocity and pressure
def inlet_bc(x, on_boundary):
    return on_boundary and x[1] == 0

def wall_bc(x, on_boundary):
    return on_boundary and (x[0] == 0 or x[0] == L)

def outlet_bc(x, on_boundary):
    return on_boundary and x[1] == H

bc_values = {
    'inlet': {'u_x': lambda x: 6 * 2.5 / H * (1 - x[1]/H), 'u_y': 0},
    'wall': {'u_x': 0, 'u_y': 0},
    'outlet': {'p': 0}
}

# Create a function space for the velocity and pressure fields
V = FunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define the boundary conditions
u_x_bc = DirichletBC(V, bc_values['inlet']['u_x'], inlet_bc)
u_y_bc = DirichletBC(V, bc_values['inlet']['u_y'], inlet_bc)
p_bc = DirichletBC(Q, bc_values['outlet']['p'], outlet_bc)

# Create a variational problem
u, p = Function(V), Function(Q)
problem = MixedFunctional(u*u + p*p, u*dV, p*ds(0))

# Compute the variational residual
J = problem.jacobian()

# Set up the solver parameters
solver_parameters = {
    'linear_solver': 'lu',
    'preconditioner': 'sor',
    'max_iter': 1000,
    'output_prefix': 'q4_'
}

# Solve the system of equations
solve(problem, u_x_bc, u_y_bc, p_bc, solver_parameters=solver_parameters)

# Save a color map of u_x over Ω as q4_ux.png
plot(u_x, cmap='viridis')
save('q4_ux.png')

# Save the velocity field (u) and pressure field (p) to q4_soln.xdmf
plot((u, p), format='xdmf', filename='q4_soln.xdmf')