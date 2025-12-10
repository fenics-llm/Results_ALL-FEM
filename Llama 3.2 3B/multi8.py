from dolfin import *

# Define parameters
mu = 0.02  # Dynamic Viscosity in Pa·s
K = 1e-6   # Permeability in m²
H = 0.2    # Channel Height in m
U_bar = 0.1  # Mean Inlet Speed in m/s

# Create mesh and finite element function space
mesh = UnitSquareMesh(100, 100)
V_f = VectorFunctionSpace(mesh, 'P', 1)  # Function space for fluid velocity
Q_f = FunctionSpace(mesh, 'P', 1)  # Function space for fluid pressure
V_p = VectorFunctionSpace(mesh, 'P', 1)  # Function space for porous medium velocity
Q_p = FunctionSpace(mesh, 'P', 1)  # Function space for porous medium pressure

# Define boundary conditions
def inlet_bc(x):
    return x[0] == 0.0

def fluid_walls_bc(x):
    return x[0] == 0.0 or x[0] == 1.0 or x[1] == 0.0 or x[1] == H

def porous_walls_bc(x):
    return x[0] == 0.0 or x[0] == 1.0 or x[1] == 0.0 or x[1] == H

# Define boundary conditions for velocity and pressure
u_f_bc = DirichletBC(V_f, Constant(0), 'left')
p_f_bc = DirichletBC(Q_f, Constant(0), 'left')

u_p_bc = NeumannBC(V_p, Constant(0), 'right')
p_p_bc = NeumannBC(Q_p, Constant(0), 'right')

# Define the variational problem for fluid and porous medium
def fluid_problem(u_f, p_f):
    v_f = TestFunction(V_f)
    q_f = TestFunction(Q_f)
    u_f_n = u_f[0] * dx

    a_f = inner(epsilon(u_f), epsilon(v_f)) * dx + inner(p_f, q_f) * dx
    L_f = dot(grad(u_f), v_f) * dx

    return a_f, L_f

def porous_problem(u_p, p_p):
    v_p = TestFunction(V_p)
    u_p_n = u_p[0] * dx

    a_p = inner(K/μ * gradient(p_p), gradient(v_p)) * dx
    L_p = -dot(grad(u_p), v_p) * dx

    return a_p, L_p

# Define the variational problem for interface conditions
def interface_problem(u_f, u_p):
    v_u = TestFunction(V_f)
    u_t = u_f[1] * dx

    a_i = inner(u_t, v_u) * dx
    L_i = 0

    return a_i, L_i

# Define the function to solve
def solve():
    # Initialize the variational problem
    u_f, p_f = Function(V_f), Function(Q_f)
    u_p, p_p = Function(V_p), Function(Q_p)

    # Assemble the system of equations
    a, L = fluid_problem(u_f, p_f)
    b, L = porous_problem(u_p, p_p)

    a_i, L_i = interface_problem(u_f, u_p)

    # Solve the system of equations
    solve(a == 0, L == 0)
    solve(b == 0, L == 0)
    solve(a_i == 0, L_i == 0)

# Solve the problem
solve()