from fenics import *

# Define parameters
L = 6.0         # length of the domain
H = 1.0         # height of the domain
h = 0.1         # thickness of the walls
mu_f = 0.003    # fluid viscosity
rho_f = 1.0     # fluid density
rho_s = 1.1     # solid density
nu_s = 0.49     # Poisson ratio of the solid
E_s = 3.0e5     # Young's modulus of the solid
T = 0.1         # total time
dt = 1.0e-4     # time step size

# Create mesh
mesh_f = RectangleMesh(Point(0, -H/2), Point(L, H/2), 100, 20)
mesh_s_upper = RectangleMesh(Point(0, H/2), Point(L, H/2 + h), 100, 5)
mesh_s_lower = RectangleMesh(Point(0, -H/2 - h), Point(L, -H/2), 100, 5)

# Define function spaces
V_f = VectorFunctionSpace(mesh_f, 'P', 2)
Q_f = FunctionSpace(mesh_f, 'P', 1)

V_s_upper = VectorFunctionSpace(mesh_s_upper, 'P', 2)
V_s_lower = VectorFunctionSpace(mesh_s_lower, 'P', 2)

# Define trial and test functions
u_f = TrialFunction(V_f)
p_f = TrialFunction(Q_f)
v_f = TestFunction(V_f)
q_f = TestFunction(Q_f)

u_s_upper = TrialFunction(V_s_upper)
v_s_upper = TestFunction(V_s_upper)

u_s_lower = TrialFunction(V_s_lower)
v_s_lower = TestFunction(V_s_lower)

# Define boundary conditions
def inlet(x, on_boundary):
    return on_boundary and near(x[0], 0)

def outlet(x, on_boundary):
    return on_boundary and near(x[0], L)

bc_f_inlet = DirichletBC(V_f, (0, 0), inlet)
bc_f_outlet = DirichletBC(Q_f, 0, outlet)

bc_s_upper = DirichletBC(V_s_upper, (0, 0), 'on_boundary')
bc_s_lower = DirichletBC(V_s_lower, (0, 0), 'on_boundary')

# Define variational forms
a_u = rho_f * dot(u_f, v_f) * dx + mu_f * inner(grad(u_f), grad(v_f)) * dx
L_u = Constant(0) * v_f[0] * dx

a_p = p_f * q_f * dx
L_p = Constant(0) * q_f * dx

a_s_upper = rho_s * dot(u_s_upper, v_s_upper) * dx + E_s / (1 + nu_s) * inner(sym(grad(u_s_upper)), sym(grad(v_s_upper))) * dx
n = FacetNormal(mesh_s_upper)
L_s_upper = -mu_f * u_f[0] * n[0] * ds(1, domain=mesh_s_upper)

a_s_lower = rho_s * dot(u_s_lower, v_s_lower) * dx + E_s / (1 + nu_s) * inner(sym(grad(u_s_lower)), sym(grad(v_s_lower))) * dx
n = FacetNormal(mesh_s_lower)
L_s_lower = -mu_f * u_f[0] * n[0] * ds(2, domain=mesh_s_lower)

# Time-stepping
t = 0
u_f_n = Function(V_f)
p_f_n = Function(Q_f)
u_s_upper_n = Function(V_s_upper)
u_s_lower_n = Function(V_s_lower)

while t < T:
    # Update boundary conditions
    bc_f_inlet = DirichletBC(V_f, (-2e4 / 2 * (1 - cos(pi * t / 2.5e-3)), 0) if t < 2.5e-3 else (0, 0), inlet)

    # Solve fluid equations
    solve(a_u == L_u, u_f_n)
    solve(a_p == L_p, p_f_n)

    # Solve solid equations
    solve(a_s_upper == L_s_upper, u_s_upper_n)
    solve(a_s_lower == L_s_lower, u_s_lower_n)

    # Update time
    t += dt

    # Save output at specific times
    if near(t, 5e-3):
        file_u = XDMFFile('velocity_5ms.xdmf')
        file_u.write(u_f_n, t)
        file_p = XDMFFile('pressure_5ms.xdmf')
        file_p.write(p_f_n, t)

    if near(t, 1e-1):
        file_u = XDMFFile('velocity_100ms.xdmf')
        file_u.write(u_f_n, t)
        file_p = XDMFFile('pressure_100ms.xdmf')
        file_p.write(p_f_n, t)
