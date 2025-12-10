from dolfin import *

# Define parameters
g = 1.0
rho = 1.0
nu = 1.0
k = 1.0
K = 1.0
alpha = 1.0

# Create mesh and finite element function space
mesh = RectangleMesh(Point(0, -1), Point(pi, 1), 64, 128)
V_S = FunctionSpace(mesh, 'P', 2)  # Stokes velocity
V_D = FunctionSpace(mesh, 'P', 2)  # Darcy pressure

# Define boundary conditions as lambda functions
def u_s_boundary(x):
    y, x = x[0], x[1]
    if y == 1 or x == 0:
        return 0.0
    elif y == -1 or x == pi:
        return 0.0
    else:
        return 0.0

def p_d_boundary(x):
    y, x = x[0], x[1]
    if y == -1 or x == 0:
        return 0.0
    else:
        return 0.0

bc_u_s = DirichletBC(V_S, u_s_boundary, 'on_boundary')
bc_p_d = DirichletBC(V_D, p_d_boundary, 'on_boundary')

# Define variational problem
u_s, p_d = TrialFunction(V_S), TrialFunction(V_D)
v_s, v_d = TestFunction(V_S), TestFunction(V_D)

# Stokes equation
F_stokes = -div(epsilon(u_s)) * v_s + dot(nu*grad(p_d), v_s) + integrate(b_x(x,y)*v_s[0] + b_y(x,y)*v_s[1], (x,y) in mesh)
J_stokes = inner(F_stokes, v_s) + rho*inner(grad(u_s), grad(v_s)) + nu*inner(dot(grad(p_d), grad(v_s)))

# Darcy equation
F_darcy = -dot(k/nu*grad(p_d), v_d) + integrate(b_y(x,y)*v_d[1], (x,y) in mesh)
J_darcy = inner(F_darcy, v_d)

# Assemble and solve the coupled system
M_S = assemble(inner(u_s, u_s)*dx)
M_D = assemble(inner(p_d, p_d)*dx)
b_S = assemble(integrate(b_x(x,y)*u_s[0] + b_y(x,y)*u_s[1], (x,y) in mesh))
b_D = assemble(integrate(b_y(x,y)*p_d[1], (x,y) in mesh))

J = J_stokes + J_darcy
M = M_S + M_D

bc_u_s.apply(J, u_s)
bc_p_d.apply(J, p_d)

u_s, p_d = solve((J, M), (v_s, v_d))

# Save the solution to XDMF format
file = File("solution.xdmf")
file << (u_s, p_d)
