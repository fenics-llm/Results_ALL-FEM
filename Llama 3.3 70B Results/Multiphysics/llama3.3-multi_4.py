from fenics import *

# Parameters
g = 1
rho = 1
nu = 1
k = 1
K = k * rho * g / nu
alpha = 1

# Define the mesh
mesh = RectangleMesh(Point(0, -1), Point(pi, 1), 100, 20)

# Define the function spaces
V_S = VectorFunctionSpace(mesh, 'P', 2)
Q_S = FunctionSpace(mesh, 'P', 1)
V_D = VectorFunctionSpace(mesh, 'P', 2)
Q_D = FunctionSpace(mesh, 'P', 1)

# Define the trial and test functions
u_S = TrialFunction(V_S)
p_S = TrialFunction(Q_S)
v_S = TestFunction(V_S)
q_S = TestFunction(Q_S)
u_D = TrialFunction(V_D)
p_D = TrialFunction(Q_D)
v_D = TestFunction(V_D)
q_D = TestFunction(Q_D)

# Define the body force
x, y = SpatialCoordinate(mesh)
b_x = ((nu*K) - (alpha*g)/(2*nu)) * y - g/2
b_y = (((nu*K)/2) - (alpha*g)/(4*nu)) * y**2 - (g/2) * y + ((alpha*g)/(2*nu) - 2*nu*K)

# Define the boundary conditions for Stokes
def stokes_bc(x, on_boundary):
    return on_boundary and (x[1] == 1 or x[0] == 0 or x[0] == pi)

u_S_bc = DirichletBC(V_S, Expression(('w_prime(y)*cos(x)', 'w(y)*sin(x)'), degree=2), stokes_bc)

# Define the boundary conditions for Darcy
def darcy_bc(x, on_boundary):
    return on_boundary and (x[1] == -1 or x[0] == 0 or x[0] == pi)

p_D_bc = DirichletBC(Q_D, Expression('rho*g*exp(y)*sin(x)', degree=2), darcy_bc)

# Define the variational formulation for Stokes
a_S = inner(grad(u_S), grad(v_S))*dx + div(u_S)*q_S*dx - p_S*div(v_S)*dx
L_S = b_x*v_S[0]*dx + b_y*v_S[1]*dx

# Define the variational formulation for Darcy
a_D = (k/nu)*inner(grad(p_D), grad(q_D))*dx 
L_D = Constant(0)*q_D*dx

# Assemble and solve the system
u_S_sol = Function(V_S)
p_S_sol = Function(Q_S)
u_D_sol = Function(V_D)
p_D_sol = Function(Q_D)

solve(a_S == L_S, (u_S_sol, p_S_sol), [u_S_bc])
solve(a_D == L_D, p_D_sol, [p_D_bc])

# Save the solutions to XDMF format
XDMFFile('stokes_velocity.xdmf').write(u_S_sol)
XDMFFile('darcy_pressure.xdmf').write(p_D_sol)