# filename: navier_stokes.py
from fenics import *

# Define parameters
H = 1.0               # Height of the channel
U_bar = 1.0           # Mean inlet speed
rho = 1.0             # Density
mu = 0.01             # Dynamic viscosity

# Create mesh
mesh = RectangleMesh(Point(-3*H, 0), Point(20*H, 2*H), 100, 40)
mesh = Mesh(mesh)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
u_inlet = Expression(('6*U_bar*(x[1]/H)*(1-x[1]/H)', '0'), U_bar=U_bar, H=H, degree=2)
u_wall = Constant((0, 0))

bc_u_inlet = DirichletBC(V, u_inlet, 'on(x[0] == -3*H)')
bc_u_wall_bottom = DirichletBC(V, u_wall, 'on(x[1] == 0)')
bc_u_wall_top_upstream = DirichletBC(V, u_wall, 'on(x[1] == H) && (x[0] < 0)')
bc_u_wall_top_downstream = DirichletBC(V, u_wall, 'on(x[1] == 2*H) && (x[0] > 0)')
bc_u_wall_step = DirichletBC(V, u_wall, 'on(x[0] == 0) && (x[1] > H)')

bcs_u = [bc_u_inlet, bc_u_wall_bottom, bc_u_wall_top_upstream, bc_u_wall_top_downstream, bc_u_wall_step]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

a = inner(grad(u), grad(v))*dx + inner(p, div(v))*dx
L = Constant(0)*v[0]*dx

a_p = p*q*dx
L_p = -div(u)*q*dx

# Solve variational problem
u_sol = Function(V)
p_sol = Function(Q)

solve(a == L, u_sol, bcs_u)
solve(a_p == L_p, p_sol)

# Compute wall shear stress
tau_w = mu * u_sol.dx(1)

# Save velocity field as image
plot(u_sol, title='Velocity field')
interactive()

# Save solution fields in XDMF format
u_sol_file = XDMFFile('q6_u.xdmf')
p_sol_file = XDMFFile('q6_p.xdmf')

u_sol_file.write(u_sol)
p_sol_file.write(p_sol)

# Compute re-attachment point
x = np.linspace(0, 20*H, 1000)
tau_w_values = []
for xi in x:
    tau_w_value = tau_w(xi, H)
    tau_w_values.append(tau_w_value)

re_attachment_point = x[np.argmin(np.abs(tau_w_values))]
print('Re-attachment point:', re_attachment_point)