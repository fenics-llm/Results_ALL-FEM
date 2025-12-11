from fenics import *

# Define geometry parameters
L = 2.5  # length of channel
H = 0.41  # height of channel
flag_length = 0.35  # length of flag
flag_thickness = 0.02  # thickness of flag
pole_radius = 0.05  # radius of pole

# Define material parameters
rho_f = 1000  # fluid density (kg/m^3)
nu_f = 1e-3  # fluid kinematic viscosity (m^2/s)
rho_s = 1000  # solid density (kg/m^3)
nu_s = 0.4  # Poisson ratio of solid
mu_s = 5e5  # shear modulus of solid (Pa)

# Define boundary conditions
U_bar = 0.2  # average velocity at inlet (m/s)

# Create mesh for fluid domain
mesh_f = RectangleMesh(Point(0, 0), Point(L, H), 100, 40)
mesh_f = Mesh(mesh_f)

# Create mesh for solid domain
mesh_s = RectangleMesh(Point(0.6, 0.19), Point(0.95, 0.21), 35, 2)
mesh_s = Mesh(mesh_s)

# Define function spaces for fluid and solid domains
V_f = VectorFunctionSpace(mesh_f, 'P', 2)
Q_f = FunctionSpace(mesh_f, 'P', 1)
V_s = VectorFunctionSpace(mesh_s, 'P', 2)

# Define trial and test functions for fluid domain
u_f, v_f = TrialFunction(V_f), TestFunction(V_f)
p_f, q_f = TrialFunction(Q_f), TestFunction(Q_f)

# Define trial and test functions for solid domain
u_s, v_s = TrialFunction(V_s), TestFunction(V_s)

# Define variational forms for fluid domain
F_f = inner(grad(u_f), grad(v_f))*dx + nu_f*inner(div(u_f), q_f)*dx - p_f*div(v_f)*dx

# Define variational forms for solid domain
E_s = mu_s/(1+nu_s)
F_s = E_s*inner(sym(grad(u_s)), sym(grad(v_s)))*dx

# Impose boundary conditions on fluid domain
bc_u_f = DirichletBC(V_f, (0, 0), 'on_boundary && near(x[1], 0)')
bc_u_f_2 = DirichletBC(V_f, (0, 0), 'on_boundary && near(x[1], H)')
bc_p_f = DirichletBC(Q_f, 0, 'on_boundary && near(x[0], L)')

# Impose boundary conditions on solid domain
bc_u_s = DirichletBC(V_s, (0, 0), 'on_boundary && near(x[0], 0.6)')

# Solve fluid-structure interaction problem
u_f_sol = Function(V_f)
p_f_sol = Function(Q_f)
u_s_sol = Function(V_s)

solve(F_f == 0, u_f_sol, bc_u_f)
solve(F_f == 0, p_f_sol, bc_p_f)
solve(F_s == 0, u_s_sol, bc_u_s)

# Save solution to XDMF file
xdmffile_u_f = XDMFFile('fluid_velocity.xdmf')
xdmffile_p_f = XDMFFile('fluid_pressure.xdmf')
xdmffile_u_s = XDMFFile('solid_displacement.xdmf')

xdmffile_u_f.write(u_f_sol, 0)
xdmffile_p_f.write(p_f_sol, 0)
xdmffile_u_s.write(u_s_sol, 0)

# Report displacement components of point A
A_x = 0.6
A_y = 0.2

u_A_x = u_s_sol(A_x, A_y)[0]
u_A_y = u_s_sol(A_x, A_y)[1]

print('Displacement components of point A: ({}, {})'.format(u_A_x, u_A_y))