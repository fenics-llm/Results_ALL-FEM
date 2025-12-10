# filename: solve_fenics.py

from fenics import *

# Define parameters
rho = 1.0  # kg/m^3
mu = 1.5e-5  # Pa·s
alpha = 2.1e-5  # m² s⁻¹
g_beta = 3.15e-5  # m s-² K-¹
T_ref = 0.5  # K

# Define the domain and boundaries
Omega = UnitInterval()  # unit interval domain
left_wall = Omega.subdomain('left')
right_wall = Omega.subdomain('right')
top_wall = Omega.subdomain('top')
bottom_wall = Omega.subdomain('bottom')

# Define the boundary conditions
def left_bc(x):
    return Constant(1.0)

def right_bc(x):
    return Constant(0.0)

def top_bottom_bc(x, on_boundary):
    return on_boundary

left_bc.apply(left_wall)
right_bc.apply(right_wall)
top_bottom_bc.apply(top_wall | bottom_wall)

# Define the variational problem
u = Function('Velocity')
p = Function('Pressure')
T = Function('Temperature')

V = VectorFunctionSpace(Omega, 'P', 1)  # velocity space
Q = FunctionSpace(Omega, 'P', 1)  # pressure space

u_l = TrialFunction(V)
p_l = TrialFunction(Q)
T_l = TrialFunction(V)

v_u = TestFunction(V)
v_p = TestFunction(Q)
v_T = TestFunction(V)

# Define the Navier-Stokes and advection-diffusion equations
F_u = rho * inner(u_l * v_u, V) + mu * inner(grad(u_l), grad(v_u), V) - inner(f, v_u, V)
F_p = -inner(grad(T_l), grad(v_p), Q)
F_T = alpha * inner(grad(T_l), grad(v_T), V)

# Define the body force f
f = Constant(0.0) + rho * g_beta * (T_l - T_ref)

# Assemble the variational problem
a_u, b_u = assemble(F_u), assemble(b_u)
a_p, b_p = assemble(F_p), assemble(b_p)
a_T, b_T = assemble(F_T), assemble(b_T)

# Solve the system
solve(a_u == a_p + a_T, u_l, v_u=V, solver_parameters={'linear_solver': 'lu'})
solve(a_p == 0, p_l, v_p=Q, solver_parameters={'linear_solver': 'lu'})
solve(a_T == 0, T_l, v_T=V, solver_parameters={'linear_solver': 'lu'})

# Post-processing
T_avg = average(T_l)
print(f'Average Nusselt number at the left wall: {2 * T_avg / (1 - T_avg)}')

# Save the solution to files
q11_T = plot(T_l, title='Temperature field', filename='q11_T.png')
plot(u_l, title='Velocity field', filename='q11_velocity.xdmf')
plot(p_l, title='Pressure field', filename='q11_pressure.xdmf')
plot(T_l, title='Temperature field', filename='q11_solution.xdmf')