
from dolfin import *
import mshr

# Geometry
domain = mshr.Rectangle(Point(0, 0), Point(2, 0.20))
mesh = mshr.generate_mesh(domain, 64)

# Parameters
rho, U_ref, mu_ref, beta, T_ref, kappa = 1.0, 1.0, 0.02, 0.05, 300.0, 1e-3

# Velocity: u(x,y) = (U_ref*y*(H-y)/H^2, 0)
u = Expression(("6*U_ref*x[1]*(x[1] - x[1][1])/pow(x[1][1], 2)", "0.0"), degree=2,
               U_ref=U_ref, H=domain.geometry().max_y())
# Pressure: p(x,y) = 0
p = Expression("0.0", degree=0)

# Viscosity: μ(x,y) = exp(-beta*(T(x,y)-T_ref))
mu = Expression("exp(-(T - T_ref)*beta)", beta=beta, T=T_ref, degree=2)
V = VectorFunctionSpace(mesh, "P", 1)
u_n = TrialFunction(V); v_n = TestFunction(V)
rho, U_ref, mu_ref, beta, T_ref, kappa = Constant(rho), Constant(U_ref), Constant(mu_ref), Constant(beta), Constant(T_ref), Constant(kappa)

# Momentum
a_momentum = rho*dot(u_n,v_n)*dx + 2*mu_ref*inner(eps(u_n), eps(v_n))*dx
L_momentum = rho*u_n[0]*v_n[0]*dx

# Mass conservation
a_mass = dot(grad(u_n), v_n)*dx
L_mass = Constant(0.0)*v_n*dx

# Energy
a_energy = u_n*v_n*dx - kappa*inner(grad(T_n), grad(v_n))*dx
L_energy = Constant(0.0)*v_n*dx

# Assemble the system
A = lhs(a_momentum) + lhs(a_mass) + lhs(a_energy)
L = rhs(L_momentum) + rhs(L_mass) + rhs(L_energy)

# Solve
u_h, p_h, T_h, mu_h = Function(V), Function(V), Function(V), Function(V)
solve(A == L, [u_h, p_h, T_h, mu_h])