from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt
import numpy as np

# Geometry
mesh = generate_mesh(Rectangle(Point(0, 0), Point(2.0, 0.20)), 64)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p, T) = TrialFunctions(W)
(v, q, s) = TestFunctions(W)

# Parameters
rho = 1.0
mu_ref = 0.02
beta = 0.05
T_ref = 300.0
kappa = 1e-3
U_bar = 1.0
H = 0.20

# Boundary markers
bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bndry.set_all(0)
class Inlet(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0)
class Outlet(SubDomain):
    def inside(self, x, on): return on and near(x[0], 2.0)
class Bottom(SubDomain):
    def inside(self, x, on): return on and near(x[1], 0.0)
class Top(SubDomain):
    def inside(self, x, on): return on and near(x[1], 0.20)
Inlet().mark(bndry, 1)
Outlet().mark(bndry, 2)
Bottom().mark(bndry, 3)
Top().mark(bndry, 4)
ds = Measure("ds", domain=mesh, subdomain_data=bndry)

# Boundary conditions
# Inlet velocity
u_in = Expression(("6*U_bar*x[1]*(H - x[1])/(H*H)", "0.0"),
                  U_bar=U_bar, H=H, degree=2)
bc_in_vel = DirichletBC(W.sub(0), u_in, bndry, 1)
# Inlet temperature
bc_in_temp = DirichletBC(W.sub(2), Constant(T_ref), bndry, 1)
# No-slip on walls
bc_wall_vel = DirichletBC(W.sub(0), Constant((0.0, 0.0)), bndry, 3)
# Bottom wall T = T_ref + 10
bc_bottom_temp = DirichletBC(W.sub(2), Constant(T_ref + 10.0), bndry, 3)
# Top wall ∂T/∂n = 0 (natural)
# Outlet traction-free (natural)
# Outlet ∂T/∂n = 0 (natural)
bcs = [bc_in_vel, bc_in_temp, bc_wall_vel, bc_bottom_temp]

# Variational forms
mu = Expression("mu_ref*exp(-beta*(T - T_ref))",
                mu_ref=mu_ref, beta=beta, T_ref=T_ref, T=T, degree=2)
eps = sym(grad(u))
F_mom = rho*dot(u, grad(u))*v*dx - inner(2*mu*eps, grad(v))*dx + div(v)*p*dx
F_mass = div(u)*q*dx
F_temp = dot(u, grad(T))*s*dx - kappa*dot(grad(T), grad(s))*dx
F = F_mom + F_mass + F_temp

# Solve
w = Function(W)
solve(lhs(F) == rhs(F), w, bcs, solver_parameters={"newton_solver": {"maximum_iterations": 50}})
u_sol, p_sol, T_sol = w.split()

# Post-processing
mu_sol = project(Expression("mu_ref*exp(-beta*(T - T_ref))",
                            mu_ref=mu_ref, beta=beta, T_ref=T_ref, T=T_sol, degree=2),
                 FunctionSpace(mesh, "CG", 1))
plt.figure(); c = plot(mu_sol); plt.colorbar(c); plt.savefig("q13_mu.png")

# Velocity profile at x=1.0
coords = mesh.coordinates()
cells = mesh.cells()
y_vals = coords[cells].mean(axis=1)[:, 1]
ux_vals = u_sol.vector().get_local()[::2]
np.savetxt("q13_profile.csv", np.column_stack((y_vals, ux_vals)),
           header="y, ux", comments="")

# Export
with XDMFFile("q13_solution.xdmf") as xf:
    xf.write(u_sol, 0.0)
    xf.write(p_sol, 0.0)
    xf.write(T_sol, 0.0)
    xf.write(mu_sol, 0.0)