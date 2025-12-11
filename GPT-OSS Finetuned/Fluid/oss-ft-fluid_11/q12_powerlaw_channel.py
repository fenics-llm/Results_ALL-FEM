# filename: q12_powerlaw_channel.py
from dolfin import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Geometry and mesh
L, H = 2.0, 0.20
nx, ny = 240, 24
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny, "crossed")

# Taylor–Hood mixed space (P2-P1)
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# Boundaries
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], H))
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)
class PointZero(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)

inlet   = Inlet()
walls   = Walls()
outlet  = Outlet()
pt_zero = PointZero()

# Inlet profile
U_bar = 1.0
inlet_profile = Expression(("6.0*U_bar*x[1]*(H - x[1]) / (H*H)", "0.0"),
                           U_bar=U_bar, H=H, degree=2)

# Dirichlet BCs
bcs = []
bcs.append(DirichletBC(W.sub(0), inlet_profile, inlet))
bcs.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls))
bcs.append(DirichletBC(W.sub(1), Constant(0.0), pt_zero, method='pointwise'))

# Power‑law fluid parameters
rho = 1.0
mu0 = 0.5
n   = 0.5
eps = 1e-8

# Variational formulation
w = Function(W)               # (u,p)
(u, p) = split(w)
(v, q) = TestFunctions(W)

D = sym(grad(u))
gamma = sqrt(2*inner(D, D) + eps)
mu_eff = mu0*pow(gamma, n - 1.0)

F = ( 2*mu_eff*inner(sym(grad(u)), sym(grad(v))) * dx
      - div(v)*p*dx
      - q*div(u)*dx
      - rho*dot(dot(u, nabla_grad(u)), v)*dx )
J = derivative(F, w, TrialFunction(W))

# Solve
problem = NonlinearVariationalProblem(F, w, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["newton_solver"]["relative_tolerance"] = 1e-6
prm["newton_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["linear_solver"] = "mumps"
solver.solve()

# Split solution
(u_sol, p_sol) = w.split(deepcopy=True)

# Effective viscosity field
V_mu = FunctionSpace(mesh, "P", 1)
D_sol = sym(grad(u_sol))
gamma_sol = sqrt(2*inner(D_sol, D_sol) + eps)
mu_eff_func = project(mu0*pow(gamma_sol, n - 1.0), V_mu)

# Save to XDMF
with XDMFFile(mesh.mpi_comm(), "q12_solution.xdmf") as xdmf:
    xdmf.write(u_sol, 0.0)
    xdmf.write(p_sol, 0.0)
    xdmf.write(mu_eff_func, 0.0)

# Velocity magnitude plot
u_mag = sqrt(dot(u_sol, u_sol))
plt.figure(figsize=(8, 2))
p = plot(u_mag, title="Velocity magnitude |u|", cmap="viridis")
plt.colorbar(p)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q12_speed.png", dpi=300)

# ux profile at x = L/2
y_vals = np.linspace(0.0, H, ny*5 + 1)
mid_x = L/2
ux_vals = np.array([u_sol(Point(mid_x, y))[0] for y in y_vals])

# Save profile
np.savetxt("q12_profile.csv", np.column_stack([y_vals, ux_vals]),
           header="y (m), ux (m/s)", delimiter=",", comments='')

# Report max ux
max_ux = np.max(ux_vals)
print("Maximum streamwise velocity ux at x = L/2 : {:.6f} m/s".format(max_ux))