from dolfin import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------ Mesh and geometry ------
L = 2.0
H = 0.20
Nx = 200
Ny = 20
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), Nx, Ny, "crossed")

# ------ Physical parameters ------
rho   = 1.0          # kg/m^3
mu    = 0.01         # Pa·s
Ubar  = 0.1          # m/s
kappa = 1.0e-3       # m^2/s

# ------ Function spaces (Taylor–Hood) ------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W_NS = FunctionSpace(mesh, TH)

W_c = FunctionSpace(mesh, "Lagrange", 1)

# ------ Boundary definitions ------
tol = 1E-14

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, tol)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0, tol) or near(x[1], H, tol))

inlet  = Inlet()
outlet = Outlet()
walls  = Walls()

# ------ Inlet velocity (Poiseuille profile) ------
u_inlet_expr = Expression(("6.0*Ubar*x[1]/H*(1.0 - x[1]/H)", "0.0"),
                          H=H, Ubar=Ubar, degree=2)

# ------ Dirichlet BCs for Navier–Stokes ------
bcu_inlet = DirichletBC(W_NS.sub(0), u_inlet_expr, inlet)
bcu_walls = DirichletBC(W_NS.sub(0), Constant((0.0, 0.0)), walls)

class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, tol) and near(x[1], 0.0, tol)

bc_p_gauge = DirichletBC(W_NS.sub(1), Constant(0.0), PressurePoint(), method="pointwise")

bcs_NS = [bcu_inlet, bcu_walls, bc_p_gauge]

# ------ Navier–Stokes variational problem (Picard iteration) ------
(u, p) = TrialFunctions(W_NS)
(v, q) = TestFunctions(W_NS)

# Mixed solution and its components
u_k = Function(W_NS)               # contains (u,p)
(u_k_, p_k_) = split(u_k)

# Picard linearisation: convection uses previous iterate u_k_
F_NS = ( rho*dot(dot(u_k_, nabla_grad(u)), v) * dx
         + 2*mu*inner(sym(grad(u)), sym(grad(v))) * dx
         - p*div(v) * dx
         + q*div(u) * dx )

a_NS, L_NS = lhs(F_NS), rhs(F_NS)

tol_NS = 1e-8
max_iter_NS = 30
for iter in range(max_iter_NS):
    solve(a_NS == L_NS, u_k, bcs_NS)
    r = assemble(L_NS - a_NS*u_k)
    if norm(r, 'l2') < tol_NS:
        break

# ------ Scalar transport (steady advection–diffusion) ------
c = TrialFunction(W_c)
w = TestFunction(W_c)
c_k = Function(W_c)

# Dirichlet BCs for concentration
bc_c_inlet  = DirichletBC(W_c, Constant(0.0), inlet)
bc_c_outlet = DirichletBC(W_c, Constant(1.0), outlet)
bcs_c = [bc_c_inlet, bc_c_outlet]

# Weak form
u_sol, p_sol = u_k.split(deepcopy=True)
a_c = ( dot(u_sol, nabla_grad(c))*w + kappa*dot(nabla_grad(c), nabla_grad(w)) ) * dx
L_c = Constant(0.0)*w*dx

solve(a_c == L_c, c_k, bcs_c)

# ------ Output to XDMF ------
xdmf = XDMFFile(mesh.mpi_comm(), "q10_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.write(c_k, 0.0)
xdmf.close()

# ------ Plot concentration and save as PNG ------
plt.figure(figsize=(8, 2))
c_plot = plot(c_k, mesh=mesh, cmap="viridis")
plt.colorbar(c_plot, label="Concentration")
plt.title("Steady concentration field")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q10_conc.png", dpi=300)
plt.close()