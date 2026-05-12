# Backward-facing step – steady incompressible Navier–Stokes (legacy dolfin)

from dolfin import *
import matplotlib.pyplot as plt

# ------ Geometry & mesh (single conforming mesh) ------
H      = 1.0                     # step height (m)
L_up   = 3.0 * H                 # upstream length
L_down = 20.0 * H                # downstream length

# Global resolution (feel free to refine locally later)
nx_up   = 30
ny_up   = 30
nx_down = 200
ny_down = 60

mesh = RectangleMesh(Point(-L_up, 0.0),
                     Point( L_down, 2.0*H),
                     nx_up + nx_down,   # total cells in x
                     ny_down)            # cells in y (covers both heights)

# ------ Boundary markers (single mesh) ------
tol = DOLFIN_EPS
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -L_up, tol) and (0.0 - tol <= x[1] <= H + tol)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)

class TopUp(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H, tol) and (x[0] <= 0.0 + tol)

class TopDown(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 2.0*H, tol) and (x[0] >= 0.0 - tol)

class StepWall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol) and (x[1] >= H - tol)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L_down, tol) and (0.0 - tol <= x[1] <= 2.0*H + tol)

Inlet().mark(boundaries, 1)
Bottom().mark(boundaries, 2)
TopUp().mark(boundaries, 3)
TopDown().mark(boundaries, 4)
StepWall().mark(boundaries, 5)
Outlet().mark(boundaries, 6)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# ------ Function spaces (Taylor–Hood) ------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# ------ Boundary conditions ------
U_bar = 1.0
mu    = 1.0e-2
rho   = 1.0

inlet_expr = Expression(("6.0*U_bar*x[1]/H*(1.0 - x[1]/H)", "0.0"),
                        degree=2, U_bar=U_bar, H=H)

noslip = Constant((0.0, 0.0))
bcu_inlet   = DirichletBC(W.sub(0), inlet_expr, boundaries, 1)
bcu_bottom  = DirichletBC(W.sub(0), noslip, boundaries, 2)
bcu_topup   = DirichletBC(W.sub(0), noslip, boundaries, 3)
bcu_topdown = DirichletBC(W.sub(0), noslip, boundaries, 4)
bcu_step    = DirichletBC(W.sub(0), noslip, boundaries, 5)

class Corner(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -L_up, tol) and near(x[1], 0.0, tol)

bcp_corner = DirichletBC(W.sub(1), Constant(0.0), Corner(), method="pointwise")
bcs = [bcu_inlet, bcu_bottom, bcu_topup, bcu_topdown, bcu_step, bcp_corner]

# ------ Variational formulation (steady Navier–Stokes) ------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

U = Function(W)          # current iterate
(u_k, p_k) = split(U)    # for Newton linearisation

def epsilon(w):
    return sym(grad(w))

# ---- corrected Navier–Stokes residual ----
F = ( rho*dot(dot(u_k, nabla_grad(u_k)), v)*dx
      + 2*mu*inner(epsilon(u_k), epsilon(v))*dx
      - div(v)*p_k*dx
      + q*div(u_k)*dx )
J = derivative(F, U, TrialFunction(W))

# ------ Stokes pre-solve (good initial guess) ------
U0 = Function(W)                     # zero initial guess
(u0, p0) = split(U0)

# ---- corrected Stokes residual ----
F_stokes = ( 2*mu*inner(epsilon(u0), epsilon(v))*dx
             - div(v)*p0*dx
             + q*div(u0)*dx )
J_stokes = derivative(F_stokes, U0, TrialFunction(W))

solve(F_stokes == 0, U0, bcs, J=J_stokes,
      solver_parameters={"newton_solver": {"linear_solver": "mumps"}})   # legacy key

U.assign(U0)   # use Stokes solution as Newton start

# ------ Newton solve ------
solve(F == 0, U, bcs, J=J,
      solver_parameters={"newton_solver":
                         {"linear_solver": "mumps",
                          "relative_tolerance": 1e-6,
                          "absolute_tolerance": 1e-8,
                          "maximum_iterations": 80,
                          "relaxation_parameter": 0.7,
                          "error_on_nonconvergence": False}})

# ------ Split solution and post-processing ------
(u_sol, p_sol) = U.split(deepcopy=True)

# Wall shear stress on downstream top wall (y = 2H)
Vdg = FunctionSpace(mesh, "DG", 0)   # facetwise scalar
tau_w = Function(Vdg)

grad_u = grad(u_sol)
tau_expr = mu * grad_u[0, 1]          # ∂u_x/∂y
tau_proj = project(tau_expr, Vdg, solver_type="cg")
tau_w.assign(tau_proj)

# Keep only the downstream top wall (boundary id 4)
tau_w_top = Function(Vdg)
tau_w_top.vector()[:] = 0.0
for cell in cells(mesh):
    mp = cell.midpoint()
    if mp.y() >= 2.0*H - tol and mp.x() >= 0.0 - tol:
        tau_w_top.vector()[cell.index()] = tau_w.vector()[cell.index()]

# ------ Output ------
# Velocity magnitude PNG
u_mag = sqrt(dot(u_sol, u_sol))
u_mag = project(u_mag, FunctionSpace(mesh, "CG", 2))
plt.figure()
p = plot(u_mag, title="Velocity magnitude")
plt.colorbar(p)
plt.savefig("q6_u.png", dpi=300)

# XDMF solution
with XDMFFile(mesh.mpi_comm(), "q6_soln.xdmf") as xdmf:
    xdmf.write(u_sol, 0.0)
    xdmf.write(p_sol, 0.0)

# Shear stress on downstream top wall
with XDMFFile(mesh.mpi_comm(), "q6_tau_top.xdmf") as xdmf:
    xdmf.write(tau_w_top, 0.0)