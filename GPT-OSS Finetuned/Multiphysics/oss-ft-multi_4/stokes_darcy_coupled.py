# filename: stokes_darcy_coupled.py
from dolfin import *
import math

# ------------------- Parameters -------------------
g = 1.0
rho = 1.0
nu = 1.0
k = 1.0
mu = 1.0
alpha = 1.0
K = k * rho * g / mu  # =1

# ------------------- Mesh -------------------
nx, ny = 48, 48
mesh = RectangleMesh(Point(0.0, -1.0), Point(math.pi, 1.0), nx, ny)

# ------------------- Subdomains -------------------
domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)
class StokesDomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] >= 0.0 - DOLFIN_EPS
class DarcyDomain(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= 0.0 + DOLFIN_EPS
StokesDomain().mark(domains, 1)
DarcyDomain().mark(domains, 2)
dx = Measure("dx", domain=mesh, subdomain_data=domains)

# ------------------- Boundaries -------------------
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0)
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], -1.0)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], math.pi)
class Interface(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0)
Top().mark(boundaries, 1)
Bottom().mark(boundaries, 2)
Left().mark(boundaries, 3)
Right().mark(boundaries, 4)
Interface().mark(boundaries, 5)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# ------------------- Exact solutions -------------------
w_expr = Expression("-K - (g*x[1])/(2*nu) + (K/2 - alpha*g/(4*nu*nu))*x[1]*x[1]",
                    degree=2, K=K, g=g, nu=nu, alpha=alpha)
dw_expr = Expression("-(g)/(2*nu) + (K - alpha*g/(2*nu*nu))*x[1]",
                     degree=2, K=K, g=g, nu=nu, alpha=alpha)

uS_exact = Expression(("dw_expr*cos(x[0])",
                        "w_expr*sin(x[0])"),
                       degree=3, w_expr=w_expr, dw_expr=dw_expr)

pD_exact = Expression("rho*g*exp(x[1])*sin(x[0])",
                       degree=3, rho=rho, g=g)

# ------------------- Stokes (Taylor–Hood) -------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W = FunctionSpace(mesh, TH)

(uS, pS) = TrialFunctions(W)
(vS, qS) = TestFunctions(W)

# Body force
b_expr = Expression((
    "((nu*K - (alpha*g)/(2*nu))*x[1] - g/2)*cos(x[0])",
    "(((nu*K)/2 - (alpha*g)/(4*nu))*x[1]*x[1] - (g/2)*x[1] + ((alpha*g)/(2*nu) - 2*nu*K))*sin(x[0])"
    ),
    degree=3, nu=nu, K=K, alpha=alpha, g=g)

aS = (2*nu*inner(sym(grad(uS)), sym(grad(vS))) - div(vS)*pS - qS*div(uS))*dx(1)
LStokes = inner(b_expr, vS)*dx(1)

# Dirichlet BCs for velocity on Stokes boundaries (including interface)
bcuS_top    = DirichletBC(W.sub(0), uS_exact, boundaries, 1)
bcuS_left   = DirichletBC(W.sub(0), uS_exact, boundaries, 3)
bcpS_right  = DirichletBC(W.sub(0), uS_exact, boundaries, 4)
bcuS_int    = DirichletBC(W.sub(0), uS_exact, boundaries, 5)

# Pressure gauge (pointwise) to fix nullspace
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.5)
p_point = PressurePoint()
bcpS_point = DirichletBC(W.sub(1), Constant(0.0), p_point, method="pointwise")

bcsS = [bcuS_top, bcuS_left, bcpS_right, bcuS_int, bcpS_point]

# Assemble Stokes system with diagonal preallocation
A_S = PETScMatrix()
b_S = PETScVector()
assemble(aS, tensor=A_S, keep_diagonal=True)
assemble(LStokes, tensor=b_S)
for bc in bcsS:
    bc.apply(A_S, b_S)

# Solve Stokes (direct LU)
wS = Function(W)
solve(A_S, wS.vector(), b_S)
(uS_h, pS_h) = wS.split(deepcopy=True)

# ------------------- Darcy (scalar pressure) -------------------
Vd = FunctionSpace(mesh, "Lagrange", 1)
(pD, qD) = TrialFunction(Vd), TestFunction(Vd)

aD = (k/mu)*dot(grad(pD), grad(qD))*dx(2)
LD = Constant(0.0)*qD*dx(2)

# Dirichlet BCs for Darcy pressure on Darcy boundaries (including interface)
bcpD_bottom = DirichletBC(Vd, pD_exact, boundaries, 2)
bcpD_left   = DirichletBC(Vd, pD_exact, boundaries, 3)
bcpD_right  = DirichletBC(Vd, pD_exact, boundaries, 4)
bcpD_int    = DirichletBC(Vd, pD_exact, boundaries, 5)
bcsD = [bcpD_bottom, bcpD_left, bcpD_right, bcpD_int]

# Assemble Darcy system with diagonal preallocation
A_D = PETScMatrix()
b_D = PETScVector()
assemble(aD, tensor=A_D, keep_diagonal=True)
assemble(LD, tensor=b_D)
for bc in bcsD:
    bc.apply(A_D, b_D)

# Solve Darcy (direct LU)
pD_h = Function(Vd)
solve(A_D, pD_h.vector(), b_D)

# ------------------- Output -------------------
xdmf_u = XDMFFile(mesh.mpi_comm(), "stokes_velocity.xdmf")
xdmf_p = XDMFFile(mesh.mpi_comm(), "darcy_pressure.xdmf")
xdmf_u.write(uS_h)
xdmf_p.write(pD_h)

print("Stokes velocity and Darcy pressure have been saved to XDMF files.")