# -*- coding: utf-8 -*-
#
#  Plane‑stress linear elasticity on a rectangular plate
#  (legacy FEniCS / dolfin)
#
from dolfin import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Mesh and subdomains
# ----------------------------------------------------------------------
Lx, Ly = 1.0, 0.20
Nx, Ny = 100, 20
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny, "crossed")

# Mark boundaries: 0=left, 1=right, 2=bottom, 3=top
class Left(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0)
class Right(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], Lx)
class Bottom(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.0)
class Top(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], Ly)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
Left().mark(boundaries, 0)
Right().mark(boundaries, 1)
Bottom().mark(boundaries, 2)
Top().mark(boundaries, 3)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# ----------------------------------------------------------------------
# 2. Function space (vector quadratic)
# ----------------------------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# ----------------------------------------------------------------------
# 3. Material parameters (plane stress, height‑dependent E)
# ----------------------------------------------------------------------
nu = 0.30
E_expr = Expression("1e11*(1 + 5*x[1])", degree=1)   # Pa, varies linearly with y

# Lame coefficients as Expressions
lam_expr = E_expr * nu / (1.0 - nu**2)                # λ(y)
mu_expr  = E_expr / (2.0 * (1.0 + nu))               # μ(y)

# ----------------------------------------------------------------------
# 4. Variational formulation
# ----------------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

def eps(w):
    return sym(grad(w))

def sigma(w):
    eps_w = eps(w)
    return 2.0 * mu_expr * eps_w + lam_expr * tr(eps_w) * Identity(2)

a = inner(sigma(u), eps(v)) * dx
t = Constant((2e6, 0.0))               # traction on the right edge (Pa·m)
L = dot(t, v) * ds(1)                  # only right side (marker 1)

# ----------------------------------------------------------------------
# 5. Boundary conditions
# ----------------------------------------------------------------------
bc = DirichletBC(V, Constant((0.0, 0.0)), boundaries, 0)   # left edge fixed

# ----------------------------------------------------------------------
# 6. Solve
# ----------------------------------------------------------------------
u_sol = Function(V, name="Displacement")
solve(a == L, u_sol, bc, solver_parameters={"linear_solver": "mumps"})

# ----------------------------------------------------------------------
# 7. Post‑processing: magnitude and output
# ----------------------------------------------------------------------
# 7.1. XDMF export
with XDMFFile(mesh.mpi_comm(), "q9_disp.xdmf") as xdmf:
    xdmf.write(u_sol)

# 7.2. PNG of |u|
V_mag = FunctionSpace(mesh, "Lagrange", 2)
u_mag = project(sqrt(dot(u_sol, u_sol)), V_mag)

plt.figure(figsize=(6, 2.5))
p = plot(u_mag, title=r"$|\mathbf{u}|$ (m)", cmap="viridis")
plt.colorbar(p)
plt.xlabel(r"$x$ (m)")
plt.ylabel(r"$y$ (m)")
plt.tight_layout()
plt.savefig("q9_disp.png", dpi=300)
plt.close()