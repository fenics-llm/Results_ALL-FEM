# -*- coding: utf-8 -*-
#
# Plane‑stress linear elasticity on a rectangular plate (legacy dolfin)
#
from dolfin import *
import matplotlib
matplotlib.use('Agg')          # headless backend
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Geometry and mesh
# ----------------------------------------------------------------------
Lx, Ly = 1.0, 0.20
nx, ny = 40, 8
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny, "crossed")

# ----------------------------------------------------------------------
# 2. Function space (quadratic vector Lagrange)
# ----------------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
V  = FunctionSpace(mesh, Ve)

# ----------------------------------------------------------------------
# 3. Material parameters (plane‑stress)
# ----------------------------------------------------------------------
E  = 200e9          # Pa
nu = 0.30
lmbda = E*nu/(1.0 - nu**2)
mu    = E/(2.0*(1.0 + nu))

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda*tr(epsilon(u))*Identity(2) + 2.0*mu*epsilon(u)

# ----------------------------------------------------------------------
# 4. Boundary definitions
# ----------------------------------------------------------------------
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], Ly, DOLFIN_EPS)

left = LeftBoundary()
top  = TopBoundary()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
left.mark(boundaries, 1)
top.mark(boundaries, 2)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# ----------------------------------------------------------------------
# 5. Dirichlet BC (left edge fixed)
# ----------------------------------------------------------------------
zero = Constant((0.0, 0.0))
bc_left = DirichletBC(V, zero, left)

# ----------------------------------------------------------------------
# 6. Traction on the top edge
# ----------------------------------------------------------------------
t_val = Constant((0.0, -2000.0))   # N/m (downward)

# ----------------------------------------------------------------------
# 7. Variational problem
# ----------------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

a = inner(sigma(u), epsilon(v))*dx
L = dot(t_val, v)*ds(2)   # only top edge contributes

# ----------------------------------------------------------------------
# 8. Solve
# ----------------------------------------------------------------------
u_sol = Function(V)
solve(a == L, u_sol, bc_left, solver_parameters={"linear_solver": "mumps"})

# ----------------------------------------------------------------------
# 9. Post‑processing: vertical displacement u_y
# ----------------------------------------------------------------------
uy = u_sol.sub(1)                     # component 1 (y‑direction)
uy_proj = project(uy, FunctionSpace(mesh, "Lagrange", 2))

# Plot and save PNG
plt.figure(figsize=(6, 2.5))
p = plot(uy_proj, title=r"$u_y$ (m)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q2_uy.png", dpi=300)

# ----------------------------------------------------------------------
# 10. Export displacement field to XDMF
# ----------------------------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "plate_disp.xdmf") as xdmf:
    xdmf.write(u_sol)