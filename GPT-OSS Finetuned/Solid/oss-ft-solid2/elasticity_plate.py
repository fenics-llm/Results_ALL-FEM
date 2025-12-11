# filename: elasticity_plate.py
from dolfin import *
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Mesh and geometry
# ------------------------------------------------------------
Lx, Ly = 1.0, 0.20
nx, ny = 40, 8
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny, "crossed")

# ------------------------------------------------------------
# Material parameters (plane stress)
# ------------------------------------------------------------
E  = 200e9          # Pa
nu = 0.30
mu = E/(2.0*(1.0+nu))
lmbda = (E*nu)/((1.0+nu)*(1.0-nu))

# ------------------------------------------------------------
# Function space (quadratic vector field)
# ------------------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# ------------------------------------------------------------
# Boundary definitions
# ------------------------------------------------------------
tol = 1e-8
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], Ly, tol)

left_boundary = LeftBoundary()
top_boundary   = TopBoundary()

# Mark boundaries for ds integration
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
left_boundary.mark(boundaries, 1)
top_boundary.mark(boundaries, 2)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# ------------------------------------------------------------
# Dirichlet condition (fixed left edge)
# ------------------------------------------------------------
zero = Constant((0.0, 0.0))
bc_left = DirichletBC(V, zero, left_boundary)

# ------------------------------------------------------------
# Traction on the top edge
# ------------------------------------------------------------
t_top = Constant((0.0, -2000.0))   # N/m (downward)

# ------------------------------------------------------------
# Variational problem (plane stress elasticity)
# ------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda*tr(epsilon(u))*Identity(2) + 2.0*mu*epsilon(u)

a = inner(sigma(u), epsilon(v))*dx
L = dot(t_top, v)*ds(2)   # only top edge gets traction

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
u_sol = Function(V)
solve(a == L, u_sol, bc_left, solver_parameters={"linear_solver": "mumps"})

# ------------------------------------------------------------
# Output: XDMF
# ------------------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "plate_displacement.xdmf") as xdmf:
    xdmf.write(u_sol)

# ------------------------------------------------------------
# Plot vertical displacement u_y
# ------------------------------------------------------------
uy = u_sol.sub(1)
plt.figure(figsize=(6, 3))
p = plot(uy, title=r"$u_y$ (m)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q2_uy.png", dpi=300)
plt.close()