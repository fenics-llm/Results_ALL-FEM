# filename: linear_elasticity_plate.py
from dolfin import *
import matplotlib
matplotlib.use('Agg')  # for headless environments
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Geometry and mesh
Lx, Ly = 1.0, 0.20
nx, ny = 100, 20
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny, "crossed")

# ------------------------------------------------------------
# Material parameters (plane stress)
nu = 0.30

# Young's modulus varies linearly with height y
E_expr = Expression('100e9 + 100e9*x[1]/0.20', degree=1)  # Pa

# Lame parameters as Expressions (plane stress)
lambda_expr = (E_expr*nu)/((1.0+nu)*(1.0-nu))
mu_expr     = E_expr/(2.0*(1.0+nu))

# ------------------------------------------------------------
# Function space (Taylor–Hood not needed – pure elasticity)
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# ------------------------------------------------------------
# Boundary definitions
tol = 1E-10

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, tol)

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
LeftBoundary().mark(boundaries, 1)
RightBoundary().mark(boundaries, 2)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# ------------------------------------------------------------
# Dirichlet BC on left edge (u = 0)
zero = Constant((0.0, 0.0))
bc_left = DirichletBC(V, zero, boundaries, 1)

# ------------------------------------------------------------
# Traction on right edge
t_right = Constant((2.0e6, 0.0))  # N/m (plane stress, per unit thickness)

# ------------------------------------------------------------
# Variational problem
u = TrialFunction(V)
v = TestFunction(V)

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lambda_expr*tr(epsilon(u))*Identity(2) + 2.0*mu_expr*epsilon(u)

a = inner(sigma(u), epsilon(v))*dx
L = dot(t_right, v)*ds(2)   # only right edge

# ------------------------------------------------------------
# Solve
u_sol = Function(V, name="Displacement")
solve(a == L, u_sol, bc_left,
      solver_parameters={"linear_solver": "mumps"})

# ------------------------------------------------------------
# Post‑processing: displacement magnitude
W = FunctionSpace(mesh, "Lagrange", 2)
u_magnitude = project(sqrt(dot(u_sol, u_sol)), W)

# Save magnitude plot
plt.figure(figsize=(8, 3))
p = plot(u_magnitude, title=r"$|u|$ (m)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q9_disp.png", dpi=300)

# ------------------------------------------------------------
# Save displacement field in XDMF format
with XDMFFile(mesh.mpi_comm(), "q9_disp.xdmf") as xdmf:
    xdmf.write(u_sol)

print("Simulation completed.")
print("  - Displacement magnitude saved as q9_disp.png")
print("  - Full displacement field saved as q9_disp.xdmf")