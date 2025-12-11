# filename: linear_elasticity_variable_E.py
from dolfin import *
import matplotlib.pyplot as plt

# -------------------------------------------------
# Parameters
# -------------------------------------------------
nu = 0.30                         # Poisson's ratio
traction_value = 2e6              # (Pa·m) uniform traction on the right edge

# -------------------------------------------------
# Geometry & Mesh
# -------------------------------------------------
Lx, Ly = 1.0, 0.20                # plate dimensions (m)
nx, ny = 100, 20                  # number of cells in x and y
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)

# -------------------------------------------------
# Function space (vector P1)
# -------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# -------------------------------------------------
# Variable Young's modulus E(y)
# -------------------------------------------------
E_expr = Expression("E0 + E1*x[1]",
                    E0=100e9,                     # 100 GPa
                    E1=100e9/0.20,                # slope 100 GPa / 0.20 m
                    degree=1)

# Plane‑stress Lamé parameters as UFL expressions (point‑wise)
lam_expr = (E_expr*nu)/((1.0 + nu)*(1.0 - nu))   # λ(y)
mu_expr  = E_expr/(2.0*(1.0 + nu))               # μ(y)

# -------------------------------------------------
# Strain and stress
# -------------------------------------------------
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lam_expr*tr(epsilon(u))*Identity(2) + 2.0*mu_expr*epsilon(u)

# -------------------------------------------------
# Boundary conditions
# -------------------------------------------------
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0)

bc = DirichletBC(V, Constant((0.0, 0.0)), left_boundary)

# -------------------------------------------------
# Neumann traction on the right edge
# -------------------------------------------------
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
right = RightBoundary()
right.mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

t = Constant((traction_value, 0.0))   # traction vector (Pa·m)

# -------------------------------------------------
# Variational problem
# -------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

a = inner(sigma(u), epsilon(v))*dx
L = dot(t, v)*ds(1)

# -------------------------------------------------
# Solve (direct solver to avoid convergence issues)
# -------------------------------------------------
u_sol = Function(V, name="Displacement")
solve(a == L, u_sol, bc,
      solver_parameters={"linear_solver": "mumps"})

# -------------------------------------------------
# Save displacement field (XDMF)
# -------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "displacement.xdmf")
xdmf.write(u_sol)
xdmf.close()

# -------------------------------------------------
# Displacement magnitude |u|
# -------------------------------------------------
V0 = FunctionSpace(mesh, "Lagrange", 1)
u_mag = Function(V0, name="Displacement_magnitude")
u_mag_expr = sqrt(dot(u_sol, u_sol))
u_mag.assign(project(u_mag_expr, V0))

# -------------------------------------------------
# Plot and save as PNG
# -------------------------------------------------
plt.figure(figsize=(8, 3))
p = plot(u_mag, cmap="viridis")
plt.colorbar(p, label="|u| (m)")
plt.title("Displacement magnitude")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q9_disp.png", dpi=300)
plt.close()