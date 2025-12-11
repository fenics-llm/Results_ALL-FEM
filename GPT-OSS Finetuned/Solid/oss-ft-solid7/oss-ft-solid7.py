# filename: elasticity_two_materials.py
from dolfin import *
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Mesh
# -------------------------------------------------
nx, ny = 80, 16
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 0.20), nx, ny, "crossed")

# -------------------------------------------------
# 2. Material parameters (plane stress)
# -------------------------------------------------
E_Al, E_Steel = 70e9, 200e9
nu = 0.30
mu_Al = E_Al/(2.0*(1.0+nu))
mu_Steel = E_Steel/(2.0*(1.0+nu))
lmbda_Al = E_Al*nu/((1.0+nu)*(1.0-nu))
lmbda_Steel = E_Steel*nu/((1.0+nu)*(1.0-nu))

# Piecewise material coefficients (DG0)
mu = Expression("x[1] > 0.10 ? mu_Al : mu_Steel",
                mu_Al=mu_Al, mu_Steel=mu_Steel, degree=0)
lmbda = Expression("x[1] > 0.10 ? lmbda_Al : lmbda_Steel",
                    lmbda_Al=lmbda_Al, lmbda_Steel=lmbda_Steel, degree=0)

# -------------------------------------------------
# 3. Function space (vector Lagrange degree 2)
# -------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# -------------------------------------------------
# 4. Boundary conditions
# -------------------------------------------------
tol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0, tol)
bc = DirichletBC(V, Constant((0.0, 0.0)), left_boundary)

# -------------------------------------------------
# 5. Traction on right edge
# -------------------------------------------------
class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0, tol)
right_boundary = RightBoundary()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
right_boundary.mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
T = Constant((0.0, -5000.0))   # N/m (plane stress)

# -------------------------------------------------
# 6. Variational problem (plane stress)
# -------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

def sigma(u, mu, lmbda):
    eps = sym(grad(u))
    return lmbda*tr(eps)*Identity(2) + 2.0*mu*eps

a = inner(sigma(u, mu, lmbda), sym(grad(v)))*dx
L = dot(T, v)*ds(1)

# -------------------------------------------------
# 7. Solve
# -------------------------------------------------
u_sol = Function(V)
solve(a == L, u_sol, bc, solver_parameters={"linear_solver":"mumps"})

# -------------------------------------------------
# 8. Plot displacement magnitude
# -------------------------------------------------
V0 = FunctionSpace(mesh, "Lagrange", 2)
u_mag = project(sqrt(dot(u_sol, u_sol)), V0)

plt.figure(figsize=(8, 3))
p = plot(u_mag, title="Displacement magnitude |u| (m)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q7_disp.png", dpi=300)
plt.close()

# -------------------------------------------------
# 9. Save displacement field in XDMF format
# -------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "q7_displacement.xdmf") as xdmf:
    xdmf.write(u_sol)

print("Finished: displacement saved to q7_displacement.xdmf and plot to q7_disp.png")