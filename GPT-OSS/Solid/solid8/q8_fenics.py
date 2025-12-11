# filename: q8_fenics.py
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Geometry and mesh
Lx, Ly = 1.0, 0.20
nx, ny = 50, 25
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)

# -------------------------------------------------
# Function spaces
V = VectorFunctionSpace(mesh, "Lagrange", degree=1)          # displacement
TensorFS = TensorFunctionSpace(mesh, "Lagrange", degree=1)   # stress (tensor)

# -------------------------------------------------
# Material data (orthotropic, plane‑stress)
E1   = 40e9          # Pa
E2   = 10e9          # Pa
G12  = 5e9           # Pa
nu12 = 0.25
nu21 = nu12 * E2 / E1   # reciprocity

# Local reduced stiffness matrix Q (Voigt 3×3)
Q11 = E1 / (1.0 - nu12 * nu21)
Q22 = E2 / (1.0 - nu12 * nu21)
Q12 = nu12 * E2 / (1.0 - nu12 * nu21)
Q66 = G12
Q_local = np.array([[Q11, Q12, 0.0],
                    [Q12, Q22, 0.0],
                    [0.0, 0.0, Q66]])

# -------------------------------------------------
# Rotation to global axes (theta = 30° anticlockwise)
theta = np.radians(30.0)
c = np.cos(theta)
s = np.sin(theta)

# Transformation matrix for plane‑stress Voigt notation
Tmat = np.array([[c**2,      s**2,      2*c*s],
                [s**2,      c**2,     -2*c*s],
                [-c*s,      c*s,   c**2 - s**2]])

# Global reduced stiffness matrix
Q_global = Tmat @ Q_local @ Tmat.T

# Convert to a UFL Constant (3×3)
C = Constant(Q_global)

# -------------------------------------------------
# Strain and stress in Voigt form
def epsilon(u):
    return sym(grad(u))

def epsilon_voigt(u):
    eps = epsilon(u)
    return as_vector([eps[0, 0],
                      eps[1, 1],
                      2*eps[0, 1]])   # engineering shear strain

def sigma_voigt(u):
    eps_v = epsilon_voigt(u)
    return dot(C, eps_v)

def sigma_tensor(u):
    sv = sigma_voigt(u)
    return as_tensor([[sv[0], sv[2]],
                      [sv[2], sv[1]]])

# -------------------------------------------------
# Boundary definitions
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], Ly) and on_boundary

bottom = Bottom()
top    = Top()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
bottom.mark(boundaries, 1)
top.mark(boundaries, 2)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Variational problem
u = TrialFunction(V)
v = TestFunction(V)

a = inner(sigma_tensor(u), epsilon(v)) * dx

# Traction on the top edge: (0, -10 MPa)
t = Constant((0.0, -10e6))          # Pa
L = dot(t, v) * ds(2)

# Dirichlet BC on bottom edge (fixed)
zero = Constant((0.0, 0.0))
bc = DirichletBC(V, zero, bottom)

# Solve
u_sol = Function(V, name="Displacement")
solve(a == L, u_sol, bc, solver_parameters={"linear_solver": "mumps"})

# -------------------------------------------------
# Post‑processing: stress and von Mises
sigma = project(sigma_tensor(u_sol), TensorFS, solver_type='cg')
sigma.rename("Stress", "sigma")

# von Mises stress (plane stress)
sxx = sigma[0, 0]
syy = sigma[1, 1]
sxy = sigma[0, 1]
vm_expr = sqrt(sxx**2 - sxx*syy + syy**2 + 3*sxy**2)
V0 = FunctionSpace(mesh, "Lagrange", 1)
vm = project(vm_expr, V0, solver_type='cg')
vm.rename("Von_Mises", "von_mises")

# -------------------------------------------------
# Save results to XDMF
with XDMFFile(mesh.mpi_comm(), "q8_solution.xdmf") as xdmf:
    xdmf.write(u_sol, 0.0)
    xdmf.write(sigma, 0.0)
    xdmf.write(vm, 0.0)

# -------------------------------------------------
# Plot horizontal displacement ux
plt.figure()
p = plot(u_sol.sub(0), title=r"Horizontal displacement $u_x$", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q8_ux.png", dpi=300)
plt.clf()

# Plot von Mises stress
plt.figure()
p = plot(vm, title="Von Mises stress (Pa)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q8_vm.png", dpi=300)
plt.clf()