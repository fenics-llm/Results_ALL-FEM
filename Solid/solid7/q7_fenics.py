# filename: q7_fenics.py
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# Mesh (80 × 16 structured)
# -------------------------------------------------
nx, ny = 80, 16
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 0.20), nx, ny, "crossed")

# -------------------------------------------------
# Subdomains (1 = Al (top), 2 = Steel (bottom))
# -------------------------------------------------
materials = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

class Aluminum(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 0.10 + DOLFIN_EPS

class Steel(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < 0.10 - DOLFIN_EPS

Aluminum().mark(materials, 1)
Steel().mark(materials, 2)

dx = Measure("dx", domain=mesh, subdomain_data=materials)

# -------------------------------------------------
# Function space (quadratic vector Lagrange)
# -------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# -------------------------------------------------
# Boundary conditions (left edge fixed)
# -------------------------------------------------
def left_boundary(x, on_boundary):
    return near(x[0], 0.0) and on_boundary

bc_left = DirichletBC(V, Constant((0.0, 0.0)), left_boundary)
bcs = [bc_left]

# -------------------------------------------------
# Material parameters (plane stress)
# -------------------------------------------------
E_al, E_steel = 70e9, 200e9          # Pa
nu = 0.30

def plane_stress_tensor(E):
    """Return the 3×3 elasticity tensor (Voigt) for plane stress."""
    coeff = E / (1.0 - nu**2)
    C = as_tensor([[coeff,      coeff*nu, 0.0],
                   [coeff*nu,   coeff,    0.0],
                   [0.0,        0.0,      coeff*(1.0 - nu)/2.0]])
    return C

C_al   = plane_stress_tensor(E_al)
C_steel = plane_stress_tensor(E_steel)

# -------------------------------------------------
# Strain / stress helpers
# -------------------------------------------------
def epsilon(u):
    return sym(grad(u))

def sigma(u, C):
    eps = epsilon(u)
    # Voigt vector [ε_xx, ε_yy, 2ε_xy]
    eps_voigt = as_vector([eps[0, 0], eps[1, 1], 2*eps[0, 1]])
    sig_voigt = dot(C, eps_voigt)
    return as_tensor([[sig_voigt[0], sig_voigt[2]],
                      [sig_voigt[2], sig_voigt[1]]])

# -------------------------------------------------
# Variational form (piecewise material)
# -------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

a = (inner(sigma(u, C_al),   epsilon(v))*dx(1) +
     inner(sigma(u, C_steel), epsilon(v))*dx(2))

# -------------------------------------------------
# Neumann traction on the right edge (uniform vertical)
# -------------------------------------------------
traction = Constant((0.0, -5000.0))   # N/m

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and on_boundary

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
RightBoundary().mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

L = dot(traction, v) * ds(1)

# -------------------------------------------------
# Assemble system (keep_diagonal=True fixes Dirichlet rows)
# -------------------------------------------------
A = assemble(a, keep_diagonal=True)
b = assemble(L)
for bc in bcs:
    bc.apply(A, b)

# -------------------------------------------------
# Solve (direct LU/MUMPS solver)
# -------------------------------------------------
u_sol = Function(V, name="Displacement")
solve(A, u_sol.vector(), b, "lu")   # robust direct solver

# -------------------------------------------------
# Post‑processing – displacement magnitude |u|
# -------------------------------------------------
V0 = FunctionSpace(mesh, "Lagrange", 2)   # scalar quadratic space

class MagnitudeExpr(UserExpression):
    def __init__(self, u, **kwargs):
        super().__init__(**kwargs)
        self.u = u
    def eval(self, values, x):
        ux, uy = self.u(x)
        values[0] = np.sqrt(ux*ux + uy*uy)
    def value_shape(self):
        return ()

mag_expr = MagnitudeExpr(u_sol, degree=2)
u_mag = interpolate(mag_expr, V0)

# -------------------------------------------------
# Save results
# -------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "q7_disp.xdmf") as xdmf:
    xdmf.write(u_sol)

plt.figure(figsize=(8, 3))
p = plot(u_mag, title=r"Displacement magnitude $|u|$ (m)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q7_disp.png", dpi=300)
plt.close()

print("Simulation completed.")
print(" - Displacement field saved to q7_disp.xdmf")
print(" - Displacement magnitude plot saved to q7_disp.png")