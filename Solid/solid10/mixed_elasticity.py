# filename: mixed_elasticity.py
from dolfin import *
import matplotlib
matplotlib.use('Agg')          # non‑interactive backend
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Geometry and mesh
Lx, Ly = 1.0, 0.20
nx, ny = 100, 20
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)

# --------------------------------------------------------------
# Material parameters (plane strain, nearly incompressible)
E  = 5.0e6          # Pa  (5 MPa)
nu = 0.49
mu = E / (2.0 * (1.0 + nu))               # shear modulus
K  = E / (3.0 * (1.0 - 2.0 * nu))         # bulk modulus

# --------------------------------------------------------------
# Mixed function space (Taylor–Hood: P2 for u, P1 for p)
V = VectorFunctionSpace(mesh, "Lagrange", 2)   # displacement (P2)
Q = FunctionSpace(mesh, "Lagrange", 1)         # pressure (P1)

mixed_elem = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# --------------------------------------------------------------
# Trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Strain tensor
def epsilon(u):
    return sym(grad(u))

# Bilinear form (mixed displacement–pressure formulation)
a = (2.0 * mu * inner(epsilon(u), epsilon(v))) * dx \
    - p * div(v) * dx \
    - q * div(u) * dx \
    + (1.0 / K) * p * q * dx

# Right‑hand side (zero body forces, zero traction)
L = dot(Constant((0.0, 0.0)), v) * dx + Constant(0.0) * q * dx

# --------------------------------------------------------------
# Boundary conditions
tol = 1e-14

def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0, tol)

def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], Lx, tol)

# Left edge: u = (0,0)
bc_left = DirichletBC(W.sub(0), Constant((0.0, 0.0)), left_boundary)

# Right edge: u_x = 0.03 m, u_y = 0
bc_right = DirichletBC(W.sub(0), Constant((0.03, 0.0)), right_boundary)

bcs = [bc_left, bc_right]

# --------------------------------------------------------------
# Solve the linear system
w = Function(W)   # combined (u, p)
solve(a == L, w, bcs, solver_parameters={"linear_solver": "mumps"})

# Extract displacement and pressure
(u_sol, p_sol) = w.split()

# --------------------------------------------------------------
# Save displacement field (vector) in XDMF format
xdmf_file = XDMFFile(mesh.mpi_comm(), "displacement.xdmf")
xdmf_file.write(u_sol)
xdmf_file.close()

# --------------------------------------------------------------
# Compute displacement magnitude |u|
V_scalar = FunctionSpace(mesh, "Lagrange", 2)
u_magnitude = project(sqrt(dot(u_sol, u_sol)), V_scalar)

# Plot and save as PNG
plt.figure(figsize=(8, 3))
p = plot(u_magnitude, title="Displacement magnitude |u| (m)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q10_disp.png", dpi=300)
plt.close()