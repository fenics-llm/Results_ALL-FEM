# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dolfin import *

# ------------------------------------------------------------
# 1. Mesh
# ------------------------------------------------------------
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 0.20), 100, 20)

# ------------------------------------------------------------
# 2. Material parameters (plane strain)
# ------------------------------------------------------------
E  = 5.0e6          # Pa
nu = 0.49
mu     = E / (2.0 * (1.0 + nu))
lmbda  = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# ------------------------------------------------------------
# 3. Mixed Taylor–Hood space (P2 velocity, P1 pressure)
# ------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# ------------------------------------------------------------
# 4. Boundary definitions
# ------------------------------------------------------------
tol = 1E-8
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0, tol)

left  = LeftBoundary()
right = RightBoundary()

# ------------------------------------------------------------
# 5. Dirichlet conditions for displacement (component 0 of W)
# ------------------------------------------------------------
zero_disp   = Constant((0.0, 0.0))
right_disp  = Constant((0.03, 0.0))

bcs = [DirichletBC(W.sub(0), zero_disp, left),
       DirichletBC(W.sub(0), right_disp, right)]

# ------------------------------------------------------------
# 6. Variational problem
# ------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

def epsilon(v):
    return sym(grad(v))

a = (2.0 * mu) * inner(epsilon(u), epsilon(v)) * dx \
    - p * div(v) * dx \
    + q * div(u) * dx \
    + (1.0 / lmbda) * p * q * dx

L = Constant(0.0) * v[0] * dx   # traction‑free → RHS = 0

# ------------------------------------------------------------
# 7. Solve
# ------------------------------------------------------------
w = Function(W)
solve(a == L, w, bcs, solver_parameters={"linear_solver": "mumps"})

# ------------------------------------------------------------
# 8. Post‑processing
# ------------------------------------------------------------
(u_h, p_h) = w.split(deepcopy=True)

# displacement magnitude
u_magnitude = sqrt(dot(u_h, u_h))

# Plot and save PNG
plt.figure(figsize=(6, 3))
p = plot(u_magnitude, title="Displacement magnitude |u|", cmap="viridis")
plt.colorbar(p)
plt.tight_layout()
plt.savefig("q10_disp.png", dpi=300)
plt.close()

# Write XDMF (mixed solution)
with XDMFFile(mesh.mpi_comm(), "q10_solution.xdmf") as xdmf:
    xdmf.write(w, 0.0)