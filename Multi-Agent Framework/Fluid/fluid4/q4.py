# q4_navier_stokes.py
# Legacy FEniCS (dolfin) implementation of steady 2D channel flow

import matplotlib
matplotlib.use('Agg')          # headless backend for PNG output
import matplotlib.pyplot as plt

from dolfin import *
import numpy as np

# ----------------------------------------------------------------------
# 1. Geometry and mesh
# ----------------------------------------------------------------------
L = 2.0
H = 0.20
Nx = 160
Ny = 16
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), Nx, Ny, "crossed")

# ----------------------------------------------------------------------
# 2. Function spaces (Taylor–Hood)
# ----------------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # velocity (P2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure (P1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# ----------------------------------------------------------------------
# 3. Boundary conditions
# ----------------------------------------------------------------------
Umean = 2.5
mu    = 0.01
rho   = 1.0

# Inlet velocity profile (parabolic)
inlet_expr = Expression(("6.0*Umean*x[1]/H*(1.0 - x[1]/H)", "0.0"),
                        Umean=Umean, H=H, degree=2)

# No-slip walls (top & bottom)
noslip = Constant((0.0, 0.0))

# Define subdomains for Dirichlet BCs
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], H))

inlet = Inlet()
walls  = Walls()

# Velocity BCs (inlet + walls)
bcu_inlet = DirichletBC(W.sub(0), inlet_expr, inlet)
bcu_walls = DirichletBC(W.sub(0), noslip, walls)

# Pressure pointwise constraint to fix nullspace (optional but safe)
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)

p0 = PressurePoint()
bc_p = DirichletBC(W.sub(1), Constant(0.0), p0, method="pointwise")

bcs = [bcu_inlet, bcu_walls, bc_p]

# ----------------------------------------------------------------------
# 4. Variational formulation (steady Navier–Stokes)
# ----------------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Define symmetric gradient
def epsilon(w):
    return sym(grad(w))

# Convective term (Newton linearisation)
U = Function(W)               # current iterate (u,p)
(u_k, p_k) = split(U)

F = ( rho*dot(dot(u_k, nabla_grad(u_k)), v)
      + 2*mu*inner(epsilon(u_k), epsilon(v))
      - div(v)*p_k
      + q*div(u_k) )*dx

# Jacobian of F
J = derivative(F, U, TrialFunction(W))

# ----------------------------------------------------------------------
# 5. Newton solver
# ----------------------------------------------------------------------
solve(F == 0, U, bcs, J=J,
      solver_parameters={"newton_solver":
                         {"relative_tolerance": 1e-6,
                          "absolute_tolerance": 1e-8,
                          "maximum_iterations": 25,
                          "linear_solver": "mumps"}})

# Split solution
(u_sol, p_sol) = U.split(deepcopy=True)

# ----------------------------------------------------------------------
# 6. Post‑processing
# ----------------------------------------------------------------------
# Save ux colour map
plt.figure(figsize=(8, 2))
p = plot(u_sol[0], title=r"$u_x$", cmap="viridis")
plt.colorbar(p)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.tight_layout()
plt.savefig("q4_ux.png", dpi=300)

# Save full fields to XDMF
with XDMFFile(mesh.mpi_comm(), "q4_soln.xdmf") as xdmf:
    xdmf.write(u_sol, 0.0)
    xdmf.write(p_sol, 0.0)

print("Simulation completed: q4_ux.png and q4_soln.xdmf written.")