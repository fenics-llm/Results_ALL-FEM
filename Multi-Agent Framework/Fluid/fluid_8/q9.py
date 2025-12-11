# -*- coding: utf-8 -*-
#
# 2‑D steady incompressible Navier–Stokes in a periodic channel
# ---------------------------------------------------------------
# Domain: Ω = [0,1] × [0,0.20]   (m)
# Mesh: 128 × 32 uniform quadrilaterals
# BCs: periodic in x, no‑slip on y‑walls, pressure gauge at (0,0)
# Parameters: ρ = 1.0, μ = 0.01, G = 1.0 (body force (G,0))
# Output: velocity u and pressure p to q9_soln.xdmf
#
# ---------------------------------------------------------------
# NOTE: legacy FEniCS (dolfin) – no dolfinx calls
# ---------------------------------------------------------------

from dolfin import *
import numpy as np

# ----------------------------------------------------------------------
# 1. Mesh and periodic boundary definition
# ----------------------------------------------------------------------
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 0.20), 128, 32, "crossed")

class PeriodicBoundaryX(SubDomain):
    # Map left (x=0) to right (x=1) for all y
    def inside(self, x, on_boundary):
        return bool(on_boundary and near(x[0], 0.0))
    def map(self, x, y):
        y[0] = x[0] + 1.0
        y[1] = x[1]

pbc = PeriodicBoundaryX()

# ----------------------------------------------------------------------
# 2. Mixed Taylor–Hood space (P2/P1) with periodicity
# ----------------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH, constrained_domain=pbc)

# ----------------------------------------------------------------------
# 3. Boundary conditions
# ----------------------------------------------------------------------
# No‑slip on top and bottom walls (y = 0 and y = 0.20)
noslip = Constant((0.0, 0.0))
bc_bottom = DirichletBC(W.sub(0), noslip, "near(x[1], 0.0)")
bc_top    = DirichletBC(W.sub(0), noslip, "near(x[1], 0.20)")

# Pressure gauge at (0,0) – pointwise Dirichlet on pressure component
class PressureGauge(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0) and near(x[1], 0.0)

pg = PressureGauge()
bc_pressure = DirichletBC(W.sub(1), Constant(0.0), pg, method="pointwise")

bcs = [bc_bottom, bc_top, bc_pressure]

# ----------------------------------------------------------------------
# 4. Define trial/test functions, parameters and body force
# ----------------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

rho = Constant(1.0)          # density
mu  = Constant(0.01)         # dynamic viscosity
G   = Constant(1.0)          # body force magnitude
f   = Constant((G, 0.0))     # body force vector

# ----------------------------------------------------------------------
# 5. Weak formulation (Picard linearisation)
# ----------------------------------------------------------------------
# Start with an initial guess for the velocity
w   = Function(W)               # (u,p) at current iteration
(u_k, p_k) = w.split(deepcopy=True)

# Symmetric gradient
def epsilon(v):
    return sym(grad(v))

# Momentum + continuity (Picard)
a = ( rho*dot(dot(u_k, nabla_grad(u)), v) * dx
      + 2.0*mu*inner(epsilon(u), epsilon(v))*dx
      - p*div(v)*dx
      + q*div(u)*dx )
L = dot(f, v)*dx

# ----------------------------------------------------------------------
# 6. Solver loop
# ----------------------------------------------------------------------
tol   = 1e-8
maxit = 30
for it in range(maxit):
    solve(a == L, w, bcs, solver_parameters={"linear_solver":"mumps"})
    (u_new, p_new) = w.split(deepcopy=True)

    # Compute residual norm (velocity change)
    du = Function(W.sub(0).collapse())
    du.vector()[:] = u_new.vector() - u_k.vector()
    res = norm(du, "l2")
    print("Picard iteration {}: ||Δu|| = {:.3e}".format(it+1, res))
    if res < tol:
        break
    u_k.assign(u_new)

# ----------------------------------------------------------------------
# 7. Output to XDMF
# ----------------------------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "q9_soln.xdmf") as xdmf:
    xdmf.write(u_new, 0.0)
    xdmf.write(p_new, 0.0)