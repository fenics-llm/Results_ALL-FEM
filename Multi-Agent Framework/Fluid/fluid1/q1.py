# -*- coding: utf-8 -*-
#
# 2‑D steady Stokes flow in a rectangular channel (legacy FEniCS)
#
#  Domain: Ω = (0, L) × (0, H) with L = 2.0 m, H = 0.20 m
#  Mesh: 100 × 10 structured cells
#  Viscosity: μ = 1.0 Pa·s
#  Inlet traction: σ·n = -p_in n,   p_in = 1.0 Pa
#  Outlet traction: σ·n = -p_out n, p_out = 0.0 Pa
#  Walls: no‑slip (u = 0)
#  Pressure gauge: p = 0 at (0,0)
#
#  Output:
#    - speed |u| → q1_speed.png
#    - velocity & pressure → q1_soln.xdmf
#
#  Note: legacy dolfin (FEniCS 1.6/1.7) API is used.
#

from dolfin import *
import matplotlib
matplotlib.use('Agg')          # headless backend
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# 1. Geometry and mesh
# ----------------------------------------------------------------------
L, H = 2.0, 0.20
Nx, Ny = 100, 10
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), Nx, Ny, "crossed")

# ----------------------------------------------------------------------
# 2. Boundary markers
# ----------------------------------------------------------------------
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], H))

Inlet().mark(boundaries, 1)
Outlet().mark(boundaries, 2)
Walls().mark(boundaries, 3)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# ----------------------------------------------------------------------
# 3. Mixed Taylor–Hood space (P2/P1)
# ----------------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# ----------------------------------------------------------------------
# 4. Trial / test functions
# ----------------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# ----------------------------------------------------------------------
# 5. Material parameters
# ----------------------------------------------------------------------
mu = Constant(1.0)          # dynamic viscosity

# ----------------------------------------------------------------------
# 6. Symmetric gradient
# ----------------------------------------------------------------------
def epsilon(w):
    return sym(grad(w))

# ----------------------------------------------------------------------
# 7. Variational forms (sign corrected for inlet traction)
# ----------------------------------------------------------------------
a = (2.0*mu*inner(epsilon(u), epsilon(v)) - div(v)*p - q*div(u))*dx

p_in  = Constant(1.0)       # inlet pressure
p_out = Constant(0.0)       # outlet pressure (zero)

n = FacetNormal(mesh)
# NOTE: minus sign because σ·n = -p_in n is prescribed
Lrhs = (-p_in*dot(n, v))*ds(1) + (-p_out*dot(n, v))*ds(2)

# ----------------------------------------------------------------------
# 8. Boundary conditions
# ----------------------------------------------------------------------
# No‑slip on walls (velocity = 0)
bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 3)

# Pressure gauge at (0,0) using a SubDomain for pointwise BC
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)

pressure_pt = PressurePoint()
bc_pressure = DirichletBC(W.sub(1), Constant(0.0), pressure_pt, "pointwise")

bcs = [bc_walls, bc_pressure]

# ----------------------------------------------------------------------
# 9. Assemble and solve
# ----------------------------------------------------------------------
A, b = assemble_system(a, Lrhs, bcs)

w = Function(W)                     # solution (u,p)
solve(A, w.vector(), b, "lu")

# ----------------------------------------------------------------------
# 10. Split solution
# ----------------------------------------------------------------------
u_h, p_h = w.split(deepcopy=True)

# ----------------------------------------------------------------------
# 11. Post‑processing: speed and output files
# ----------------------------------------------------------------------
# Speed magnitude
speed = project(sqrt(dot(u_h, u_h)), FunctionSpace(mesh, "P", 2))

# Save speed plot
plt.figure(figsize=(8, 2))
c = plot(speed, title=r"$|\mathbf{u}|$", cmap="viridis")
plt.colorbar(c)
plt.axis('off')
plt.tight_layout()
plt.savefig("q1_speed.png", dpi=300)

# Save velocity and pressure to XDMF
with XDMFFile(mesh.mpi_comm(), "q1_soln.xdmf") as xdmf:
    xdmf.write(u_h, 0.0)
    xdmf.write(p_h, 0.0)

# ----------------------------------------------------------------------
# 12. Optional sanity check: outlet flow rate vs analytical Poiseuille
# ----------------------------------------------------------------------
# Convert symbolic Constants to floats for the analytical expression
mu_val   = float(mu)
p_in_val = float(p_in)
p_out_val= float(p_out)

Q_exact = (H**3) / (12.0 * mu_val * L) * (p_in_val - p_out_val)   # analytical flow per unit depth
U_out   = assemble(u_h[0] * ds(2))                                 # numerical flow through outlet

print("Outlet flow rate (numerical) = {:.6e}".format(U_out))
print("Outlet flow rate (exact)    = {:.6e}".format(Q_exact))

print("Computation finished: q1_speed.png and q1_soln.xdmf written.")