# -*- coding: utf-8 -*-
"""
Lid‑driven cavity (steady incompressible Navier–Stokes) – legacy FEniCS
Mesh: 128 × 128 structured unit square
Taylor–Hood (P2/P1) elements
ρ = 1.0, μ = 0.01
Pressure gauge: p(0,0) = 0
Outputs:
    - q5_speed.png   (|u| colormap)
    - q5_soln.xdmf   (velocity + pressure)
"""

import matplotlib
matplotlib.use('Agg')                     # headless backend
import matplotlib.pyplot as plt

from dolfin import (Mesh, RectangleMesh, Point,
                    FunctionSpace, MixedElement,
                    VectorElement, FiniteElement,
                    DirichletBC, SubDomain,
                    Constant, Function, TestFunctions,
                    split, grad, div, dot,
                    inner, sym, dx, solve,
                    XDMFFile, plot, MPI, near,
                    DOLFIN_EPS, sqrt)

# ----------------------------------------------------------------------
# 1. Mesh
# ----------------------------------------------------------------------
N = 128
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), N, N, "crossed")

# ----------------------------------------------------------------------
# 2. Mixed Taylor–Hood space (P2 velocity, P1 pressure)
# ----------------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # velocity
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# ----------------------------------------------------------------------
# 3. Boundary conditions
# ----------------------------------------------------------------------
class Lid(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0, DOLFIN_EPS)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0, DOLFIN_EPS) or
                                near(x[0], 0.0, DOLFIN_EPS) or
                                near(x[0], 1.0, DOLFIN_EPS))

lid   = Lid()
walls = Walls()

bcu_lid   = DirichletBC(W.sub(0), Constant((1.0, 0.0)), lid)
bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)

class Point00(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, DOLFIN_EPS) and near(x[1], 0.0, DOLFIN_EPS)

p00 = Point00()
bcp = DirichletBC(W.sub(1), Constant(0.0), p00, method="pointwise")

bcs = [bcu_lid, bcu_walls, bcp]

# ----------------------------------------------------------------------
# 4. Variational formulation (steady Navier–Stokes)
# ----------------------------------------------------------------------
rho = Constant(1.0)
mu  = Constant(0.01)

w = Function(W)                # unknown (u,p) – initial guess = 0
(u, p) = split(w)            # trial functions
(v, q) = TestFunctions(W)    # test functions

F_conv = rho * inner(dot(u, grad(u)), v) * dx
F_visc = 2 * mu * inner(sym(grad(u)), sym(grad(v))) * dx
F_pres = - p * div(v) * dx + q * div(u) * dx
F = F_conv + F_visc + F_pres

# ----------------------------------------------------------------------
# 5. Nonlinear solve (Newton)
# ----------------------------------------------------------------------
solve(F == 0, w, bcs,
      solver_parameters={"newton_solver":
                         {"relative_tolerance": 1e-8,
                          "absolute_tolerance": 1e-10,
                          "maximum_iterations": 30,
                          "linear_solver": "mumps"}})

# ----------------------------------------------------------------------
# 6. Extract components
# ----------------------------------------------------------------------
u_h, p_h = w.split(deepcopy=True)

# ----------------------------------------------------------------------
# 7. Post‑processing: speed magnitude and PNG output
# ----------------------------------------------------------------------
speed = sqrt(dot(u_h, u_h))
plt.figure(figsize=(6,5))
c = plot(speed, title=r"$|\mathbf{u}|$", cmap="viridis")
plt.colorbar(c)
plt.tight_layout()
plt.savefig("q5_speed.png", dpi=300)

# ----------------------------------------------------------------------
# 8. XDMF export
# ----------------------------------------------------------------------
with XDMFFile(MPI.comm_world, "q5_soln.xdmf") as xdmf:
    xdmf.write(u_h, 0.0)
    xdmf.write(p_h, 0.0)

print("Computation finished: q5_speed.png and q5_soln.xdmf written.")