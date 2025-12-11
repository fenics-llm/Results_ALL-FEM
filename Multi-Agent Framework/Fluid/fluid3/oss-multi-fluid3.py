# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dolfin import *

# ------------------------------------------------------------
# 1. Mesh and geometry
# ------------------------------------------------------------
nx, ny = 96, 96
mesh = UnitSquareMesh(nx, ny)

# ------------------------------------------------------------
# 2. Taylor–Hood mixed finite element (P2 velocity, P1 pressure)
# ------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# ------------------------------------------------------------
# 3. Boundary conditions
# ------------------------------------------------------------
tol = 1E-14

class Lid(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0, tol)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0, tol) or near(x[0], 0.0, tol) or near(x[0], 1.0, tol))

lid   = Lid()
walls = Walls()

# Velocity BCs
u_lid  = Constant((1.0, 0.0))
u_wall = Constant((0.0, 0.0))
bc_u_lid   = DirichletBC(W.sub(0), u_lid, lid)
bc_u_walls = DirichletBC(W.sub(0), u_wall, walls)

# Pressure pinning at a single point (0,0)
class PinPoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, tol) and near(x[1], 0.0, tol)

pin = PinPoint()
bc_p = DirichletBC(W.sub(1), Constant(0.0), pin, method='pointwise')

bcs = [bc_u_lid, bc_u_walls, bc_p]

# ------------------------------------------------------------
# 4. Variational problem (steady Stokes)
# ------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

mu = Constant(1.0)   # dynamic viscosity

a = mu*inner(grad(u), grad(v))*dx - div(v)*p*dx + q*div(u)*dx
L = Constant(0.0)*v[0]*dx   # zero body force

# ------------------------------------------------------------
# 5. Assemble and solve
# ------------------------------------------------------------
w = Function(W)
solve(a == L, w, bcs, solver_parameters={"linear_solver": "mumps"})

# ------------------------------------------------------------
# 6. Split solution
# ------------------------------------------------------------
(u_h, p_h) = w.split(deepcopy=True)

# ------------------------------------------------------------
# 7. Post‑processing: speed field and PNG output
# ------------------------------------------------------------
speed = sqrt(dot(u_h, u_h))
V_cg = FunctionSpace(mesh, "CG", 1)
speed_cg = project(speed, V_cg)

plt.figure(figsize=(6,5))
p = plot(speed_cg, title="Speed |u|", cmap="viridis")
plt.colorbar(p)
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.savefig("q3_speed.png", dpi=300)

# ------------------------------------------------------------
# 8. XDMF export of velocity and pressure
# ------------------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "q3_soln.xdmf") as xdmf:
    xdmf.write(u_h, 0.0)
    xdmf.write(p_h, 0.0)