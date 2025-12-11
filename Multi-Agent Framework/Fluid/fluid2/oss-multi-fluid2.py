# -*- coding: utf-8 -*-
#
# 2-D steady Stokes flow in a rectangular channel (legacy FEniCS)
#
#  Domain: Ω = (0, L) × (0, H)  with L = 2.0 m, H = 0.20 m
#  Mesh: 120 × 12 structured quadrilaterals (crossed)
#  Viscosity μ = 1.0 Pa·s, body force f = (1.0, 0.0) N·m⁻³
#  BCs:  u = 0 on walls (y = 0, y = H); traction-free on inlet/outlet
#  Output: speed field PNG and XDMF file with (u, p)
#
# --------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')                     # headless backend
import matplotlib.pyplot as plt
from dolfin import *

# --------------------------------------------------------------
# 1. Geometry and mesh
L, H = 2.0, 0.20
Nx, Ny = 120, 12
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), Nx, Ny, "crossed")

# --------------------------------------------------------------
# 2. Function spaces (Taylor–Hood)
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # P2 velocity
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # P1 pressure
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# --------------------------------------------------------------
# 3. Boundary conditions (no-slip on walls)
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], H))

walls = Walls()
bc_u = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)   # only velocity part
bc_p = DirichletBC(W.sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)", method="pointwise")

# --------------------------------------------------------------
# 4. Variational formulation
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

mu = Constant(1.0)                     # dynamic viscosity
f  = Constant((1.0, 0.0))              # body force

def epsilon(v):
    return sym(grad(v))

a = (2*mu*inner(epsilon(u), epsilon(v))*dx
     - div(v)*p*dx
     + q*div(u)*dx)

Lform = inner(f, v)*dx

# --------------------------------------------------------------
# 5. Pressure null-space handling
null_vec = Vector(mesh.mpi_comm(), W.dim())
W.sub(0).dofmap().set(null_vec, 0.0)          # velocity part zero
W.sub(1).dofmap().set(null_vec, 1.0)          # pressure part constant 1
null_vec *= 1.0/null_vec.norm("l2")
null_space = VectorSpaceBasis([null_vec])

# --------------------------------------------------------------
# 6. Assemble and solve the saddle-point system
A = PETScMatrix()
assemble(a, tensor=A)          # assemble bilinear form into PETSc matrix
b = assemble(Lform)           # RHS vector

# Apply Dirichlet BC on walls
bc_u.apply(A, b)
bc_p.apply(A, b)

# Attach pressure null-space to the PETSc matrix
A.set_nullspace(null_space)
null_space.orthogonalize(b)

# MINRES solver that respects the null-space
w = Function(W)                # (u, p) solution
solve(A, w.vector(), b, "mumps")

# --------------------------------------------------------------
# 7. Split solution
(u_sol, p_sol) = w.split(deepcopy=True)

# --------------------------------------------------------------
# 8. Post-processing: speed magnitude
speed = sqrt(dot(u_sol, u_sol))
V_cg = FunctionSpace(mesh, "CG", 2)    # smoother space for plotting
speed_cg = project(speed, V_cg)

# --------------------------------------------------------------
# 9. Save PNG of speed
plt.figure(figsize=(8, 2))
p = plot(speed_cg, title=r"Speed $|\mathbf{u}|$ (m·s$^{-1}$)",
         cmap="viridis")
plt.colorbar(p)
plt.xlabel(r"$x$ (m)")
plt.ylabel(r"$y$ (m)")
plt.tight_layout()
plt.savefig("q2_speed.png", dpi=300)

# --------------------------------------------------------------
# 10. Save solution to XDMF (velocity + pressure)
with XDMFFile(mesh.mpi_comm(), "q2_solution.xdmf") as xdmf:
    xdmf.write(u_sol, 0.0)
    xdmf.write(p_sol, 0.0)

# --------------------------------------------------------------
# End of script
