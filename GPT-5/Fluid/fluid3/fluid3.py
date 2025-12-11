# q3_stokes_lid_driven_cavity.py
# Legacy FEniCS (dolfin) implementation: steady incompressible Stokes with Taylor–Hood (P2–P1)

from dolfin import *
import matplotlib.pyplot as plt

# --- Geometry & mesh ---
nx = ny = 96
mesh = UnitSquareMesh(nx, ny)

# --- Material/parameters ---
rho = Constant(1.0)         # kg m^-3 (not used in steady Stokes)
mu  = Constant(1.0)         # Pa·s
zero = Constant(0.0)

# --- Finite element spaces: Taylor–Hood P2–P1 ---
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)  # velocity (P2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # pressure (P1)
W  = FunctionSpace(mesh, MixedElement([Ve, Pe]))

# --- Boundary indicators ---
tol = DOLFIN_EPS

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0, tol)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0, tol)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)

# --- Dirichlet boundary conditions on velocity ---
u_lid   = Constant((1.0, 0.0))
u_noslip = Constant((0.0, 0.0))

bcs = []
bcs.append(DirichletBC(W.sub(0), u_lid,   Top()))
bcs.append(DirichletBC(W.sub(0), u_noslip, Left()))
bcs.append(DirichletBC(W.sub(0), u_noslip, Right()))
bcs.append(DirichletBC(W.sub(0), u_noslip, Bottom()))

# --- Fix pressure at one point (reference pressure) for uniqueness ---
#    This pins p(0,0) = 0 without affecting the Stokes solution.
class PPoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, tol) and near(x[1], 0.0, tol)
bcs.append(DirichletBC(W.sub(1), Constant(0.0), PPoint(), method="pointwise"))

# --- Variational formulation for steady Stokes ---
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# a(u,p; v,q) = (2 mu sym(grad u) : sym(grad v)) - (div v) p + q (div u)
def eps(w):
    return sym(grad(w))

a = (2.0*mu*inner(eps(u), eps(v)) - div(v)*p + q*div(u)) * dx
L = Constant(0.0) * q * dx   # no body force

# --- Solve linear system ---
w = Function(W, name="w")
solve(a == L, w, bcs,
      solver_parameters={
          "linear_solver": "mumps"  # robust direct solver; PETSc default also fine
      })

(u_sol, p_sol) = w.split(True)
u_sol.rename("u", "velocity")
p_sol.rename("p", "pressure")

# --- Post-processing: speed field and colour map ---
V1 = FunctionSpace(mesh, "Lagrange", 1)
speed = project(sqrt(dot(u_sol, u_sol)), V1)
speed.rename("speed", "|u|")

plt.figure()
# FEniCS' plot() returns a matplotlib scalar mappable when using P1
pm = plot(speed, title="Speed |u|", mode="color")  # legacy matplotlib backend
plt.colorbar(pm)
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("q3_speed.png", dpi=200)
plt.close()

# --- Write fields to XDMF ---
# One file containing both variables; functions_share_mesh improves ParaView/VTK reuse.
xdmf = XDMFFile(mesh.mpi_comm(), "q3_soln.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.close()

print("Saved colour map to q3_speed.png and solution fields to q3_soln.xdmf.")