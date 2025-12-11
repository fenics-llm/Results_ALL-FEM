# filename: q5_cavity.py
from dolfin import *
from ufl_legacy import sqrt, dot   # legacy UFL for compatibility with current dolfin
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
rho = Constant(1.0)          # density
mu  = Constant(0.01)         # dynamic viscosity
eps = Constant(1e-10)        # tiny pressure stabilization (removes singularity)

# ----------------------------------------------------------------------
# Mesh (128 × 128 uniform)
# ----------------------------------------------------------------------
nx, ny = 128, 128
mesh = UnitSquareMesh(nx, ny)

# ----------------------------------------------------------------------
# Function spaces (Taylor–Hood: P2/P1)
# ----------------------------------------------------------------------
V = VectorFunctionSpace(mesh, "CG", 2)   # velocity (P2)
Q = FunctionSpace(mesh, "CG", 1)        # pressure (P1)

# Mixed space built from the two elements
mixed_elem = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# ----------------------------------------------------------------------
# Boundary definitions
# ----------------------------------------------------------------------
class Lid(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[0], 0.0) or near(x[0], 1.0))

lid   = Lid()
walls = Walls()

# ----------------------------------------------------------------------
# Dirichlet boundary conditions (velocity only)
# ----------------------------------------------------------------------
bc_lid   = DirichletBC(W.sub(0), Constant((1.0, 0.0)), lid)
bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)

bcs = [bc_lid, bc_walls]   # pressure left free; we will re‑reference it later

# ----------------------------------------------------------------------
# Unknown, trial and test functions
# ----------------------------------------------------------------------
w = Function(W)                     # (u, p)
(u, p) = split(w)                   # current iterate (used in nonlinear form)
(v, q) = TestFunctions(W)           # test functions
dw = TrialFunction(W)               # trial function for Jacobian

# ----------------------------------------------------------------------
# Variational form (steady incompressible Navier–Stokes)
# ----------------------------------------------------------------------
F = ( rho*dot(dot(u, nabla_grad(u)), v)*dx
      + mu*inner(grad(u), grad(v))*dx
      - div(v)*p*dx
      - q*div(u)*dx
      + eps*p*q*dx )                # tiny pressure stabilization

# Jacobian of the residual
J = derivative(F, w, dw)

# ----------------------------------------------------------------------
# Non‑linear solver (Newton)
# ----------------------------------------------------------------------
problem = NonlinearVariationalProblem(F, w, bcs, J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm["newton_solver"]["relative_tolerance"] = 1e-6
prm["newton_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["linear_solver"] = "mumps"
prm["newton_solver"]["report"] = True

solver.solve()

# ----------------------------------------------------------------------
# Separate velocity and pressure
# ----------------------------------------------------------------------
(u_sol, p_sol) = w.split()

# ----------------------------------------------------------------------
# Re‑reference pressure: enforce p(0,0)=0
# ----------------------------------------------------------------------
p0 = p_sol(Point(0.0, 0.0))
p_sol.vector()[:] -= p0
p_sol.vector().apply("insert")

# ----------------------------------------------------------------------
# Save solution to XDMF
# ----------------------------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q5_soln.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.close()

# ----------------------------------------------------------------------
# Compute speed magnitude (project onto CG1 so that values match mesh vertices)
# ----------------------------------------------------------------------
speed = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "CG", 1))

# ----------------------------------------------------------------------
# Plot speed magnitude and save as PNG
# ----------------------------------------------------------------------
plt.figure(figsize=(6, 5))
trip = plt.tripcolor(mesh.coordinates()[:, 0],
                     mesh.coordinates()[:, 1],
                     mesh.cells(),
                     speed.vector().get_local(),
                     shading='gouraud',
                     cmap='viridis')
plt.colorbar(trip, label='|u|')
plt.title('Lid‑driven cavity speed magnitude')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.tight_layout()
plt.savefig("q5_speed.png", dpi=300)
plt.close()