# filename: q7_navier_stokes.py
import matplotlib
matplotlib.use('Agg')          # headless backend
import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh

# ------------------------------------------------------------
# Geometry and mesh
# ------------------------------------------------------------
L, H = 2.2, 0.41
R = 0.05
center = Point(0.20, 0.20)

domain = Rectangle(Point(0.0, 0.0), Point(L, H)) - Circle(center, R, 64)
mesh = generate_mesh(domain, 64)   # increase resolution if needed

# ------------------------------------------------------------
# Physical parameters
# ------------------------------------------------------------
mu = 0.001          # dynamic viscosity
rho = 1.0            # density
Ubar = 0.2           # mean inlet velocity
D = 2.0*R           # cylinder diameter

# ------------------------------------------------------------
# Function spaces (Taylor–Hood)
# ------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# ------------------------------------------------------------
# Boundary definitions
# ------------------------------------------------------------
tol = 1E-10

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, tol)

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0, tol) or near(x[1], H, tol))

class CircleBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[0]-center.x())**2 + (x[1]-center.y())**2 < (R+tol)**2)

class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L, tol) and near(x[1], H/2, tol)

inlet = Inlet()
outlet = Outlet()
walls = Walls()
circle = CircleBoundary()
p_point = PressurePoint()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
inlet.mark(boundaries, 1)
outlet.mark(boundaries, 2)
walls.mark(boundaries, 3)
circle.mark(boundaries, 4)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# ------------------------------------------------------------
# Inlet velocity profile
# ------------------------------------------------------------
inlet_profile = Expression(("6.0*Ubar*x[1]*(H - x[1])/(H*H)", "0.0"),
                           Ubar=Ubar, H=H, degree=2)

# ------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------
bcu_inlet  = DirichletBC(W.sub(0), inlet_profile, inlet)
bcu_walls  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)
bcu_circle = DirichletBC(W.sub(0), Constant((0.0, 0.0)), circle)
bcp_point  = DirichletBC(W.sub(1), Constant(0.0), p_point, method="pointwise")
bcs = [bcu_inlet, bcu_walls, bcu_circle, bcp_point]   # outlet is natural

# ------------------------------------------------------------
# Variational formulation (steady Navier–Stokes)
# ------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

U = Function(W)          # current iterate (u,p)
(u_, p_) = split(U)

F = (2*mu*inner(sym(grad(u_)), sym(grad(v)))*dx
     + rho*inner(dot(u_, nabla_grad(u_)), v)*dx
     - div(v)*p_*dx
     + q*div(u_)*dx)

J = derivative(F, U, TrialFunction(W))

# ------------------------------------------------------------
# Solve nonlinear problem
# ------------------------------------------------------------
problem = NonlinearVariationalProblem(F, U, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["newton_solver"]["relative_tolerance"] = 1e-6
prm["newton_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["linear_solver"] = "mumps"
prm["newton_solver"]["krylov_solver"]["monitor_convergence"] = False
prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
prm["newton_solver"]["krylov_solver"]["relative_tolerance"] = 1e-6
prm["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["krylov_solver"]["maximum_iterations"] = 1000

solver.solve()
U0, P0 = U.split(deepcopy=True)   # velocity and pressure

# ------------------------------------------------------------
# Drag coefficient on the circular obstacle
# ------------------------------------------------------------
n = FacetNormal(mesh)
sigma = -P0*Identity(2) + mu*(grad(U0) + grad(U0).T)
drag = assemble(dot(sigma*n, Constant((1.0, 0.0)))*ds(4))
C_D = 2.0*drag/(rho*Ubar**2*D)

print("Drag force (x-direction) = {:.6e} N".format(drag))
print("Drag coefficient C_D    = {:.6e}".format(C_D))

# ------------------------------------------------------------
# Save speed field as PNG
# ------------------------------------------------------------
speed = sqrt(dot(U0, U0))
plt.figure(figsize=(8,3))
c = plot(speed, title="Speed |u| (m/s)", cmap="viridis")
plt.colorbar(c)
plt.axis('off')
plt.tight_layout()
plt.savefig("q7_speed.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# Save solution to XDMF
# ------------------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q7_soln.xdmf")
xdmf.write(U0, 0.0)
xdmf.write(P0, 0.0)
xdmf.close()