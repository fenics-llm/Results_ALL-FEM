# filename: vms_cylinder.py
from dolfin import *
import mshr
import numpy as np

# -------------------------------------------------
# Geometry and mesh (sufficiently resolved)
# -------------------------------------------------
D = 1.0                     # cylinder diameter
R = D/2.0                   # radius
U_in = 1.0                  # inflow velocity
nu = 2.56e-5                # kinematic viscosity
rho = 1.0                   # density
mu = rho*nu                 # dynamic viscosity

L = 30.0*D                 # half domain length
domain = mshr.Rectangle(Point(-L, -L), Point(L, L)) \
         - mshr.Circle(Point(0.0, 0.0), R, 64)   # 64 points on circle
mesh = mshr.generate_mesh(domain, 128)           # 128 cells per direction

# -------------------------------------------------
# Taylor–Hood mixed space (P2–P1)
# -------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# -------------------------------------------------
# Boundary markers
# -------------------------------------------------
tol = 1E-10
class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -L, tol)
class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0],  L, tol)
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], -L, tol) or near(x[1], L, tol))
class Cylinder(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0]**2 + x[1]**2 < (R+tol)**2)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
Inflow().mark(boundaries, 1)
Outflow().mark(boundaries, 2)
Walls().mark(boundaries, 3)
Cylinder().mark(boundaries, 4)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Trial / test functions
# -------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# -------------------------------------------------
# Functions for time stepping
# -------------------------------------------------
w   = Function(W)          # (u,p) at new time level
w0  = Function(W)          # (u,p) at previous time level
(u0, p0) = w0.split(deepcopy=True)

# -------------------------------------------------
# Initial condition (zero + tiny random perturbation)
# -------------------------------------------------
u0.vector()[:] = 0.0
p0.vector()[:] = 0.0
u0.vector().set_local(u0.vector().get_local() + 1e-4*np.random.randn(u0.vector().size()))

# -------------------------------------------------
# Boundary conditions
# -------------------------------------------------
bcu_in = DirichletBC(W.sub(0), Constant((U_in, 0.0)), boundaries, 1)
bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 3)
bcu_cyl = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 4)
bcp_out = DirichletBC(W.sub(1), Constant(0.0), boundaries, 2)   # pressure gauge at outflow
bcs = [bcu_in, bcu_walls, bcu_cyl, bcp_out]

# -------------------------------------------------
# Time stepping parameters
# -------------------------------------------------
T   = 10.0
dt  = 0.01
t   = 0.0

# -------------------------------------------------
# Stabilisation parameters
# -------------------------------------------------
h = CellDiameter(mesh)
gamma = 0.1   # grad‑div penalty

# -------------------------------------------------
# Stress tensor and normal
# -------------------------------------------------
def sigma(u, p):
    return 2.0*mu*sym(nabla_grad(u)) - p*Identity(2)
n = FacetNormal(mesh)

# -------------------------------------------------
# Variational form (backward Euler)
# -------------------------------------------------
U_norm = sqrt(dot(u0, u0) + DOLFIN_EPS)
tau_m = 1.0 / sqrt( (2.0/dt)**2 + (2.0*U_norm/h)**2 + (4.0*mu/(rho*h**2))**2 )
tau_c = tau_m * mu

F = rho*dot((u - u0)/dt, v)*dx \
    + rho*dot(dot(u0, nabla_grad(u0)), v)*dx \
    + 2.0*mu*inner(sym(nabla_grad(u)), sym(nabla_grad(v)))*dx \
    - div(v)*p*dx \
    + q*div(u)*dx

# Residuals
R_m = rho*((u - u0)/dt + dot(u0, nabla_grad(u0))) - mu*div(2.0*sym(nabla_grad(u))) + nabla_grad(p)
R_c = div(u)

# SUPG / PSPG terms
F += tau_m*dot(dot(u0, nabla_grad(v)), R_m)*dx \
     + tau_c*dot(nabla_grad(q), R_m)*dx \
     + tau_c*R_c*q*dx

# grad‑div stabilisation
F += gamma*div(u)*div(v)*dx

a = lhs(F)
L = rhs(F)

A = assemble(a)
[bc.apply(A) for bc in bcs]

# -------------------------------------------------
# Drag accumulation
# -------------------------------------------------
drag_acc = 0.0
drag_time = 0.0

# -------------------------------------------------
# Output file
# -------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "vms_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

# -------------------------------------------------
# Time loop
# -------------------------------------------------
while t < T - DOLFIN_EPS:
    t += dt

    # recompute tau based on current u0
    U_norm = sqrt(dot(u0, u0) + DOLFIN_EPS)
    tau_m = 1.0 / sqrt( (2.0/dt)**2 + (2.0*U_norm/h)**2 + (4.0*mu/(rho*h**2))**2 )
    tau_c = tau_m * mu

    # assemble RHS with updated tau
    b = assemble(L)
    [bc.apply(b) for bc in bcs]

    # solve linear system
    solve(A, w.vector(), b, "bicgstab", "ilu")

    # split solution
    (u, p) = w.split(deepcopy=True)

    # drag on cylinder (boundary id 4) for t >= 8.0
    if t >= 8.0:
        traction = dot(sigma(u, p), n)
        # x‑component of traction (drag direction)
        drag_local = -traction[0]*ds(4)
        drag_acc += assemble(drag_local)
        drag_time += dt

    # update previous solution
    w0.assign(w)

    # write final fields at t = T
    if near(t, T, dt/2):
        xdmf.write(u, t)
        xdmf.write(p, t)

# -------------------------------------------------
# Mean drag coefficient over [8,10] s
# -------------------------------------------------
U_ref = U_in
D_ref = D
C_D = (2.0/(rho*U_ref**2*D_ref)) * (drag_acc/drag_time)
print("Mean drag coefficient C_D over [8,10] s = %.6f" % C_D)

xdmf.close()