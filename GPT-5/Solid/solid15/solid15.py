# q15_fenics.py
# Legacy FEniCS (dolfin + mshr) nonlinear Saint-Venant–Kirchhoff (finite strain), plane strain
from __future__ import print_function
from dolfin import *
from mshr import *
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parameters["form_compiler"]["quadrature_degree"] = 3
parameters["form_compiler"]["representation"] = "uflacs"

# --------------------------
# Geometry and mesh
# --------------------------
Lx = 1.20
Ly = 0.20
yc = 0.10
a  = 0.03

rect = Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
holes = [
    Circle(Point(0.30, yc), a, 64),
    Circle(Point(0.60, yc), a, 64),
    Circle(Point(0.90, yc), a, 64),
]
domain = rect
for h in holes:
    domain = domain - h

# Mesh resolution: choose a target cell size ~ 0.004–0.006 m to resolve the holes well
mesh_resolution = 260  # increase if you want a finer mesh
mesh = generate_mesh(domain, mesh_resolution)

# --------------------------
# Function spaces
# --------------------------
V = VectorFunctionSpace(mesh, "CG", 2)  # quadratic for accuracy
u = Function(V, name="u")               # current displacement
du = TrialFunction(V)
v  = TestFunction(V)

# --------------------------
# Material (MPa -> Pa)
# --------------------------
lam = 5.769e6  # Pa
mu  = 3.846e6  # Pa

I2 = Identity(2)

def F_def(u_):
    return I2 + grad(u_)

def E_green(u_):
    F = F_def(u_)
    C = F.T*F
    return 0.5*(C - I2)

def S_svk(u_):
    E = E_green(u_)
    return lam*tr(E)*I2 + 2.0*mu*E

def P_firstPK(u_):
    F = F_def(u_)
    S = S_svk(u_)
    return F*S

# --------------------------
# Boundary conditions
# --------------------------
# Left edge fixed: u = (0,0)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, DOLFIN_EPS)

# Right edge prescribed ux = t*Ux, uy = 0
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, DOLFIN_EPS)

left = Left()
right = Right()

Ux_final = 0.012  # metres
t_load = Constant(0.0)

# Dirichlet BCs
zero_vec = Constant((0.0, 0.0))
bc_left  = DirichletBC(V, zero_vec, left)

# Only ux prescribed on the right, uy = 0:
# Build a mixed-style vector value where ux = t*Ux_final and uy = 0.
ux_expr = Expression(("t*Ux", "0.0"), t=0.0, Ux=Ux_final, degree=1)
bc_right = DirichletBC(V, ux_expr, right)

bcs = [bc_left, bc_right]

# --------------------------
# Variational form (total Lagrangian)
# --------------------------
P = P_firstPK(u)
Res = inner(P, grad(v))*dx
Jac = derivative(Res, u, du)  # consistent tangent

# --------------------------
# Helpers: principal strains and von Mises of S deviatoric
# --------------------------
# For a 2x2 symmetric tensor T, principal values:
# λ_max = 0.5*(trT + sqrt((T11-T22)^2 + 4*T12^2))
def principal_max_value(T):
    t = tr(T)
    d = T[0,0] - T[1,1]
    off = T[0,1]
    rad = sqrt(0.25*d*d + off*off)
    return 0.5*t + rad

def deviatoric_2nd(S):
    return S - (1.0/3.0)*tr(S)*I2

def von_mises_S(S):
    s = deviatoric_2nd(S)
    # σ_vm(S) = sqrt(1.5 * s:s)
    return sqrt(1.5*inner(s, s))

# Scalar spaces for projection/visualisation (colour maps)
Q = FunctionSpace(mesh, "CG", 1)

def compute_Emax(u_):
    E = E_green(u_)
    Emax_expr = principal_max_value(E)
    Emax = project(Emax_expr, Q, solver_type="cg", preconditioner_type="ilu")
    Emax.rename("E_max", "E_max")
    return Emax

def compute_vmS(u_):
    S = S_svk(u_)
    vm_expr = von_mises_S(S)
    vm = project(vm_expr, Q, solver_type="cg", preconditioner_type="ilu")
    vm.rename("vmS", "vmS")
    return vm

# --------------------------
# Nonlinear solver
# --------------------------
problem = NonlinearVariationalProblem(Res, u, bcs, Jac)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["absolute_tolerance"] = 1e-9
prm["newton_solver"]["relative_tolerance"] = 1e-8
prm["newton_solver"]["maximum_iterations"] = 30
prm["newton_solver"]["linear_solver"] = "mumps"

# --------------------------
# Load stepping with strain cap E_max <= 0.03
# --------------------------
Emax_cap = 0.03
num_steps = 20  # start reasonably fine; solver is robust for this material
accepted_u = Function(V)  # to hold last accepted state

# XDMF writers (append last state at the end)
xdmf_u    = XDMFFile(MPI.comm_world, "q15_u.xdmf")
xdmf_Emax = XDMFFile(MPI.comm_world, "q15_Emax.xdmf")
xdmf_u.parameters["flush_output"] = True
xdmf_Emax.parameters["flush_output"] = True
xdmf_u.parameters["functions_share_mesh"] = True
xdmf_Emax.parameters["functions_share_mesh"] = True

# Step the load and stop if the principal strain cap would be exceeded
last_accepted_step = -1
for i in range(1, num_steps+1):
    tval = float(i)/float(num_steps)
    ux_expr.t = tval  # update prescribed displacement multiplier

    try:
        solver.solve()
    except RuntimeError as e:
        # Newton failed: back off and break
        print("Newton failed at step {} (t = {:.3f}). Stopping.".format(i, tval))
        break

    # Check principal Green–Lagrange strain
    Emax = compute_Emax(u)
    Emax_array = Emax.vector().get_local()
    Emax_max   = float(Emax_array.max()) if len(Emax_array) else 0.0
    print("Step {:2d}/{:2d}, t = {:.3f}, max E_max = {:.6f}".format(i, num_steps, tval, Emax_max))

    if Emax_max <= Emax_cap + 1e-12:
        # Accept and store
        accepted_u.assign(u)
        last_accepted_step = i
        # Continue to next step
    else:
        # Do not advance the load; keep last accepted solution and stop
        print("Strain cap exceeded (E_max = {:.6f} > {}). Reverting and stopping.".format(Emax_max, Emax_cap))
        u.assign(accepted_u)
        # Recompute Emax for the reverted state for outputs
        Emax = compute_Emax(u)
        break

# If nothing was accepted (pathological), keep the initial (zero) state
if last_accepted_step < 0:
    print("No step accepted; outputs will reflect the initial state.")
    Emax = compute_Emax(u)

# --------------------------
# Post-processing & outputs
# --------------------------
# 1) Deformed configuration plot
#    Using displacement plot (Fenics visual). Scale = 1 (true), since units are metres.
plt.figure()
plot(u, mode="displacement", title="Deformed configuration (true scale)")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.tight_layout()
plt.savefig("q15_def.png", dpi=200)
plt.close()

# 2) E_max colour map
plt.figure()
pE = plot(Emax, title="Max principal Green–Lagrange strain, E_max")
plt.colorbar(pE)
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.tight_layout()
plt.savefig("q15_Emax.png", dpi=200)
plt.close()

# 3) von Mises of S (deviatoric)
vmS = compute_vmS(u)
plt.figure()
pV = plot(vmS, title="von Mises of second PK stress, ||s|| (Pa)")
plt.colorbar(pV)
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.tight_layout()
plt.savefig("q15_vmS.png", dpi=200)
plt.close()

# 4) XDMF exports (final accepted state)
#    Write mesh + functions (one time slice)
u.rename("u", "u")
xdmf_u.write(u, 0.0)
xdmf_Emax.write(Emax, 0.0)

print("Saved files: q15_def.png, q15_Emax.png, q15_vmS.png, q15_u.xdmf, q15_Emax.xdmf")