# filename: taylor_green_periodic_ipcs.py
from dolfin import *
import numpy as np

# ----------------------- Parameters -----------------------
rho = 1.0
nu  = 1.0e-3
N   = 64
T   = 1.0
CFL = 0.4

# ----------------------- Periodic BC -----------------------
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and (near(x[0], 0.0) or near(x[1], 0.0)))
    def map(self, x, y):
        if   near(x[0], 1.0) and near(x[1], 1.0):
            y[0] = x[0] - 1.0; y[1] = x[1] - 1.0
        elif near(x[0], 1.0):
            y[0] = x[0] - 1.0; y[1] = x[1]
        else:  # near(x[1], 1.0)
            y[0] = x[0];       y[1] = x[1] - 1.0

pbc  = PeriodicBoundary()

# ----------------------- Mesh & spaces -----------------------
mesh = UnitSquareMesh(N, N)
h    = mesh.hmin()

V = VectorFunctionSpace(mesh, "CG", 2, constrained_domain=pbc)  # velocity (P2)
Q = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)        # pressure (P1)

u  = TrialFunction(V)
v  = TestFunction(V)
p  = TrialFunction(Q)
q  = TestFunction(Q)

u_n  = Function(V)  # velocity at t^n
u_   = Function(V)  # tentative velocity
u_1  = Function(V)  # corrected velocity (t^{n+1})

p_n  = Function(Q)  # pressure at t^n
p_1  = Function(Q)  # pressure at t^{n+1}

# ----------------------- Initial condition -----------------------
# Taylor–Green vortex
u0_expr = Expression((
    " sin(2*pi*x[0]) * cos(2*pi*x[1])",
    "-cos(2*pi*x[0]) * sin(2*pi*x[1])"
), degree=4, pi=np.pi)
u_n.interpolate(u0_expr)
u_1.assign(u_n)  # for first pressure Poisson

# ----------------------- Time step (CFL & viscous) -----------------------
umax0 = 1.0  # |u|_max of the initial field
dt_adv = CFL * h / max(umax0, 1e-12)
dt_vis = 0.5 * h*h / max(nu, 1e-12)
dt     = float(min(dt_adv, dt_vis))
dtc    = Constant(dt)

# ----------------------- Forms: IPCS-A -----------------------
# 1) Tentative velocity step
Uadv = u_n  # explicit advection velocity
F1 = (1.0/dtc)*inner(u - u_n, v)*dx \
     + inner(dot(grad(Uadv), u_n), v)*dx \
     + nu*inner(grad(u), grad(v))*dx \
     - inner(p_n, div(v))*dx
a1, L1 = lhs(F1), rhs(F1)

# 2) Pressure correction
F2 = inner(grad(p), grad(q))*dx - (1.0/dtc)*div(u_)*q*dx
a2, L2 = lhs(F2), rhs(F2)

# 3) Velocity correction
F3 = (1.0/dtc)*inner(u - u_, v)*dx + inner(grad(p_1 - p_n), v)*dx
a3, L3 = lhs(F3), rhs(F3)

A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# ----------------------- Output -----------------------
xdmf = XDMFFile(mesh.mpi_comm(), "tg_periodic_u.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

save_times = {0.0, 0.25, 0.5, 1.0}

# Save initial
t = 0.0
if any(abs(t - s) < 1e-12 for s in save_times):
    u_n.rename("u", "velocity")
    xdmf.write(u_n, t)

# ----------------------- Time loop -----------------------
while t < T - 1e-12:
    # (Optional) basic adaptive dt using current max speed
    umax = max(umax0, np.sqrt(mesh.mpi_comm().allreduce(np.max(u_n.vector().get_local()**2), op=MPI.SUM)))  # conservative fallback
    dt_adv = CFL * h / max(umax, 1e-12)
    dt_vis = 0.5 * h*h / max(nu, 1e-12)
    dt = float(min(dt_adv, dt_vis))
    dtc.assign(dt)

    # 1) Tentative velocity
    b1 = assemble(L1)
    solve(A1, u_.vector(), b1, "bicgstab", "ilu")

    # 2) Pressure correction
    b2 = assemble(L2)
    solve(A2, p_1.vector(), b2, "bicgstab", "ilu")
    # Enforce zero-mean pressure for periodic domain
    mean_p = assemble(p_1*dx) / assemble(1.0*dx(mesh))
    p_1.vector().axpy(-1.0, interpolate(Constant(mean_p), Q).vector())

    # 3) Velocity correction
    b3 = assemble(L3)
    solve(A3, u_1.vector(), b3, "bicgstab", "ilu")

    # Update for next step
    u_n.assign(u_1)
    p_n.assign(p_1)

    # Advance time
    t = t + dt

    # Output at requested times (within tolerance)
    if any(abs(t - s) <= 0.5*dt for s in save_times):
        u_1.rename("u", "velocity")
        xdmf.write(u_1, t)

# Save final in case the loop did not hit exactly t=1.0
if 1.0 not in save_times:
    u_1.rename("u", "velocity")
    xdmf.write(u_1, 1.0)