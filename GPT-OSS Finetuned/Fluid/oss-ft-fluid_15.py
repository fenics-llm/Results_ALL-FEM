from fenics import *
import numpy as np

# ------------------- Mesh and periodic BC -------------------
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool(on_boundary and (near(x[0], 0) or near(x[1], 0)))
    def map(self, x, y):
        if near(x[0], 1) and near(x[1], 1):
            y[0] = x[0] - 1.0
            y[1] = x[1] - 1.0
        elif near(x[0], 1):
            y[0] = x[0] - 1.0
            y[1] = x[1]
        else:  # near(x[1], 1)
            y[0] = x[0]
            y[1] = x[1] - 1.0

mesh = UnitSquareMesh(64, 64)
pbc = PeriodicBoundary()

# ------------------- Function spaces -------------------
V_el = VectorElement("CG", mesh.ufl_cell(), 2)
Q_el = FiniteElement("CG", mesh.ufl_cell(), 1)
W_el = MixedElement([V_el, Q_el])
W = FunctionSpace(mesh, W_el, constrained_domain=pbc)

# ------------------- Trial/Test functions -------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# ------------------- Parameters -------------------
rho = 1.0
nu  = 1e-3
T   = 1.0
dt  = 5e-4
num_steps = int(T/dt)

# ------------------- Initial condition -------------------
u0_expr = Expression(("sin(2*pi*x[0])*cos(2*pi*x[1])",
                       "-cos(2*pi*x[0])*sin(2*pi*x[1])"),
                      degree=5, pi=np.pi)
w_n = Function(W)
assign(w_n.sub(0), interpolate(u0_expr, W.sub(0).collapse()))
assign(w_n.sub(1), interpolate(Constant(0.0), W.sub(1).collapse()))

# ------------------- Variational problem (Backward Euler) -------------------
U_n, P_n = split(w_n)
F = (rho/dt)*inner(u - U_n, v)*dx \
    + rho*inner(dot(U_n, nabla_grad(u)), v)*dx \
    + nu*inner(grad(u), grad(v))*dx \
    - div(v)*p*dx \
    - q*div(u)*dx
a, L = lhs(F), rhs(F)

# ------------------- Output -------------------
xdmf = XDMFFile("navier_stokes_periodic.xdmf")
xdmf.parameters["flush_output"] = True
save_times = [0.0, 0.25, 0.5, 1.0]

# write initial condition
u_h, _ = w_n.split()
xdmf.write(u_h, 0.0)

# ------------------- Time stepping -------------------
t = dt
while t <= T + DOLFIN_EPS:
    solve(a == L, w_n, [])
    u_h, _ = w_n.split()
    if any(abs(t - ts) < dt/2 for ts in save_times):
        xdmf.write(u_h, t)
    t += dt