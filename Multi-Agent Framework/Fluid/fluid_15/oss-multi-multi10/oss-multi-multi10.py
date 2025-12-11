# -*- coding: utf-8 -*-
#
# 2-D incompressible Navier–Stokes on a periodic unit square (legacy FEniCS)
#
# ρ = 1, ν = 1e-3, T = 1.0
#  u0 = sin(2πx)cos(2πy), v0 = -cos(2πx)sin(2πy)
#
#  Output: velocity (and pressure) at t = 0, 0.25, 0.5, 1.0  (XDMF)
#
from dolfin import *
import numpy as np

# -----------------------------------------------------------------
# 1. Periodic boundary definition
# -----------------------------------------------------------------
class PeriodicBoundary(SubDomain):
    # Left (x=0) and bottom (y=0) are master; right (x=1) and top (y=1) are slave
    def inside(self, x, on_boundary):
        return bool(((near(x[0], 0.0) or near(x[1], 0.0)) and (not (near(x[0], 0.0) and near(x[1], 0.0)))) and on_boundary)

    def map(self, x, y):
        # map right side to left
        if near(x[0], 1.0) and near(x[1], 1.0):
            y[0] = x[0] - 1.0
            y[1] = x[1] - 1.0
        elif near(x[0], 1.0):
            y[0] = x[0] - 1.0
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1] - 1.0

# -----------------------------------------------------------------
# 2. Mesh (regular) – periodicity will be enforced via constrained_domain
# -----------------------------------------------------------------
N = 32                     # mesh resolution
mesh = UnitSquareMesh(N, N, "crossed")   # regular mesh; periodicity handled later

# -----------------------------------------------------------------
# 3. Mixed Taylor–Hood space (periodic)
# -----------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
pbc = PeriodicBoundary()
W  = FunctionSpace(mesh, TH, constrained_domain=pbc)

# -----------------------------------------------------------------
# 4. Trial / test functions
# -----------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# -----------------------------------------------------------------
# 5. Physical parameters
# -----------------------------------------------------------------
rho = Constant(1.0)
nu  = Constant(1e-3)
mu  = rho*nu

# -----------------------------------------------------------------
# 6. Initial condition (divergence-free, periodic)
# -----------------------------------------------------------------
u0_expr = Expression(("sin(2*pi*x[0])*cos(2*pi*x[1])",
                       "-cos(2*pi*x[0])*sin(2*pi*x[1])"),
                      degree=5, pi=np.pi)

w0 = Function(W)                     # mixed (velocity, pressure)

# velocity part
V = FunctionSpace(mesh, Ve, constrained_domain=pbc)              # VectorFunctionSpace (2 components)
u0 = interpolate(u0_expr, V)              # interpolate the 2-component expression
assign(w0.sub(0), u0)                # copy into mixed function

# pressure part
Q = FunctionSpace(mesh, Pe, constrained_domain=pbc)              # scalar sub-space
p0 = interpolate(Constant(0.0), Q)        # zero pressure initial field
assign(w0.sub(1), p0)                # copy into mixed function

# -----------------------------------------------------------------
# 7. Time stepping parameters (CFL-based)
# -----------------------------------------------------------------
T  = 1.0
dt = 0.005
num_steps = int(T/dt)

# -----------------------------------------------------------------
# 8. Variational formulation (backward Euler, linearised convection)
# -----------------------------------------------------------------
U = u0

def epsilon(uu):
    return sym(grad(uu))

def sigma(uu, pp):
    return 2*nu*epsilon(uu) - pp*Identity(len(uu))

(u_prev, p_prev) = split(w0)

F = ( rho*dot((u - u_prev)/dt, v) * dx
      + rho*dot(dot(U, nabla_grad(u)), v) * dx
      + inner(sigma(u, p), epsilon(v)) * dx
      - div(v)*p * dx
      - q*div(u) * dx )

a = lhs(F)
L = rhs(F)

# -----------------------------------------------------------------
# 9. Solver
# -----------------------------------------------------------------
w = Function(W)   # solution at new time level
solver = LUSolver()
# (no unsupported parameters to set)

# -----------------------------------------------------------------
# 10. XDMF output
# -----------------------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "navier_stokes_periodic.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

output_times = [0.0, 0.25, 0.5, 1.0]
t = 0.0
step = 0

# write initial state
xdmf.write(u0, t)

# -----------------------------------------------------------------
# 11. Time loop
# -----------------------------------------------------------------
while step < num_steps:
    step += 1
    t += dt

    solve(a == L, w, solver_parameters={"linear_solver": "mumps"})
    u_new, p_new = w.split()

    assign(w0.sub(0), u_new)
    assign(w0.sub(1), p_new)

    if any(abs(t - tout) < 1e-8 for tout in output_times):
        xdmf.write(u_new, t)

xdmf.close()