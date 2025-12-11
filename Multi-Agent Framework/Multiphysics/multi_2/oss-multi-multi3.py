# Allen–Cahn curvature-flow on the unit square (legacy FEniCS)

from dolfin import *
import numpy as np

# --------------------------------------------------------------
# 1. Mesh and function space
# --------------------------------------------------------------
N = 200                     # 200 × 200 elements
mesh = UnitSquareMesh(N, N)

V = FunctionSpace(mesh, "Lagrange", 1)   # P1, natural Neumann BCs

# --------------------------------------------------------------
# 2. Model parameters
# --------------------------------------------------------------
eps = 0.01                 # interface thickness
M   = 1.0                  # mobility
dt  = 1.0e-3               # time step
T   = 0.20                 # final time
num_steps = int(T / dt)    # 200 steps

# --------------------------------------------------------------
# 3. Signed distance to a centred square (side = 0.5)
# --------------------------------------------------------------
class SignedDistSquare(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def eval(self, values, x):
        d = max(abs(x[0] - 0.5), abs(x[1] - 0.5)) - 0.25
        values[0] = d
    def value_shape(self):
        return ()

d_rect = SignedDistSquare(degree=2)

# --------------------------------------------------------------
# 4. Initial condition: tanh(d / (sqrt(2)*eps))
# --------------------------------------------------------------
phi_n = Function(V)   # phi at previous time level
phi   = Function(V)   # unknown at new time level

# corrected expression: no Python sqrt passed, use 2.0 for safety
phi_init_expr = Expression("tanh(d/(sqrt(2.0)*eps))",
                           d=d_rect, eps=eps, degree=2)
phi_n.interpolate(phi_init_expr)

# --------------------------------------------------------------
# 5. Weak formulation (backward Euler, fully implicit)
# --------------------------------------------------------------
phi_trial = TrialFunction(V)
v          = TestFunction(V)

# double-well derivative: W'(phi) = phi**3 - phi
Wprime = phi**3 - phi

F = ( (phi - phi_n) / dt ) * v * dx \
    + M/eps * Wprime * v * dx \
    + M*eps * dot(grad(phi), grad(v)) * dx

J = derivative(F, phi, phi_trial)   # Jacobian for Newton

# No Dirichlet BCs (homogeneous Neumann is natural)
bcs = []   # empty list

# --------------------------------------------------------------
# 6. Time stepping and output
# --------------------------------------------------------------
output_times = [0.0, 0.05, 0.10, 0.20]
output_files = {}
for t_out in output_times:
    fname = "phi_t{:.2f}.xdmf".format(t_out)
    output_files[t_out] = XDMFFile(mesh.mpi_comm(), fname)

t = 0.0
# write initial field
output_files[0.0].write(phi_n, t)

for n in range(1, num_steps + 1):
    t += dt

    # solve the nonlinear problem
    phi.assign(phi_n)
    solve(F == 0, phi, bcs, J=J)

    # output at prescribed times (within a tolerance)
    for t_out in output_times:
        if abs(t - t_out) < 1e-8:
            output_files[t_out].write(phi, t)

    # update for next step
    phi_n.assign(phi)

# close all XDMF files
for f in output_files.values():
    f.close()