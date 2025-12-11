# filename: cahn_hilliard.py
from fenics import *
import numpy as np
import ufl_legacy as ufl   # legacy UFL compatible with dolfin

# ----------------------------------------------------------------------
# Periodic boundary (identifies opposite edges of the unit square)
# ----------------------------------------------------------------------
class PeriodicBoundary(SubDomain):
    # left & bottom are the "target" sides
    def inside(self, x, on_boundary):
        return bool((near(x[0], 0.0) or near(x[1], 0.0)) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.0 if near(x[0], 1.0) else x[0]
        y[1] = x[1] - 1.0 if near(x[1], 1.0) else x[1]

# ----------------------------------------------------------------------
# Mesh and function spaces (periodic)
# ----------------------------------------------------------------------
N = 64                                   # mesh resolution
mesh = UnitSquareMesh(N, N, "crossed")
pbc = PeriodicBoundary()

V = FunctionSpace(mesh, "CG", 1, constrained_domain=pbc)   # scalar CG1, periodic

# ----------------------------------------------------------------------
# Physical parameters
# ----------------------------------------------------------------------
theta = Constant(1.5)
alpha = Constant(3000.0)
T      = 0.04
t      = 0.0

# Adaptive time stepping parameters
dt      = 2e-7
dt_min  = 1e-7
dt_max  = 5e-7
dt_increase_factor = 1.2
dt_decrease_factor = 0.5

# Times at which we store output
output_times = [0.0, 3e-6, 1e-4, 1e-3, 4e-2]

# ----------------------------------------------------------------------
# Initial condition: c = c̄ + uniform perturbation in [-0.05,0.05]
# ----------------------------------------------------------------------
c_bar = 0.63
np.random.seed(42)
# perturbation on a regular grid (N+1)×(N+1)
r_grid = 0.05 * (2 * np.random.rand(N + 1, N + 1) - 1.0)   # uniform in [-0.05,0.05]

c0 = Function(V)
c0_vec = c0.vector()
dof_coords = V.tabulate_dof_coordinates().reshape((-1, 2))

for i, x in enumerate(dof_coords):
    ix = min(int(x[0] * N), N - 1)
    iy = min(int(x[1] * N), N - 1)
    c0_vec[i] = c_bar + r_grid[ix, iy]

# keep concentration strictly inside (0,1) for the logarithm
c0_vec[:] = np.clip(c0_vec, 1e-8, 1.0 - 1e-8)

# ----------------------------------------------------------------------
# Functions for the current and previous time step
# ----------------------------------------------------------------------
c   = Function(V)   # concentration at new time (unknown)
c_n = Function(V)   # concentration at previous time (known)
mu  = Function(V)   # chemical potential (computed from c_n)

c.assign(c0)
c_n.assign(c0)

# ----------------------------------------------------------------------
# Weak forms (splitting the fourth‑order equation)
# ----------------------------------------------------------------------
# 1) Compute μ from the previous concentration (explicit in c_n)
mu_trial = TrialFunction(V)
v_test   = TestFunction(V)

mu_c_n = (0.5 / theta) * ufl.ln(c_n / (1.0 - c_n)) + 1.0 - 2.0 * c_n
a_mu = mu_trial * v_test * dx
L_mu = (3.0 * alpha * mu_c_n) * v_test * dx + dot(grad(c_n), grad(v_test)) * dx

mu_problem = LinearVariationalProblem(a_mu, L_mu, mu)
mu_solver  = LinearVariationalSolver(mu_problem)
mu_solver.parameters['linear_solver'] = 'mumps'   # robust direct solver

# 2) Update concentration using backward Euler with mobility evaluated at c_n
c_trial = TrialFunction(V)
w_test  = TestFunction(V)

M_n = c_n * (1.0 - c_n)          # degenerate mobility evaluated at previous step
a_c = (c_trial / dt) * w_test * dx + M_n * dot(grad(mu), grad(w_test)) * dx
L_c = (c_n / dt) * w_test * dx

c_problem = LinearVariationalProblem(a_c, L_c, c)
c_solver  = LinearVariationalSolver(c_problem)
c_solver.parameters['linear_solver'] = 'mumps'

# ----------------------------------------------------------------------
# XDMF output (time series)
# ----------------------------------------------------------------------
xdmf_file = XDMFFile(mesh.mpi_comm(), "cahn_hilliard.xdmf")
xdmf_file.parameters["flush_output"] = True
xdmf_file.parameters["functions_share_mesh"] = True

def write_output(time):
    c.rename("c", "concentration")
    mu.rename("mu", "chemical_potential")
    xdmf_file.write(c, time)
    xdmf_file.write(mu, time)

# write initial state
write_output(t)

# ----------------------------------------------------------------------
# Time stepping loop with adaptive dt
# ----------------------------------------------------------------------
while t < T - 1e-12:
    # adjust dt so we do not step beyond final time
    if t + dt > T:
        dt = T - t

    # ---- 1) compute μ from c_n (explicit) ----
    try:
        mu_solver.solve()
    except RuntimeError:
        dt = max(dt * dt_decrease_factor, dt_min)
        continue

    # ---- 2) solve for new concentration c (implicit) ----
    try:
        c_solver.solve()
    except RuntimeError:
        dt = max(dt * dt_decrease_factor, dt_min)
        continue

    # ---- advance time ----
    t += dt
    c_n.assign(c)          # c_n becomes the newly computed concentration

    # ---- simple adaptive dt control (increase if step succeeded) ----
    dt = min(dt * dt_increase_factor, dt_max)

    # ---- write output if required ----
    if any(abs(t - tout) < 1e-12 for tout in output_times):
        write_output(t)

xdmf_file.close()