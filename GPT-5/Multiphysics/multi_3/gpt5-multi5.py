# cahn_hilliard_legacy_fenics.py
from __future__ import print_function
import sys
from math import log
import numpy as np

from dolfin import (
    UnitSquareMesh, MPI, set_log_level, WARNING,
    FiniteElement, MixedElement, FunctionSpace, Function, TestFunctions, TrialFunctions,
    split, derivative, inner, grad, dx, Constant, ln, assemble, PETScOptions,
    NonlinearVariationalProblem, NonlinearVariationalSolver, XDMFFile, MeshFunction,
    DOLFIN_EPS
)

# ---------------------------
# Problem settings (editable)
# ---------------------------
theta = 1.5
alpha = 3000.0
Tfinal = 4.0e-2
report_times = [0.0, 3.0e-6, 1.0e-4, 1.0e-3, 4.0e-2]

cbar = 0.63
perturb_amp = 0.05  # uniform in [-0.05, 0.05]
seed = 42           # set to None for fresh random initialisation

# Mesh resolution (increase for finer patterns; 128x128 is a reasonable start)
Nx = Ny = 128

# Time-step control
dt_min = 1.0e-7
dt_max = 1.0e-4
dt0    = 3.0e-7       # starting ∆t in the suggested range
fac_up = 1.2          # gentle enlargement after easy steps
fac_dn = 0.5          # cut if nonlinear solve struggles
max_retries = 8       # how many times to cut ∆t on failure before giving up

# Regularisation for log to keep c in (0,1)
eps_log = Constant(1.0e-8)

# ---------------------------
# Periodic boundary on [0,1]^2
# ---------------------------
from dolfin import SubDomain, near

class PeriodicBoundary(SubDomain):
    # Left boundary is "target", right boundary is "source"
    def inside(self, x, on_boundary):
        return bool((near(x[0], 0.0) or near(x[1], 0.0)) and
                    (not ((near(x[0], 1.0) or near(x[1], 1.0)))) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - 1.0 if near(x[0], 1.0) else x[0]
        y[1] = x[1] - 1.0 if near(x[1], 1.0) else x[1]

# ---------------------------
# Build mesh and spaces
# ---------------------------
comm = MPI.comm_world
rank = MPI.rank(comm)
set_log_level(WARNING)

mesh = UnitSquareMesh(comm, Nx, Ny)

P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = MixedElement([P1, P1])  # (c, mu)
W = FunctionSpace(mesh, ME, constrained_domain=PeriodicBoundary())

# Unknowns and tests
w = Function(W)
(c, mu) = split(w)
(v, q) = TestFunctions(W)

# Previous time level
w_n = Function(W)
(c_n, mu_n) = split(w_n)

# Time step as Constant (so UFL sees updates)
dt = Constant(dt0)

# ---------------------------
# Helper: free energy derivative μ_c(c)
# ---------------------------
# μ_c = (0.5/theta)*ln( c/(1-c) ) + 1 - 2c   (with small eps for safety)
def mu_c_expr(cc):
    return (0.5/theta) * ln((cc + eps_log) / (1.0 - cc + eps_log)) + (1.0 - 2.0*cc)

# Mobility M(c) = c(1-c) (degenerate)
def M_expr(cc):
    return cc * (1.0 - cc)

# ---------------------------
# Weak form (Backward Euler)
# ---------------------------
M = M_expr(c)
mu_c = mu_c_expr(c)

F1 = ((c - c_n)/dt) * v * dx + inner(M * grad(mu), grad(v)) * dx
F2 = (mu * q) * dx - (3.0*alpha * mu_c) * q * dx + inner(grad(c), grad(q)) * dx

F = F1 + F2

J = derivative(F, w)

problem = NonlinearVariationalProblem(F, w, J=J)
solver = NonlinearVariationalSolver(problem)

# Prefer Newton solver (robust on this system)
prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["absolute_tolerance"] = 1.0e-10
prm["newton_solver"]["relative_tolerance"] = 1.0e-8
prm["newton_solver"]["maximum_iterations"] = 30
prm["newton_solver"]["report"] = False
prm["newton_solver"]["linear_solver"] = "bicgstab"
prm["newton_solver"]["preconditioner"] = "ilu"

# ---------------------------
# Initial condition c(x,0) = cbar + r, periodic
# ---------------------------
V = W.sub(0).collapse()  # scalar P1 for c
c0_fun = Function(V)

if seed is not None:
    np.random.seed(seed)

# Perturbation defined on DOF coordinates (P1 is fine)
coords = V.tabulate_dof_coordinates().reshape((-1, 2))
nd = coords.shape[0]
rvals = (np.random.rand(nd) - 0.5) * (2.0 * perturb_amp)  # in [-0.05, 0.05]
rvals -= np.mean(rvals)  # zero mean
c_init_vals = cbar + rvals

# Keep within (0,1) safely
c_init_vals = np.clip(c_init_vals, 1e-6, 1.0 - 1e-6)
c0_fun.vector().set_local(c_init_vals)
c0_fun.vector().apply("insert")

# Put c0 into the mixed vector w_n (and w)
assigner_c = W.sub(0).collapse()
# Easiest: build a mixed vector via interpolation from components
# Create a temporary mixed Function with c=c0, mu=0
from dolfin import interpolate, as_vector

# Start with mu satisfying the auxiliary relation for consistency at t=0:
# Solve (mu, q) + (grad c, grad q) = (3α μ_c(c), q)
# Build a one-off linear solve for mu0 on the same periodic space
from dolfin import TrialFunction, TestFunction, lhs, rhs, solve

Vper = FunctionSpace(mesh, P1, constrained_domain=PeriodicBoundary())
mu0 = Function(Vper)
c0_on_per = Function(Vper)
c0_on_per.interpolate(c0_fun)

q0 = TestFunction(Vper)
mu_trial = TrialFunction(Vper)
a_mu = (mu_trial * q0 + inner(grad(c0_on_per), grad(q0))) * dx
L_mu = (3.0*alpha * mu_c_expr(c0_on_per)) * q0 * dx
solve(a_mu == L_mu, mu0)

# Now pack (c0, mu0) into mixed w and w_n
# Use a simple interpolation via a UFL expression
W0, _ = W.sub(0).collapse(), W.sub(1).collapse()

c_comp = Function(W0); c_comp.interpolate(c0_on_per)
mu_comp = Function(W0); mu_comp.interpolate(mu0)

from dolfin import as_backend_type
# Interpolate pair into W
from dolfin import Expression  # not needed actually; we can use assign on subfunctions
w.assign(Function(W))  # reset
w_n.assign(Function(W))

# Access subfunctions via w.split() gives copies; instead, use sub() with assign
w_c = w.sub(0); w_mu = w.sub(1)
w_c.assign(c_comp); w_mu.assign(mu_comp)

w_n_c = w_n.sub(0); w_n_mu = w_n.sub(1)
w_n_c.assign(c_comp); w_n_mu.assign(mu_comp)

# ---------------------------
# Output: XDMF time-series
# ---------------------------
xdmf = XDMFFile(comm, "cahn_hilliard.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

# Helper to write current (c, mu)
def write_outputs(time, ww):
    c_out, mu_out = ww.split(deepcopy=True)  # safe copy for output
    c_out.rename("c", "concentration")
    mu_out.rename("mu", "chemical_potential")
    xdmf.write(c_out, time)
    xdmf.write(mu_out, time)

# ---------------------------
# Time marching with adaptive ∆t and exact reporting at targets
# ---------------------------
t = 0.0
targets = list(report_times)
targets.sort()
target_idx = 0

def almost_equal(a, b, tol=1e-14):
    return abs(a - b) <= tol*max(1.0, abs(a), abs(b))

# Save t=0
if target_idx < len(targets) and almost_equal(t, targets[target_idx], 1e-12):
    if rank == 0:
        print("Writing output at t =", t)
    write_outputs(t, w)
    target_idx += 1

current_dt = float(dt.values()[0])
if current_dt <= 0:
    current_dt = dt0
dt.assign(current_dt)

if rank == 0:
    print("Starting time integration up to T =", Tfinal)

while t < Tfinal - 1e-16:
    # Adjust dt to hit next target exactly if needed
    if target_idx < len(targets):
        t_target = targets[target_idx]
        if t + float(dt) > t_target:
            dt.assign(max(dt_min, t_target - t))

    # Backup state
    w_backup = Function(W); w_backup.assign(w)
    w_n.assign(w)

    # Try solve with possible ∆t cuts
    success = False
    retries = 0
    while retries <= max_retries and not success:
        try:
            its = solver.solve()  # will raise on failure
            success = True
        except RuntimeError as e:
            # Cut ∆t and retry from backup
            current_dt = float(dt)
            new_dt = max(dt_min, current_dt * fac_dn)
            if rank == 0:
                print("  Nonlinear solve failed at t = {:.6e}, reducing dt: {:.3e} -> {:.3e}".format(
                    t, current_dt, new_dt))
            dt.assign(new_dt)
            w.assign(w_backup)
            retries += 1

    if not success:
        if rank == 0:
            print("ERROR: Nonlinear solver failed repeatedly. Aborting.")
        sys.exit(2)

    # Advance time
    t += float(dt)

    # If we just hit a target (exactly or within roundoff), write results
    wrote = False
    if target_idx < len(targets):
        # Snap tiny round-off
        if t > targets[target_idx] and t - targets[target_idx] < 1e-14:
            t = targets[target_idx]
        if almost_equal(t, targets[target_idx], 1e-12):
            if rank == 0:
                print("Writing output at t =", t)
            write_outputs(t, w)
            target_idx += 1
            wrote = True

    # If solve was very easy, enlarge dt slightly (but keep headroom to next target)
    if success and retries == 0:
        cur = float(dt)
        grow = min(dt_max, cur * fac_up)
        # Do not overshoot the next target badly; keep at most half the gap
        if target_idx < len(targets):
            gap = max(1e-18, targets[target_idx] - t)
            grow = min(grow, 0.5 * gap)
        dt.assign(grow)

# Ensure final requested time is written (in case Tfinal equals last target)
if target_idx < len(targets) and almost_equal(t, targets[-1], 1e-12):
    if rank == 0:
        print("Final write at t =", t)
    write_outputs(t, w)

xdmf.close()

if rank == 0:
    print("Done. Wrote time-series to cahn_hilliard.xdmf")