# Cahn–Hilliard spinodal decomposition on the unit square (legacy FEniCS)

from dolfin import *
import numpy as np
from ufl_legacy import min_value, max_value

# ----------------------------------------------------------------------
# 1. Periodic mesh and periodic boundary definition
# ----------------------------------------------------------------------
class PeriodicBoundary(SubDomain):
    # Map left to right, bottom to top
    def inside(self, x, on_boundary):
        return bool(on_boundary and (near(x[0], 0) or near(x[1], 0)))
    def map(self, x, y):
        y[0] = x[0] - 1.0 if near(x[0], 1.0) else x[0]
        y[1] = x[1] - 1.0 if near(x[1], 1.0) else x[1]

mesh = UnitSquareMesh(128, 128)          # plain mesh

# ----------------------------------------------------------------------
# 2. Mixed periodic function space (Taylor–Hood style)
# ----------------------------------------------------------------------
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([P1, P1])
W  = FunctionSpace(mesh, TH, constrained_domain=PeriodicBoundary())

# ----------------------------------------------------------------------
# 3. Model parameters
# ----------------------------------------------------------------------
theta = 1.5
alpha = 3000.0
c_bar = 0.63
pert_amp = 0.05
T = 0.04

# ----------------------------------------------------------------------
# 3a. Initial time step (must be defined before building the forms)
# ----------------------------------------------------------------------
dt = 1e-7   # will be adapted during the run
dtc = Constant(dt)

# ----------------------------------------------------------------------
# 4. Initial condition (zero-mean perturbation)
# ----------------------------------------------------------------------
c0 = Function(W.sub(0).collapse())
c0_vec = c0.vector()
c0_vec[:] = c_bar + pert_amp * (2*np.random.rand(c0_vec.size()) - 1.0)
c0_vec[:] -= np.mean(c0_vec[:]) - c_bar   # enforce zero-mean perturbation
c0_vec[:] = np.clip(c0_vec, 1e-8, 1.0-1e-8)

# Chemical potential from μ = 3α μ_c - Δc
mu_c_expr = (0.5/theta) * ln(c0/(1.0 - c0)) + 1.0 - 2.0*c0
mu0_expr   = 3.0*alpha*mu_c_expr - div(grad(c0))
mu0 = project(mu0_expr, W.sub(1).collapse())

# Store previous solution as a mixed Function
w_old = Function(W)
assign(w_old.sub(0), c0)
assign(w_old.sub(1), mu0)
(c_old, mu_old) = w_old.split()   # convenient references

# ----------------------------------------------------------------------
# 5. Variational problem (backward Euler, fully implicit)
# ----------------------------------------------------------------------
w = Function(W)                     # unknown at new time level
(c, mu) = split(w)
(v, q) = TestFunctions(W)          # v for c, q for μ

c_hat = min_value(max_value(c, Constant(1e-8)), Constant(1.0-1e-8))
M_c = c_hat*(1.0 - c_hat)                  # degenerate mobility
mu_c = (0.5/theta)*ln(c_hat/(1.0 - c_hat)) + 1.0 - 2.0*c_hat

F1 = (c - c_old)/dtc * v * dx + M_c*dot(grad(mu), grad(v))*dx
F2 = mu*q*dx - 3.0*alpha*mu_c*q*dx - dot(grad(c), grad(q))*dx
F  = F1 + F2
J  = derivative(F, w)

# Build a NonlinearVariationalProblem / Solver
problem = NonlinearVariationalProblem(F, w, J=J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["relative_tolerance"] = 1e-6
solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-8
solver.parameters["newton_solver"]["maximum_iterations"] = 20
solver.parameters["newton_solver"]["linear_solver"] = "bicgstab"
solver.parameters["newton_solver"]["preconditioner"] = "ilu"

# ----------------------------------------------------------------------
# 6. XDMF output
# ----------------------------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "cahn_hilliard.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

save_times = [0.0, 3e-6, 1e-4, 1e-3, 4e-2]
next_save = iter(save_times)
t_save = next(next_save)

c_init, mu_init = w_old.split()
xdmf.write(c_init, 0.0)
xdmf.write(mu_init, 0.0)
t_save = next(next_save)

# ----------------------------------------------------------------------
# 7. Time stepping with adaptive dt
# ----------------------------------------------------------------------
t = 0.0
while t < T - 1e-12:
    converged = False
    nit = 0
    while not converged:
        try:
            # initialise Newton iteration with previous solution
            w.assign(w_old)
            # solve the nonlinear system
            nit, _ = solver.solve()
            converged = True
        except RuntimeError:
            dt *= 0.5
            dtc.assign(dt)
            if dt < 1e-12:
                raise RuntimeError("Time step collapsed")
    # successful step
    t += dt

    # adapt dt based on iteration count
    if nit <= 5:
        dt = min(dt*1.2, 5e-5)
    elif nit > 10:
        dt = max(dt*0.5, 1e-9)
    dtc.assign(dt)

    # update old solution
    w_old.assign(w)

    # write output at prescribed times
    if t >= t_save - 1e-12:
        c_sol, mu_sol = w.split()
        xdmf.write(c_sol, t)
        xdmf.write(mu_sol, t)
        try:
            t_save = next(next_save)
        except StopIteration:
            t_save = T + 1.0

xdmf.close()
