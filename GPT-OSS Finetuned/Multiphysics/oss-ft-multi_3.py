# filename: cahn_hilliard.py
from dolfin import *
import numpy as np

# ----------------------------------------------------------------------
# 1. Periodic boundary (applied through the function space)
# ----------------------------------------------------------------------
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return bool((near(x[0], 0.0) or near(x[1], 0.0)) and on_boundary)
    def map(self, x, y):
        y[0] = x[0] - 1.0 if near(x[0], 1.0) else x[0]
        y[1] = x[1] - 1.0 if near(x[1], 1.0) else x[1]

pbc = PeriodicBoundary()

# ----------------------------------------------------------------------
# 2. Mesh (periodicity is enforced by the function space)
# ----------------------------------------------------------------------
mesh = UnitSquareMesh(64, 64)

# ----------------------------------------------------------------------
# 3. Mixed Taylor–Hood space (c, μ) with periodic constraints
# ----------------------------------------------------------------------
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = MixedElement([P1, P1])
W  = FunctionSpace(mesh, ME, constrained_domain=pbc)

# ----------------------------------------------------------------------
# 4. Initial concentration: c̄ + uniform perturbation in [−0.05,0.05]
# ----------------------------------------------------------------------
c_bar = 0.63
V_c   = W.sub(0).collapse()
c_n   = Function(V_c)
rng   = np.random.default_rng(42)
c_n.vector()[:] = c_bar + 0.1 * (rng.random(V_c.dim()) - 0.5)
c_n.vector().apply("insert")

# ----------------------------------------------------------------------
# 5. Model parameters (regularised mobility & chemical potential)
# ----------------------------------------------------------------------
theta = 1.5
alpha = 3000.0
eps   = 1e-8                                 # regularisation for log & mobility
Mmob  = lambda c: c * (1.0 - c) + eps       # M(c)=c(1-c)+eps

def mu_c(c):
    # regularised μ_c = (0.5/θ)·ln(c/(1−c)) + 1 − 2c
    return (0.5/theta) * ln((c + eps) / (1.0 - c + eps)) + 1.0 - 2.0*c

# ----------------------------------------------------------------------
# 6. Time stepping control (fixed small step for robustness)
# ----------------------------------------------------------------------
t      = 0.0
T      = 0.04
dt     = 5e-8
save_times = [0.0, 3e-6, 1e-4, 1e-3, 4e-2]
save_idx   = 0

# ----------------------------------------------------------------------
# 7. XDMF output (mesh is shared)
# ----------------------------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "cahn_hilliard.xdmf")
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["flush_output"] = True

# ----------------------------------------------------------------------
# 8. Variational forms (semi‑implicit: mobility & μ_c evaluated at c_n)
# ----------------------------------------------------------------------
(c, mu) = TrialFunctions(W)
(v, q)   = TestFunctions(W)

c_n_fun = Function(V_c)
c_n_fun.assign(c_n)

# Bilinear form a( (c,mu); (v,q) )
a = ((c) / dt) * v * dx \
    + Mmob(c_n_fun) * dot(grad(mu), grad(v)) * dx \
    + mu * q * dx + dot(grad(c), grad(q)) * dx

# Linear form L( (v,q) )
L = (c_n_fun / dt) * v * dx \
    + 3.0 * alpha * mu_c(c_n_fun) * q * dx

# ----------------------------------------------------------------------
# 9. Solver (linear each time step)
# ----------------------------------------------------------------------
w = Function(W)

# ----------------------------------------------------------------------
# 10. Time stepping loop
# ----------------------------------------------------------------------
while t < T - DOLFIN_EPS:
    if t + dt > T:
        dt = T - t

    # Assemble and solve the linear system
    a = ((c) / dt) * v * dx \
        + Mmob(c_n_fun) * dot(grad(mu), grad(v)) * dx \
        + mu * q * dx + dot(grad(c), grad(q)) * dx
    L = (c_n_fun / dt) * v * dx \
        + 3.0 * alpha * mu_c(c_n_fun) * q * dx
    solve(a == L, w, bcs=[])

    c_sol, mu_sol = w.split(deepcopy=True)

    # Clip concentration to (eps,1‑eps) to keep log well‑defined
    c_vals = c_sol.vector().get_local()
    c_vals = np.clip(c_vals, eps, 1.0 - eps)
    c_sol.vector().set_local(c_vals)
    c_sol.vector().apply("insert")

    t += dt
    c_n_fun.assign(c_sol)   # use as previous state for next step

    if save_idx < len(save_times) and t >= save_times[save_idx] - DOLFIN_EPS:
        xdmf.write(c_sol, t)
        xdmf.write(mu_sol, t)
        print(f"Saved fields at t = {t:.6g}")
        save_idx += 1

xdmf.close()
print("Cahn–Hilliard simulation completed.")