# Allen–Cahn (curvature-flow form) in legacy FEniCS (dolfin)
# ∂φ/∂t = -M * ( (1/ε) W'(φ) - ε ∇²φ ),    W(φ) = 0.25*(φ^2 - 1)^2  ⇒  W'(φ) = φ(φ^2 - 1)

from __future__ import print_function
from dolfin import *
import math

# --- Parameters ---
eps = 0.01        # interface thickness ε
M   = 1.0         # mobility
dt  = 1.0e-3      # time-step
T   = 0.20        # final time
nx = ny = 200     # 200 x 200 structured mesh

# Times to save
save_times = [0.00, 0.05, 0.10, 0.20]
save_tol   = 5e-7  # numerical tolerance for time matching

# --- Mesh and space ---
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, "CG", 1)

# --- Signed distance to a centred square and initial phase field ---
class PhiInit(UserExpression):
    def __init__(self, eps, cx=0.5, cy=0.5, half=0.25, **kwargs):
        super().__init__(**kwargs)
        self.eps  = float(eps)
        self.cx   = float(cx)
        self.cy   = float(cy)
        self.half = float(half)  # half side-length (0.5 / 2)

    def eval(self, values, x):
        # Rectangle SDF (axis-aligned), signed: negative inside, positive outside
        qx = abs(x[0] - self.cx) - self.half
        qy = abs(x[1] - self.cy) - self.half
        outside = math.hypot(max(qx, 0.0), max(qy, 0.0))
        inside  = min(max(qx, qy), 0.0)
        d = outside + inside
        values[0] = math.tanh(d / (math.sqrt(2.0) * self.eps))

    def value_shape(self):
        return ()

phi_n = Function(V)
phi_n.assign(project(PhiInit(eps, degree=1), V))  # t = 0 initial condition

# --- Variational problem for implicit step (Newton) ---
phi = Function(V)           # unknown at t^{n+1}
v   = TestFunction(V)
dphi = TrialFunction(V)     # for Jacobian

# W'(φ) = φ(φ^2 - 1)
def Wprime(u):
    return u * (u*u - 1.0)

F = ((phi - phi_n)/dt)*v*dx \
    + (M/eps)*Wprime(phi)*v*dx \
    + (M*eps)*dot(grad(phi), grad(v))*dx

J = derivative(F, phi, dphi)

# --- Output (XDMF) ---
xdmf = XDMFFile(MPI.comm_world, "phi_allen_cahn.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["rewrite_function_mesh"] = False

# Save at t = 0.00
t = 0.0
if any(abs(t - ts) <= save_tol for ts in save_times):
    xdmf.write(phi_n, t)

# --- Time-stepping ---
solver_params = {
    "newton_solver": {
        "relative_tolerance": 1e-8,
        "absolute_tolerance": 1e-10,
        "maximum_iterations": 30,
        "linear_solver": "mumps" if has_petsc() else "lu",
    }
}

while t < T - 0.5*dt:
    t += dt
    # Use previous solution as initial guess
    phi.assign(phi_n)

    # Solve the nonlinear system
    solve(F == 0, phi, bcs=[], J=J, solver_parameters=solver_params)

    # Save only at requested times (within tolerance)
    if any(abs(t - ts) <= save_tol for ts in save_times):
        xdmf.write(phi, t)

    # Advance
    phi_n.assign(phi)

# Ensure the last requested time (T=0.20) is saved even if rounding nicks it
if all(abs(T - ts) > save_tol for ts in save_times) is False:
    if abs(t - T) > save_tol:
        # If loop missed T due to floating error, write the final state with time T
        xdmf.write(phi_n, T)

print("Finished. Output written to 'phi_allen_cahn.xdmf'.")