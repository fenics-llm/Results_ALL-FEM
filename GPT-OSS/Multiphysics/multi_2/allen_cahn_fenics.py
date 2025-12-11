# filename: allen_cahn_fenics.py
from dolfin import *
import math

# -------------------------------------------------
# Parameters
# -------------------------------------------------
eps = 0.01          # interface thickness
M   = 1.0           # mobility
dt  = 1.0e-3        # time step
T   = 0.20          # final time
save_times = [0.0, 0.05, 0.10, 0.20]

# -------------------------------------------------
# Mesh and function space
# -------------------------------------------------
nx, ny = 200, 200
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, "CG", 1)

# -------------------------------------------------
# Signed distance to a centered square (side = 0.5)
# -------------------------------------------------
class SignedDistance(UserExpression):
    def eval(self, values, x):
        cx, cy = 0.5, 0.5          # centre of the square
        half_side = 0.25           # half of side length (0.5/2)
        # distance to the axis‑aligned square
        d = max(abs(x[0] - cx), abs(x[1] - cy)) - half_side
        values[0] = d
    def value_shape(self):
        return ()

sd = SignedDistance(degree=2)

# -------------------------------------------------
# Initial condition: φ₀ = tanh( d / (√2·ε) )
# -------------------------------------------------
class InitialPhi(UserExpression):
    def __init__(self, eps, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.sqrt2 = math.sqrt(2.0)
    def eval(self, values, x):
        # signed distance at point x
        cx, cy = 0.5, 0.5
        half_side = 0.25
        d = max(abs(x[0] - cx), abs(x[1] - cy)) - half_side
        values[0] = math.tanh(d / (self.sqrt2 * self.eps))
    def value_shape(self):
        return ()

phi_n = Function(V)                     # φⁿ
phi_n.interpolate(InitialPhi(eps=eps, degree=2))

# -------------------------------------------------
# Variational problem (implicit backward Euler)
# -------------------------------------------------
phi = Function(V)          # unknown φⁿ⁺¹
v   = TestFunction(V)

# Weak form:
#   (φ - φ_n)/dt * v
# + M * [ (1/ε)*(φ³ - φ) * v + ε * ∇φ·∇v ] = 0
F = ((phi - phi_n) / dt) * v * dx \
    + M * ((1.0 / eps) * (phi**3 - phi) * v + eps * dot(grad(phi), grad(v))) * dx

# Jacobian for Newton's method
J = derivative(F, phi)

# -------------------------------------------------
# Nonlinear solver setup
# -------------------------------------------------
problem = NonlinearVariationalProblem(F, phi, J=J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm["newton_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["relative_tolerance"] = 1e-6
prm["newton_solver"]["maximum_iterations"] = 25
# Use a direct solver (MUMPS) which is robust for this indefinite system
prm["newton_solver"]["linear_solver"] = "mumps"

# -------------------------------------------------
# XDMF output
# -------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "phi_output.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["rewrite_function_mesh"] = False

# -------------------------------------------------
# Time stepping
# -------------------------------------------------
t = 0.0
while t <= T + DOLFIN_EPS:
    # Save solution at requested times
    if any(abs(t - ts) < 1e-12 for ts in save_times):
        phi_n.rename("phi", "PhaseField")
        xdmf.write(phi_n, t)
        print(f"Saved solution at t = {t:.3f}")

    # Stop after final time reached
    if t >= T:
        break

    # Use previous solution as initial guess for Newton
    phi.assign(phi_n)

    # Solve the nonlinear problem for φⁿ⁺¹
    solver.solve()

    # Update previous solution
    phi_n.assign(phi)

    # Advance time
    t += dt

xdmf.close()