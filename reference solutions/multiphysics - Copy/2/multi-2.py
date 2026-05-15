# Allen-Cahn curvature-flow equation on the unit square
# Legacy FEniCS / DOLFIN version

from __future__ import print_function

from dolfin import *
import math


# -----------------------------------------------------------------------------
# 1. Mesh and function space
# -----------------------------------------------------------------------------
N = 200
mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, "Lagrange", 1)


# -----------------------------------------------------------------------------
# 2. Model parameters
# -----------------------------------------------------------------------------
eps_val = 0.01
M_val = 1.0
dt_val = 1.0e-3
T = 0.20
num_steps = int(round(T / dt_val))

eps = Constant(eps_val)
M = Constant(M_val)
dt = Constant(dt_val)


# -----------------------------------------------------------------------------
# 3. Initial condition
#
# Signed Euclidean distance to the surface of the centered square:
# center = (0.5, 0.5), side length = 0.5, half side = 0.25.
# The sign convention is negative inside and positive outside.
# -----------------------------------------------------------------------------
class InitialPhi(UserExpression):
    def __init__(self, eps_value, **kwargs):
        super(InitialPhi, self).__init__(**kwargs)
        self.eps_value = float(eps_value)

    def eval(self, values, x):
        half_side = 0.25

        qx = abs(x[0] - 0.5) - half_side
        qy = abs(x[1] - 0.5) - half_side

        # Euclidean signed distance to an axis-aligned square.
        outside = math.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2)
        inside = min(max(qx, qy), 0.0)
        d_rect = outside + inside

        values[0] = math.tanh(d_rect / (math.sqrt(2.0) * self.eps_value))

    def value_shape(self):
        return ()


phi_n = Function(V)
phi = Function(V)

phi_n.interpolate(InitialPhi(eps_val, degree=4))
phi_n.rename("phi", "phase field")

phi.assign(phi_n)
phi.rename("phi", "phase field")


# -----------------------------------------------------------------------------
# 4. Weak form: fully implicit backward Euler
#
# PDE:
#   phi_t = -M * ((1/eps) * W'(phi) - eps * Laplacian(phi))
#
# with
#   W'(phi) = phi^3 - phi.
#
# Equivalent strong form:
#   phi_t + (M/eps) * W'(phi) - M*eps*Laplacian(phi) = 0.
#
# Weak form after integration by parts:
#   int ((phi - phi_n)/dt) v dx
# + int (M/eps) (phi^3 - phi) v dx
# + int M eps grad(phi).grad(v) dx = 0.
#
# The boundary term vanishes because grad(phi).n = 0.
# -----------------------------------------------------------------------------
v = TestFunction(V)
dphi = TrialFunction(V)

Wprime = phi**3 - phi

F = ((phi - phi_n) / dt) * v * dx \
    + (M / eps) * Wprime * v * dx \
    + M * eps * dot(grad(phi), grad(v)) * dx

J = derivative(F, phi, dphi)

# Homogeneous Neumann boundary conditions are natural here.
bcs = []

problem = NonlinearVariationalProblem(F, phi, bcs, J)
solver = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["absolute_tolerance"] = 1.0e-10
prm["newton_solver"]["relative_tolerance"] = 1.0e-8
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["relaxation_parameter"] = 1.0

try:
    prm["newton_solver"]["error_on_nonconvergence"] = True
except Exception:
    pass


# -----------------------------------------------------------------------------
# 5. Output setup
# -----------------------------------------------------------------------------
output_steps = {
    0: "phi_t0.00.xdmf",
    int(round(0.05 / dt_val)): "phi_t0.05.xdmf",
    int(round(0.10 / dt_val)): "phi_t0.10.xdmf",
    int(round(0.20 / dt_val)): "phi_t0.20.xdmf",
}

output_files = {}

for step, filename in output_steps.items():
    xdmf = XDMFFile(mesh.mpi_comm(), filename)
    xdmf.parameters["flush_output"] = True
    output_files[step] = xdmf


# Write initial field at t = 0.00.
output_files[0].write(phi_n, 0.0)
print("Saved phi_t0.00.xdmf")


# -----------------------------------------------------------------------------
# 6. Time stepping
# -----------------------------------------------------------------------------
for step in range(1, num_steps + 1):
    t = step * dt_val

    # Use previous time step as Newton initial guess.
    phi.assign(phi_n)

    solver.solve()

    if step in output_steps:
        phi.rename("phi", "phase field")
        output_files[step].write(phi, t)
        print("Saved %s at t = %.3f" % (output_steps[step], t))

    phi_n.assign(phi)


for xdmf in output_files.values():
    xdmf.close()

print("Done.")