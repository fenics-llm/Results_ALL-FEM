# filename: allen_cahn_curvature_flow.py
from dolfin import *
import numpy as np

# Parameters
epsilon = 0.01
M = 1.0
dt = 1.0e-3
T = 0.20
output_times = [0.0, 0.05, 0.10, 0.20]

# Mesh
nx, ny = 200, 200
mesh = UnitSquareMesh(nx, ny)

# Function space (continuous Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# Signed distance to a centered square of side 0.5
class SignedDistRect(UserExpression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cx = 0.5
        self.cy = 0.5
        self.half = 0.25  # half side length

    def eval(self, values, x):
        dx = abs(x[0] - self.cx) - self.half
        dy = abs(x[1] - self.cy) - self.half
        # inside rectangle: max(dx, dy) <= 0
        if dx <= 0.0 and dy <= 0.0:
            values[0] = max(dx, dy)
        else:
            # outside: distance to rectangle
            values[0] = np.sqrt(max(dx, 0.0)**2 + max(dy, 0.0)**2)

    def value_shape(self):
        return ()

d_rect = SignedDistRect(degree=2)

# Initial condition phi0 = tanh(d_rect / (sqrt(2)*epsilon))
phi_n = Function(V)
phi_n_expr = Expression("tanh(d/(sqrt(2)*eps))",
                         d=d_rect, eps=epsilon, degree=2)
phi_n.interpolate(phi_n_expr)

# Define trial and test functions
phi = Function(V)          # unknown at new time level
v = TestFunction(V)

# Double-well derivative W'(phi) = phi*(phi**2 - 1)
def dW(phi):
    return phi*(phi**2 - 1.0)

# Weak form (implicit Euler)
F = (phi - phi_n)/dt * v * dx \
    + M * ( (1.0/epsilon) * dW(phi) * v + epsilon * dot(grad(phi), grad(v)) ) * dx

# Jacobian
J = derivative(F, phi)

# Solver settings
solver_parameters = {"newton_solver":
                     {"relative_tolerance": 1e-6,
                      "absolute_tolerance": 1e-8,
                      "maximum_iterations": 25,
                      "linear_solver": "lu"}}

# Time stepping
t = 0.0
output_counter = 0
xdmf = XDMFFile(mesh.mpi_comm(), "phi.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

while t <= T + DOLFIN_EPS:
    # Save at requested times
    if np.isclose(t, output_times[output_counter], atol=dt/2):
        xdmf.write(phi_n, t)
        output_counter = min(output_counter + 1, len(output_times)-1)

    # Solve nonlinear problem for phi at t+dt
    solve(F == 0, phi, J=J, solver_parameters=solver_parameters)

    # Update for next step
    phi_n.assign(phi)
    t += dt

# Ensure final state is written
xdmf.write(phi_n, T)
xdmf.close()