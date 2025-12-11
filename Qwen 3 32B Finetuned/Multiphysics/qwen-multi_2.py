from dolfin import *
from mshr import Rectangle, generate_mesh
import numpy as np
import matplotlib.pyplot as plt

# Geometry and mesh
mesh = generate_mesh(Rectangle(Point(0, 0), Point(1, 1)), 200)
V = FunctionSpace(mesh, "CG", 1)

# Parameters
M = 1.0
eps = 0.01
T = 0.20
dt = 1.0e-3
t = 0.0

# Initial condition
def d_rect(x, y):
    center = (0.5, 0.5)
    half_len = 0.25
    dx = np.clip(x - center[0], -half_len, half_len)
    dy = np.clip(y - center[1], -half_len, half_len)
    return np.sqrt(dx**2 + dy**2)

phi_0 = Expression("tanh(d / (sqrt(2)*eps))", d=d_rect, eps=eps, degree=2)
phi = interpolate(phi_0, V)
v = TestFunction(V)

# Variational form
F = (phi - phi_0)/dt * v * dx + M * (eps * inner(grad(phi), grad(v)) * dx - (1/eps)*(phi**3 - phi)*v * dx)
J = derivative(F, phi)

# Time-stepping
phi_sol = Function(V)
t_steps = [0.00, 0.05, 0.10, 0.20]
outputs = {t: XDMFFile(f"phi_{t:.2f}.xdmf") for t in t_steps}

while t <= T:
    solve(F == 0, phi_sol, J=J, solver_parameters={"newton_solver": {"maximum_iterations": 50}})
    phi.assign(phi_sol)
    if np.isclose(t, t_steps[0]):
        outputs[t].write(phi, t)
    elif t > t_steps[0]:
        t_steps.pop(0)
        outputs[t].write(phi, t)
    t += dt

for f in outputs.values():
    f.close()