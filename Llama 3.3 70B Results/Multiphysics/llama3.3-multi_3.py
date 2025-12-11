from fenics import *
import numpy as np

# Parameters
theta = 1.5
alpha = 3000
T = 0.04
c_bar = 0.63

# Create mesh and function space
mesh = UnitSquareMesh(100, 100)
V = FunctionSpace(mesh, 'P', 1)

# Define functions
c = Function(V)
mu = Function(V)
v = TestFunction(V)

# Initialize with random perturbation
np.random.seed(0)
x = mesh.coordinates()
r = np.random.uniform(-0.05, 0.05, size=len(x))

class InitialCondition(UserExpression):
    def __init__(self, c_bar, r, **kwargs):
        self.c_bar = c_bar
        self.r = r
        super().__init__(**kwargs)

    def eval(self, values, x):
        index = np.argmin(np.linalg.norm(x - mesh.coordinates(), axis=1))
        values[0] = self.c_bar + self.r[index]

c_init = InitialCondition(c_bar, r, degree=1)
c.interpolate(c_init)

# Define variational form for chemical potential
mu_c = (0.5 / theta) * ln(c / (1 - c)) + 1 - 2 * c
a_mu = inner(grad(mu), grad(v)) * dx + alpha * mu_c * v * dx

# Define variational form for concentration
M = c * (1 - c)
F_c = (c - c_init) / Constant(1e-7) * v * dx + M * dot(grad(mu), grad(v)) * dx

# Create XDMF file for output
xdmffile_c = XDMFFile('cahn_hilliard/c.xdmf')
xdmffile_mu = XDMFFile('cahn_hilliard/mu.xdmf')

# Time-stepping parameters
dt = 1e-7
t = 0

while t < T:
    # Update previous solution
    c_prev = Function(V)
    c_prev.assign(c)

    # Solve for chemical potential
    solve(a_mu == 0, mu)

    # Solve for concentration using backward Euler scheme
    solve(F_c == 0, c)

    # Check for convergence and adjust time step
    if abs(assemble((c - c_prev) * v * dx)) < 1e-6:
        dt *= 2
    else:
        dt /= 2

    # Write solution to file at specific times
    if np.isclose(t, [0, 3e-6, 1e-4, 1e-3, 4e-2], atol=dt).any():
        xdmffile_c.write(c, t)
        xdmffile_mu.write(mu, t)

    # Update time
    t += dt

# Close XDMF files
xdmffile_c.close()
xdmffile_mu.close()
