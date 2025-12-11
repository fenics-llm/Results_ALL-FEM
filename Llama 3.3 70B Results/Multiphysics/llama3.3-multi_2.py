# filename: solve_pde.py
from fenics import *
import math

# Create mesh
mesh = UnitSquareMesh(200, 200)

# Define function space
V = FunctionSpace(mesh, 'P', 1)

# Define parameters
T = 0.20         # final time
epsilon = 0.01   # interface thickness
M = 1.0          # mobility
dt = 0.001       # time step

# Initialize phase field using a UserExpression to avoid compilation error
class InitialCondition(UserExpression):
    def __init__(self, epsilon, **kwargs):
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def eval(self, values, x):
        d_rect = min(min(x[0] - 0.25, 0.75 - x[0]), min(x[1] - 0.25, 0.75 - x[1]))
        values[0] = math.tanh(d_rect / (math.sqrt(2) * self.epsilon))

u_n = interpolate(InitialCondition(epsilon, degree=1), V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
F = (u - u_n)/dt*v*dx + M*((1/epsilon)*(u**3 - u))*v*dx + epsilon*dot(grad(u), grad(v))*dx

# Create XDMF file for output
xdmffile_u = XDMFFile('phase_field.xdmf')

# Time-stepping
t = 0
while t <= T:
    # Compute solution
    u = Function(V)
    solve(F == 0, u)

    # Save solution to file
    xdmffile_u.write(u, t)

    # Update previous solution and time
    u_n.assign(u)
    t += dt

# Close XDMF file
xdmffile_u.close()