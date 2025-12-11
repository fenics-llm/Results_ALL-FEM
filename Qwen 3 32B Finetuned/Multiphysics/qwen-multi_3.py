from dolfin import *
from mshr import Rectangle, generate_mesh
import numpy as np
import matplotlib.pyplot as plt
import time

# Geometry
mesh = generate_mesh(Rectangle(Point(0, 0), Point(1, 1)), 64)

# Mixed space
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
w = Function(W)
c, mu = split(w)
v, q = TestFunctions(W)

# Parameters
theta = 1.5
alpha = 3000.0
T = 0.04
dt = 1e-7
t = 0.0
c_bar = 0.63
r = Expression("A + (B - A)*x[0]", A=-0.05, B=0.05, degree=1)

# Initial condition
w0 = Function(W)
c0, mu0 = w0.split()
c0.interpolate(Expression("c_bar + r", c_bar=c_bar, r=r, degree=1))
mu0.interpolate(Constant(0.0))

# Variational form
F = (c - c0)/dt*v*dx + dot(grad(mu), grad(v))*dx \
    + alpha*(3*mu - (1/theta)*ln(c/(1 - c)) - 1 + 2*c)*q*dx \
    + dot(grad(c), grad(q))*dx

# Boundary conditions
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 1) or near(x[1], 1))
    def map(self, x, y):
        if near(x[0], 1):
            y[0] = x[0] - 1
        else:
            y[0] = x[0]
        if near(x[1], 1):
            y[1] = x[1] - 1
        else:
            y[1] = x[1]

pb = PeriodicBoundary()
bc = DirichletBC(W.sub(0), Constant((0.0, 0.0)), pb)

# Time-stepping
t_steps = [0, 3e-6, 1e-4, 1e-3, 4e-2]
outputs = [w0]
while t < T:
    t += dt
    solve(F == 0, w, bc, solver_parameters={"newton_solver": {"maximum_iterations": 10}})
    if t in t_steps:
        outputs.append(w.copy())
    if t > 3e-6:
        dt = min(5*dt, 1e-4)

# Output
file = XDMFFile("cahn_hilliard.xdmf")
file.parameters["flush_output"] = True
for w_out in outputs:
    file.write(w_out)
file.close()
