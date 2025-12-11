from dolfin import *
from mshr import Rectangle, generate_mesh
import numpy as np
import matplotlib.pyplot as plt
import time

# Geometry
mesh = generate_mesh(Rectangle(Point(0, 0), Point(1, 1)), 64)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
U, p = TrialFunctions(W)
v, q = TestFunctions(W)

# Periodic boundaries
class PeriodicX(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)
    def map(self, x, y):
        y[0] = x[0] - 1.0
        y[1] = x[1]
class PeriodicY(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0)
    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] - 1.0
bc_x = PeriodicX()
bc_y = PeriodicY()
bcs = [bc_x, bc_y]

# Initial conditions
u0 = Expression(("sin(2*pi*x[0])*cos(2*pi*x[1])", "-cos(2*pi*x[0])*sin(2*pi*x[1])"), degree=2)
w0 = interpolate(Expression(("sin(2*pi*x[0])*cos(2*pi*x[1])", "-cos(2*pi*x[0])*sin(2*pi*x[1])", "0.0"), W.sub(0).ufl_element()), W)
w = Function(W)
w.assign(w0)

# Parameters
rho = 1.0
nu = 1e-3
dt = 0.001
T = 1.0
t = 0.0

# Variational forms
a = rho*dot((U - w0)/dt, v)*dx + rho*dot(dot(w0, nabla_grad(U[0])), v)*dx + nu*inner(grad(U[0]), grad(v))*dx + q*div(U[0])*dx + div(v)*p*dx
L = rho*dot(w0/dt, v)*dx

# Time-stepping
t_save = [0.0, 0.25, 0.5, 1.0]
u_out = File("velocity.pvd")
while t < T + DOLFIN_EPS:
    solve(a == L, w, bcs)
    t += dt
    if near(t, t_save[0]):
        u_out << w.sub(0)
        t_save.pop(0)
    w0.assign(w)