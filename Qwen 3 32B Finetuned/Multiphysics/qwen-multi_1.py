from dolfin import *
from mshr import Circle, generate_mesh
import numpy as np

R0 = 0.05
s = 1.0e-4
D = 1.0e-5
kappa = 1.0e-4
dt = 0.01
T = 10.0

mesh = generate_mesh(Circle(Point(0, 0), R0), 64)
V = FunctionSpace(mesh, "CG", 1)
c = Function(V)
c_old = Function(V)
v = TestFunction(V)

c.interpolate(Constant(1.0))
c_old.assign(c)

t = 0.0
while t <= T:
    w = Expression(("s*x[0]/sqrt(x[0]*x[0]+x[1]*x[1])", 
                    "s*x[1]/sqrt(x[0]*x[0]+x[1]*x[1])"),
                   s=s, degree=2)
    a = (inner((c - c_old)/dt, v) + D*dot(grad(c), grad(v)) 
         + dot(w, grad(c_old))*v - kappa*c*v)*dx
    L = Constant(0.0)*v*dx
    solve(a == L, c)
    mesh.coordinates()[:] += w.eval(mesh.coordinates(), [])*dt
    c_old.assign(c)
    t += dt
