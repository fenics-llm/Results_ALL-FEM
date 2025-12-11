from dolfin import *
import numpy as np

# Geometry
mesh = RectangleMesh(Point(0, 0), Point(1, 1), 200, 200)

# Model
V = FunctionSpace(mesh, "CG", 2)
phi_old = interpolate(Expression("tanh((sqrt(pow(x[0]-0.5,2)+pow(x[1]-0.5,2))-0.25)/(sqrt(2)*epsilon))",
                                 degree=3, epsilon=0.01), V)

# Parameters
T_end = 0.20
dt     = 1e-3
M      = 1.0
eps    = Constant(0.01)
t      = 0.0

v = TestFunction(V)

while t < T_end + DOLFIN_EPS:
    phi_new = Function(V)
    F       = (phi_new - phi_old) / dt * v * dx \
              + M * ((1/eps) * (pow(phi_new,3)-phi_new) * v * dx
                     + eps * dot(grad(phi_new), grad(v)) * dx)

    a = lhs(F)
    L = rhs(F)
    solve(a == L, phi_new)
    t      += dt
    if abs(t - 0.00) < DOLFIN_EPS or \
       abs(t - 0.05) < DOLFIN_EPS or \
       abs(t - 0.10) < DOLFIN_EPS or \
       abs(t - 0.20) < DOLFIN_EPS:
        file = XDMFFile("ac_t%.2f.xdmf" % t)
        phi_new.rename("phi", "phase_field")
        file.write(phi_new, t)

    phi_old.assign(phi_new)