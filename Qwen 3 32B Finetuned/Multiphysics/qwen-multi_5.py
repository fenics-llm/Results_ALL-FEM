from dolfin import *
from mshr import Rectangle, generate_mesh
import numpy as np
import matplotlib.pyplot as plt

# Geometry
mesh = generate_mesh(Rectangle(Point(0, -0.1), Point(6, 1.1)), 64)
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)
class Fluid(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-0.1, 1.1))
class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (1.0, 1.1))
class Lower(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (-0.1, 0.0))
Fluid().mark(subdomains, 1)
Upper().mark(subdomains, 2)
Lower().mark(subdomains, 3)
ds = Measure("ds", domain=mesh, subdomain_data=subdomains)

# Function spaces
Vf = VectorFunctionSpace(mesh, "CG", 2)
Qf = FunctionSpace(mesh, "CG", 1)
Wf = MixedFunctionSpace([Vf, Qf])
Vs = VectorFunctionSpace(mesh, "CG", 2)
Qs = FunctionSpace(mesh, "CG", 1)
Ws = MixedFunctionSpace([Vs, Qs])

# Parameters
mu_f = Constant(0.003)
rho_f = Constant(1.0)
E_s = Constant(3.0e5)
nu_s = Constant(0.49)
rho_s = Constant(1.1)
dt = 1.0e-4
T = 0.1
t = 0.0

# Variational forms
(u, p) = TrialFunctions(Wf)
(v, q) = TestFunctions(Wf)
(w, r) = TrialFunctions(Ws)
(s, t) = TestFunctions(Ws)
n = FacetNormal(mesh)
f = Constant((0.0, 0.0))
a_f = rho_f*dot((u - u_n)/dt, v)*dx(1) + rho_f*dot(dot(u_n, nabla_grad(u)), v)*dx(1) \
    + inner(2*mu_f*sym(grad(u)), sym(grad(v)))*dx(1) - div(v)*p*dx(1) - q*div(u)*dx(1)
L_f = rho_f*dot(u_n/dt, v)*dx(1) + dot(f, v)*dx(1)
a_s = rho_s*dot(w, s)*dx(2) + rho_s*dot(w, s)*dx(3) + inner(sigma(w), grad(s))*dx(2) + inner(sigma(w), grad(s))*dx(3) \
    - r*div(w)*dx(2) - r*div(w)*dx(3)
L_s = rho_s*dot(w_n/dt, s)*dx(2) + rho_s*dot(w_n/dt, s)*dx(3)

# Boundary conditions
inlet_traction = Expression(("t < 0.005 ? -1e4/2*(1 - cos(pi*t/0.0025)) : 0.0", "0.0"), t=0.0, degree=2)
bc_inlet = DirichletBC(Wf.sub(0).sub(0), inlet_traction, "near(x[0], 0.0)")
bc_outlet = DirichletBC(Wf.sub(0).sub(0), Constant(0.0), "near(x[0], 6.0)")
bc_upper = DirichletBC(Wf.sub(0), w_upper, "near(x[1], 1.0)")
bc_lower = DirichletBC(Wf.sub(0), w_lower, "near(x[1], 0.0)")

# Time loop
u_n = interpolate(Constant((0.0, 0.0)), Vf)
p_n = interpolate(Constant(0.0), Qf)
w_n = interpolate(Constant((0.0, 0.0)), Vs)
r_n = interpolate(Constant(0.0), Qs)
while t <= T:
    inlet_traction.t = t
    solve(a_f == L_f, w, [bc_inlet, bc_outlet, bc_upper, bc_lower])
    solve(a_s == L_s, w, [])
    t += dt
    if t in [0.005, 0.1]:
        File("velocity_{}.pvd".format(t)).write(w.sub(0))
        File("displacement_{}.pvd".format(t)).write(w.sub(0))