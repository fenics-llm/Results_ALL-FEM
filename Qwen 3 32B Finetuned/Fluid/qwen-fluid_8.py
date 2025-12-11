from dolfin import *
from mshr import Rectangle, generate_mesh
import numpy as np

# Geometry and mesh
mesh = generate_mesh(Rectangle(Point(0, 0), Point(1, 0.2)), 128*32)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Parameters
rho, mu, G = 1.0, 0.01, 1.0
f = Constant((G, 0.0))

# Periodic BCs
class PeriodicBC(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)
    def map(self, x, y):
        y[0] = x[0] - 1.0; y[1] = x[1]
bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bndry.set_all(0)
class Y0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0)
class YH(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.2)
Y0().mark(bndry, 1); YH().mark(bndry, 2)
ds = Measure("ds", domain=mesh, subdomain_data=bndry)

# Variational form
a = (rho*inner(grad(u)*u, v) + mu*inner(grad(u), grad(v)) - p*div(v) - q*div(u))*dx
L = rho*dot(f, v)*dx

# Boundary conditions
noslip = Constant((0.0, 0.0))
bc_y0 = DirichletBC(W.sub(0), noslip, bndry, 1)
bc_yH = DirichletBC(W.sub(0), noslip, bndry, 2)
bc_per = PeriodicBC()
bc_per.apply(a); bc_per.apply(L)
bc_p0 = DirichletBC(W.sub(1), Constant(0.0), Point(0.0, 0.0))

# Solve
w = Function(W)
solve(a == L, w, [bc_y0, bc_yH, bc_p0])
u_sol, p_sol = w.split(deepcopy=True)

# Save
with XDMFFile("q9_soln.xdmf") as file:
    file.write(u_sol, 0.0)
    file.write(p_sol, 0.0)
