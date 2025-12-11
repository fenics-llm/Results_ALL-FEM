from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

mesh = generate_mesh(Rectangle(Point(0,0), Point(1.0,0.20)), 64)
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.10, 1e-6)
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.10, 1e-6)
Top().mark(subdomains, 1)
Bottom().mark(subdomains, 2)
ds = Measure("ds", domain=mesh, subdomain_data=subdomains)

E = Expression("x[1] > 0.10 ? 70e9 : 200e9", degree=0)
nu = Constant(0.30)
mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))

V = VectorFunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(V), TestFunction(V)
def eps(w): return sym(grad(w))
def sigma(w): return lmbda*tr(eps(w))*Identity(2) + 2*mu*eps(w)

a = inner(sigma(u), eps(v))*dx
L = dot(Constant((0, -5000.0)), v)*ds(2)

bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, 1e-6)
Left().mark(bdr, 1)
bc_left = DirichletBC(V, Constant((0.0, 0.0)), bdr, 1)

u_sol = Function(V)
solve(a == L, u_sol, [bc_left])

u_mag = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "CG", 1))
plt.figure(); p = plot(u_mag); plt.colorbar(p); plt.savefig("q7_disp.png")
File("q7_disp.pvd") << u_sol