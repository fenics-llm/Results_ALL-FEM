# filename: q5.py
from dolfin import *
from mshr  import *
import matplotlib.pyplot as plt

E, nu = 200e9, 0.3
mu     = E/(2*(1 + nu))
lmbda  = E*nu/((1 + nu)*(1 - 2*nu))
lmbda  = 2*mu*lmbda/(lmbda + 2*mu)

mesh = generate_mesh(
    Rectangle(Point(0, 0), Point(1, 0.20)) -
    Rectangle(Point(0, 0.08), Point(0.06, 0.12)), 64)

V = VectorFunctionSpace(mesh, "P", 1)
u = TrialFunction(V); v = TestFunction(V)
def eps(w):   return sym(grad(w))
def sigma(w): return lmbda*tr(eps(w))*Identity(2) + 2*mu*eps(w)

tol = 1e-10
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0, tol)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1.0, tol)
class Top(SubDomain):
    def inside(self, x, on): return on and near(x[1], 0.20, tol)
class Bottom(SubDomain):
    def inside(self, x, on): return on and near(x[1], 0, tol)
Left().mark(facets, 1); Right().mark(facets, 2)
Top().mark(facets, 3);   Bottom().mark(facets, 4)

ds = Measure("ds", domain=mesh, subdomain_data=facets)
bc = DirichletBC(V, Constant((0.0, 0.0)), facets, 1)

t_R = Constant((2e6, 0.0))
a = inner(sigma(u), eps(v))*dx
L = dot(t_R, v)*ds(2)

uh = Function(V); solve(a == L, uh, bc)
s = sigma(uh)
vm = project(
    sqrt(s[0,0]**2 + s[1,1]**2 - s[0,0]*s[1,1] + 3*s[0,1]**2),
    FunctionSpace(mesh, "P", 1))
plot(vm, title="σ_vm")
plt.savefig("q5_vm.png")

XDMFFile("q5_u.xdmf").write(uh)