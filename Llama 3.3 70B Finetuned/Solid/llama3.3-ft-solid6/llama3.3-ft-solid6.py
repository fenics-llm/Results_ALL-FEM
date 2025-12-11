# filename: q6.py
from dolfin import *
from mshr  import *
import matplotlib.pyplot as plt

a = 0.05
domain = Rectangle(Point(0,0), Point(1,0.2)) - Circle(Point(0.5,0.2), a)
mesh = generate_mesh(domain, 64)

E, nu = 200e9, 0.3
mu, lmbda = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
lmbda = 2*mu*lmbda/(lmbda + 2*mu)
V = VectorFunctionSpace(mesh, "P", 1)
u, v = TrialFunction(V), TestFunction(V)

def eps(w): return sym(grad(w))
def sigma(w): return lmbda*tr(eps(w))*Identity(2) + 2*mu*eps(w)

tol = 1e-10
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Bottom(SubDomain):
    def inside(self,x,on): return on and near(x[1],0,tol)
class Top(SubDomain):
    def inside(self,x,on): return on and near(x[1],0.2,tol) \
        and (x[0] < 0.45-tol or x[0] > 0.55+tol)
class Left(SubDomain):
    def inside(self,x,on): return on and near(x[0],0,tol)
class Right(SubDomain):
    def inside(self,x,on): return on and near(x[0],1,tol)

Bottom().mark(facets,1); Top().mark(facets,2)
Left().mark(facets,3);  Right().mark(facets,4)
ds = Measure("ds", domain=mesh, subdomain_data=facets)

bc = DirichletBC(V, Constant((0.0,0.0)), facets,1)

g_top = Constant((0.0,-10e6))
a_form = inner(sigma(u), eps(v))*dx
L_form = dot(g_top, v)*ds(2)

uh = Function(V); solve(a_form == L_form, uh, bc)

s = sigma(uh)
VM = project(sqrt(s[0,0]**2 + s[1,1]**2 - s[0,0]*s[1,1] + 3*s[0,1]**2),
             FunctionSpace(mesh,"P",1))
plot_object = plot(VM, title="von Mises (plane-stress)")
plt.savefig("q6_vm.png")
XDMFFile("q6_u.xdmf").write(uh)