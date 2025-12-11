from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt

rect = Rectangle(Point(0,0), Point(1.0,0.2))
notch = Circle(Point(0.5,0.2), 0.05, 100)
mesh = generate_mesh(rect - notch, 64)

E, nu = 200e9, 0.3
mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))
lmbda_ps = 2*mu*lmbda/(lmbda + 2*mu)
def eps(u): return sym(grad(u))
def sigma(u): return lmbda_ps*tr(eps(u))*Identity(2)+2*mu*eps(u)

V = VectorFunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(V), TestFunction(V)

bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)
class Bottom(SubDomain):
    def inside(self, x, on): return on and near(x[1],0.0)
class Top(SubDomain):
    def inside(self, x, on): return on and near(x[1],0.2) and (x[0]<0.45 or x[0]>0.55)
Bottom().mark(bdr,1)
Top().mark(bdr,2)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)
bc_bottom = DirichletBC(V, Constant((0.0,0.0)), bdr, 1)

a = inner(sigma(u), eps(v))*dx
L = dot(Constant((0.0,-10e6)), v)*ds(2)

u_sol = Function(V)
solve(a == L, u_sol, bc_bottom)

s = sigma(u_sol)
vm = sqrt(s[0,0]**2 - s[0,0]*s[1,1] + s[1,1]**2 + 3*s[0,1]**2)
VM = project(vm, FunctionSpace(mesh, "CG", 1))

XDMFFile("q6_vm.xdmf").write(VM)
XDMFFile("q6_u.xdmf").write(u_sol)
plt.figure(); c=plot(VM); plt.colorbar(c); plt.savefig("q6_vm.png")