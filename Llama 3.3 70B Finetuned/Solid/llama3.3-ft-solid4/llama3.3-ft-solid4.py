# filename: q4.py

from dolfin import *
from mshr  import *
import matplotlib.pyplot as plt

Lx, Ly = 1.0, 0.20
a      = 0.04
hole1  = Circle(Point(0.33, 0.10), a)
hole2  = Circle(Point(0.67, 0.10), a)

domain = Rectangle(Point(0,0), Point(Lx, Ly)) - hole1 - hole2
mesh   = generate_mesh(domain, 64)

E, nu = Constant(200e9), Constant(0.3)
mu    = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2*nu))
lmbda = 2*mu*lmbda/(lmbda + 2*mu)

V = VectorFunctionSpace(mesh, "P", 1)
u, v = TrialFunction(V), TestFunction(V)

def eps(w): return 0.5*(grad(w)+grad(w).T)
def sigma(w): return lmbda*tr(eps(w))*Identity(2) + 2*mu*eps(w)

tol = 1e-10
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Left(SubDomain):
    def inside(self,x,on): return on and near(x[0],0,tol)
class Right(SubDomain):
    def inside(self,x,on): return on and near(x[0],Lx,tol)

Left().mark(facets,1); Right().mark(facets,2)
ds = Measure("ds", domain=mesh, subdomain_data=facets)

u_L  = Constant((0.0, 0.0))
bc   = DirichletBC(V, u_L, facets, 1)

t_R  = Constant((2e6, 0.0))
a_form = inner(sigma(u), eps(v))*dx
L_form = dot(t_R, v)*ds(2)

uh = Function(V); solve(a_form == L_form, uh, bc)

s = sigma(uh)
VM = project(sqrt(s[0,0]**2 + s[1,1]**2 - s[0,0]*s[1,1] + 3*s[0,1]**2),
             FunctionSpace(mesh,"P",1))
plot_object = plot(VM, title="von Mises stress")
plt.savefig("q4_vm.png")

file = XDMFFile("q4_u.xdmf"); file.write(uh)
print("[q4.py] Displacement field saved to q4_u.xdmf")

hole_bdry = MeshFunction("size_t", mesh, 1, 0)
class Hole(SubDomain):
    def inside(self,x,on): return on and ((x[0]-0.33)**2+(x[1]-0.10)**2 < a**2 + tol
                                        or (x[0]-0.67)**2+(x[1]-0.10)**2 < a**2 + tol)
Hole().mark(hole_bdry, 1); ds_h = Measure("ds", domain=mesh, subdomain_data=hole_bdry)

VM_max = -1e300
mesh.init(1,0)
for f in SubsetIterator(hole_bdry, 1):
    val = VM(f.midpoint())
    if val > VM_max: VM_max = val
print("[q4.py] Maximum von Mises stress:", VM_max)
Kt     = VM_max / (2e6)
print("[q4.py] Stress Concentration Factor Kt =", Kt)