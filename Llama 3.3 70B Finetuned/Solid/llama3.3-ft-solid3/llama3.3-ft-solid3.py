# filename: q3.py
from dolfin import *
from mshr   import *
import matplotlib.pyplot as plt

E, nu = 200e9, 0.30
mu    = E/(2*(1 + nu))
lmbda = E*nu/((1 + nu)*(1 - 2*nu))
lmbda = 2*mu*lmbda/(lmbda + 2*mu)

mesh = generate_mesh(
    Rectangle(Point(0, 0), Point(1, 0.20)) -
    Circle(Point(0.5, 0.10), 0.05),
    90)

V = VectorFunctionSpace(mesh, "P", 1)
u, v = TrialFunction(V), TestFunction(V)
def eps(w):   return sym(grad(w))
def sigma(w): return lmbda*tr(eps(w))*Identity(2) + 2*mu*eps(w)

tol = 1e-10
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Left(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 1.0, tol)
class Hole(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near((x[0]-0.5)**2 + (x[1]-0.1)**2, 0.05**2, tol)

Left().mark(facets, 1); Right().mark(facets, 2); Hole().mark(facets, 3)
ds = Measure("ds", domain=mesh, subdomain_data=facets)

u_L = Constant((0.0, 0.0))
bc = DirichletBC(V, u_L, facets, 1)

t_R = Constant((2e6, 0.0))
a = inner(sigma(u), eps(v))*dx
L = dot(t_R, v)*ds(2)

uh = Function(V); solve(a == L, uh, bc)

s = sigma(uh)
VM = project(
    sqrt(s[0, 0]**2 + s[1, 1]**2 - s[0, 0]*s[1, 1] + 3*s[0, 1]**2),
    FunctionSpace(mesh, "P", 1),
    solver_type="cg",
    preconditioner_type="ilu")

plot(VM, title="q3: von Mises")
plt.savefig("q3_vm.png")
plt.close()

VM_max = -1e300
mesh.init(1, 0)
for f in SubsetIterator(facets, 3):
    val = VM(f.midpoint())
    if val > VM_max: VM_max = val
Kt     = VM_max / 2e6
print("σ_vm,max =", VM_max)
print("Kt       =", Kt)

file = XDMFFile("q3_displacement.xdmf"); file.write(uh)