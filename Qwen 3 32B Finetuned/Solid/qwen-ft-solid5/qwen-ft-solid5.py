from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

# Geometry
outer = Rectangle(Point(0, 0), Point(1.0, 0.20))
notch = Rectangle(Point(0, 0.08), Point(0.06, 0.12))
mesh = generate_mesh(outer - notch, 64)

# Function space
V = VectorFunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(V), TestFunction(V)

# Material
E, nu = 200e9, 0.3
mu, lmbda = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
lmbda_ps = 2*mu*lmbda/(lmbda + 2*mu)
def eps(w): return sym(grad(w))
def sigma(w): return lmbda_ps*tr(eps(w))*Identity(2) + 2*mu*eps(w)

# Boundary conditions
bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)
tol = 1e-6
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0, tol)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1.0, tol)
Left().mark(bdr, 1); Right().mark(bdr, 2)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)
bc_left = DirichletBC(V, Constant((0, 0)), bdr, 1)

# Variational form
T_right = Constant((2e6, 0.0))
a = inner(sigma(u), eps(v))*dx
L = dot(T_right, v)*ds(2)
u_h = Function(V)
solve(a == L, u_h, [bc_left])

# Postprocessing
s = sigma(u_h)
vm = project(sqrt(s[0,0]**2 - s[0,0]*s[1,1] + s[1,1]**2 + 3*s[0,1]**2), FunctionSpace(mesh, "CG", 1))
File("q5_vm.pvd") << vm
XDMFFile("q5_disp.xdmf").write(u_h)
plt.figure(); c = plot(vm, title="von Mises stress"); plt.colorbar(c); plt.savefig("q5_vm.png")