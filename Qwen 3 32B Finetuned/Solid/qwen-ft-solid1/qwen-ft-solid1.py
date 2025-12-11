from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

mesh = RectangleMesh(Point(0,0), Point(1,0.2), 20, 4)
V = VectorFunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(V), TestFunction(V)

E, nu = 200e9, 0.3
mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))
lmbda = 2*mu*lmbda/(lmbda+2*mu)
def eps(w): return sym(grad(w))
def sigma(w): return lmbda*tr(eps(w))*Identity(2) + 2*mu*eps(w)

bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1.0)
Left().mark(bdr, 1); Right().mark(bdr, 2)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)
bc_left = DirichletBC(V, Constant((0.0, 0.0)), bdr, 1)
bc_right = DirichletBC(V, Constant((0.001, 0.0)), bdr, 2)

a = inner(sigma(u), eps(v))*dx
L = dot(Constant((0.0, 0.0)), v)*dx

u_sol = Function(V)
solve(a == L, u_sol, [bc_left, bc_right])

u_x = u_sol.sub(0)
u_x.rename("u_x", "u_x")
plt.figure(); c = plot(u_x); plt.colorbar(c); plt.savefig("q1_ux.png")
XDMFFile("q1_disp.xdmf").write(u_sol)