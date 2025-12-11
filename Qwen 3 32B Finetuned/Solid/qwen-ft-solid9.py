from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

# Geometry
mesh = generate_mesh(Rectangle(Point(0, 0), Point(1.0, 0.2)), 100*20)
V = VectorFunctionSpace(mesh, "CG", 1)

# Material
nu = Constant(0.30)
E = Expression("1e11 + 1e11*x[1]/0.2", degree=1)
mu = E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))

# Variational form
def eps(v): return sym(grad(v))
def sigma(u): return lmbda*tr(eps(u))*Identity(2) + 2.0*mu*eps(u)

u = TrialFunction(V)
v = TestFunction(V)
f = Constant((0.0, 0.0))
a = inner(sigma(u), eps(v))*dx
L = dot(Constant((2e6, 0.0)), v)*ds(1)

# Boundary conditions
bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1.0)
Left().mark(bdr, 1)
Right().mark(bdr, 2)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)
bc_left = DirichletBC(V, Constant((0.0, 0.0)), bdr, 1)

# Solve
u_sol = Function(V)
solve(a == L, u_sol, [bc_left])

# Postprocess
u_mag = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "CG", 1))
plt.figure(); p = plot(u_mag, title="|u|"); plt.colorbar(p); plt.savefig("q9_disp.png")
File("q9_disp.xdmf") << u_sol

