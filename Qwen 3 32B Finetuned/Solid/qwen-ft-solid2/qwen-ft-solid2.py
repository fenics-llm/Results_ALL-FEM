from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

# Geometry
mesh = RectangleMesh(Point(0, 0), Point(1.0, 0.2), 40, 8)
V = VectorFunctionSpace(mesh, "CG", 1)

# Material
E, nu = 200e9, 0.3
mu = E / (2*(1+nu))
lmbda = E*nu / ((1+nu)*(1-2*nu))
lmbda_plane = 2*lmbda*mu / (lmbda + 2*mu)

# Variational
def eps(u): return sym(grad(u))
def sigma(u): return lmbda_plane*tr(eps(u))*Identity(2) + 2*mu*eps(u)

u, v = TrialFunction(V), TestFunction(V)
a = inner(sigma(u), eps(v))*dx

# Boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0)
class Top(SubDomain):
    def inside(self, x, on): return on and near(x[1], 0.2)
Left().mark(boundaries, 1)
Top().mark(boundaries, 2)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

bc_left = DirichletBC(V, Constant((0.0, 0.0)), boundaries, 1)
L = dot(Constant((0.0, -2000.0)), v)*ds(2)

# Solve
u_sol = Function(V)
solve(a == L, u_sol, [bc_left])

# Output
plt.figure(); p = plot(u_sol.sub(1)); plt.colorbar(p); plt.savefig("q2_uy.png")
with XDMFFile("q2_disp.xdmf") as file:
    file.write(u_sol)