from dolfin import *
import mshr

# 1. Geometry
mesh = mshr.Rectangle(Point(0, 0), Point(1, 0.2)) + \
       mshr.Rectangle(Point(0, 0), Point(1, 0.8))
mesh = mesh.subdivide(40)
V = VectorFunctionSpace(mesh, "P", 1)

# 2. Material
E, nu = 200e9, 0.3
mu, lambda_ = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
def eps(u): return sym(grad(u))
def sigma(u): return mu * eps(u) + lambda_ * tr(eps(u)) * Identity(2)

# 3. Boundary markers
bmarks = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bmarks.set_all(0); eps = 1e-6

class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0, eps)
Left().mark(bmarks, 1)

class Top(SubDomain):
    def inside(self, x, on): return on and near(x[1], 0.2, eps)
Top().mark(bmarks, 2)

class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1, eps)
Right().mark(bmarks, 3)

class Bottom(SubDomain):
    def inside(self, x, on): return on and near(x[1], 0, eps)
Bottom().mark(bmarks, 4)

ds = Measure("ds", domain=mesh, subdomain_data=bmarks)

# 4. BCs
u_left = Constant((0, 0))
t_top = Constant((0, -2000))

bc_left = DirichletBC(V, u_left, bmarks, 1)
bc_right = DirichletBC(V, Constant((0, 0)), bmarks, 3)

# 5. Variational
u, v = TrialFunction(V), TestFunction(V)
a = sigma(u) * eps(v) * dx + t_top[1] * ds(2)
L = Constant((0, -2000)) * ds(2)

w = Function(V); solve(a == L, w, [bc_left, bc_right])

# 6. Output
import matplotlib.pyplot as plt
plt.figure(); p = plot(w[1], title="q2_uy.png"); plt.colorbar(p)
plt.savefig("q2_uy.png")
