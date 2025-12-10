from dolfin import *
from mshr   import Circle, Rectangle, generate_mesh
import matplotlib.pyplot as plt

# Geometry: rectangle with circular hole
outer = Rectangle(Point(0, 0), Point(1.0, 0.2))
inner = Circle(Point(0.5, 0.1), 0.04)
mesh = generate_mesh(outer - inner, 64)

E, nu = 5e3, 0.49
mu, lambda_ = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
f = Constant((0.0, 0.0))

# Function space and boundary markers
V = VectorFunctionSpace(mesh, "P", 1)
bmarks = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
bmarks.set_all(0)

class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0, DOLFIN_EPS)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1.0, DOLFIN_EPS)
class Bottom(SubDomain):
    def inside(self, x, on): return on and near(x[1], 0.0, DOLFIN_EPS)
class Top(SubDomain):
    def inside(self, x, on): return on and near(x[1], 0.2, DOLFIN_EPS)

Left().mark(bmarks, 1); Right().mark(bmarks, 2)
Bottom().mark(bmarks, 3); Top().mark(bmarks, 4)
hole = Circle(Point(0.5, 0.1), 0.04)
mesh = mesh.subdomain_subdiv(hole)

# Variational formulation
u, v = TrialFunction(V), TestFunction(V)
a = mu * inner(grad(u), grad(v)) - lambda_ * div(u) * div(v)
L = dot(f, v) * dx

# Boundary conditions and traction-free hole
bc_left  = DirichletBC(V, Constant((0.0, 0.0)), bmarks, 1)
bc_right = DirichletBC(V, Constant((0.001, 0.0)), bmarks, 2)

u_sol = Function(V)
solve(a == L, u_sol, [bc_left, bc_right])

# Von Mises stress
e = sym(grad(u_sol))
W = sqrt(3 * dot(e, e) - 2 * (tr(e)**2))

# Save plots
plt.figure(); c = plot(W); plt.colorbar(c)
plt.savefig("q11_vm.png")
plt.close()

plt.figure(); c = plot(u_sol[0]); plt.colorbar(c)
plt.savefig("q11_ux.png")

# XDMF output
File("q11.xdmf") << u_sol

