# filename: q1.py

from dolfin import *
import matplotlib.pyplot as plt

# Geometry
Lx, Ly = 1.0, 0.20
mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), 20, 4)

# Material
E, nu = Constant(200e9), Constant(0.30)
mu    = E / (2*(1 + nu))
lmbda = E * nu / ((1 + nu)*(1 - 2*nu))
lmbda = 2*mu*lmbda/(lmbda + 2*mu)

def eps(u):   return 0.5*(grad(u) + grad(u).T)
def sigma(u): return lmbda*tr(eps(u))*Identity(2) + 2*mu*eps(u)

# Function space
V = VectorFunctionSpace(mesh, "P", 1)
u = TrialFunction(V); v = TestFunction(V)

# Boundary markers
tol = 1e-10
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Left(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], Lx, tol)

Left().mark(facets, 1); Right().mark(facets, 2)
ds = Measure("ds", domain=mesh, subdomain_data=facets)

# Boundary conditions
bc_left   = DirichletBC(V, Constant((0.0, 0.0)), facets, 1)
u_right   = Expression(("0.001", "0.0"), degree=1)
bc_right  = DirichletBC(V, u_right, facets, 2)

# Variational problem
a = inner(sigma(u), eps(v))*dx
L = dot(Constant((0.0, 0.0)), v)*ds

uh = Function(V)
solve(a == L, uh, [bc_left, bc_right])

# Visualization
ux_mag = project(uh[0], FunctionSpace(mesh,"P",1))
plot_object = plot(ux_mag, title="u_x (m)")
plt.savefig("q1_ux.png")
plt.close()

# Output
file = XDMFFile("q1_u.xdmf"); file.write(uh)