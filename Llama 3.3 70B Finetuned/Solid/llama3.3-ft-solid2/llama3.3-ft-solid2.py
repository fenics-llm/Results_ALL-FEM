from dolfin import *
import matplotlib.pyplot as plt

# Geometry and mesh
Lx, Ly = 1.0, 0.20
nx, ny = 40, 8
mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), nx, ny)

# Function space
V = VectorFunctionSpace(mesh, "P", 1)
u = TrialFunction(V)
v = TestFunction(V)

# Material parameters
E, nu = 200e9, 0.30
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
lmbda = 2*mu*lmbda/(lmbda + 2*mu)

# Stress and strain
def eps(u):
    return 0.5 * (grad(u) + grad(u).T)
def sigma(u):
    return lmbda * tr(eps(u)) * Identity(2) + 2 * mu * eps(u)

# Boundary conditions
tol = 1e-10

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)
left = Left()
bc_left = DirichletBC(V, Constant((0.0, 0.0)), left)

# Top traction
t_top = Constant((0.0, -2000))

# Variational problem
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], Ly, tol)
top = Top()
top.mark(facets, 3)
ds = Measure("ds", domain=mesh, subdomain_data=facets)
a = inner(sigma(u), eps(v)) * dx
L = dot(t_top, v) * ds(3)
u_sol = Function(V)
solve(a == L, u_sol, bc_left)

# Output
file = XDMFFile("q2_u.xdmf")
file.write(u_sol)
plot(u_sol.sub(1), title="Problem 2: uy displacement")
plt.savefig('q2_uy.png')
plt.close()