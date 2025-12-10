from dolfin import *
import mshr

# Geometry
domain = Rectangle(Point(0, 0), Point(1, 0.2))
mesh = mshr.generate_mesh(domain, 20)

# Material
E, nu = 200e9, 0.3
mu, lambda_ = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

# Function space
V = VectorFunctionSpace(mesh, "P", 1)

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
tol = 1e-6

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0, tol)

Left().mark(boundaries, 1)
Right().mark(boundaries, 2)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Dirichlet BCs
bcs = [
    DirichletBC(V, Constant((0.0, 0.0)), boundaries, 1),
    DirichletBC(V, Constant((0.001, 0.0)), boundaries, 2)
]

# Variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = mu * inner(grad(u), grad(v)) + lambda_ * tr(u) * tr(v)
L = Constant((0.0, 0.0))

# Solve
uh = Function(V)
solve(a == L, uh, bcs)

# Save displacement field
plot(uh, title="q1 u_x", mode="displacement")
interactive()
