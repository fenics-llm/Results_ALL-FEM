from dolfin import *
from mshr import Rectangle
import matplotlib.pyplot as plt

# Geometry & mesh
mesh = Rectangle(Point(0, 0), Point(1, 1)).generate_mesh(64)
V = VectorFunctionSpace(mesh, "P", 2)
Q = FunctionSpace(mesh, "P", 1)
ME = V * Q
(u, p) = TrialFunctions(ME)
(v, q) = TestFunctions(ME)

# Boundary markers
bdr = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
bdr.set_all(0)
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1.0)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0)
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)
Top().mark(bdr, 1)
Left().mark(bds, 2)
Right().mark(bdr, 3)
Bottom().mark(bdr, 0)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)

# Boundary conditions
noslip = Constant((0.0, 0.0))
lid = Constant((1.0, 0.0))
bcs = [
    DirichletBC(V, noslip, bdr, 2),
    DirichletBC(V, noslip, bdr, 3),
    DirichletBC(V, noslip, bdr, 0),
    DirichletBC(V, lid, bdr, 1)
]

# Variational problem
mu = 1.0
a = (mu * inner(grad(u), grad(v)) - div(v) * p - q * div(u)) * dx
L = dot(Constant((0.0, 0.0)), v) * dx

# Solve
w = Function(ME)
solve(a == L, w, bcs)
u_sol, p_sol = w.split()

# Post-processing
speed = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "P", 1))
File("q3_speed.pvd") << speed
File("q3_solution.xdmf") << u_sol, p_sol