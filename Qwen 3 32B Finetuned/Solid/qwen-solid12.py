from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt

# Geometry
rect = Rectangle(Point(0, 0), Point(1, 0.2))
circle = Circle(Point(0.5, 0.1), 0.04)
domain = rect - circle
mesh = generate_mesh(domain, 64)

# Function spaces
V = VectorFunctionSpace(mesh, "P", 1)
Q = FunctionSpace(mesh, "P", 1)
W = MixedFunctionSpace(V, Q)

# Variational problem
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
I = Identity(2)
F = I + grad(u)
C = F.T * F
J = det(F)
a = inner(grad(u), grad(v)) * dx - div(v) * p * dx - q * div(u) * dx
L = inner(Constant((0.0, 0.0)), v) * dx

# Boundary conditions
bnd = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, 1e-1)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0, 1e-1)

Left().mark(bnd, 1)
Right().mark(bnd, 2)

bc = DirichletBC(W.sub(0), Constant((0.0, 0.0)), bnd, 1)

# Solve
w = Function(W)
solve(a == L, w, bc)

# Post-processing
u_h, p_h = w.split()

# Plot
plt.figure()
plot(u_h, title="Velocity")
plt.figure()
plot(p_h, title="Pressure")
plt.show()
