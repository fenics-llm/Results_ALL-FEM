from dolfin import *
from mshr import Rectangle
from mshr import generate_mesh
import matplotlib.pyplot as plt

# Define geometry and mesh
L, H = 2.0, 0.2
domain = Rectangle(Point(0, 0), Point(L, H))
mesh = generate_mesh(domain, 64)

# Define function space
V = VectorFunctionSpace(mesh, "P", 1)
Q = FunctionSpace(mesh, "P", 1)
W = MixedFunctionSpace([V, Q])

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0.0, 0.0))
mu = Constant(1.0)
a = mu * inner(grad(u), grad(v)) * dx + div(u) * q * dx + p * div(v) * dx
L = inner(f, v) * dx

# Define boundary conditions
bcs = [DirichletBC(W.sub(0), Constant((0.0, 0.0), V), "on_boundary")]

# Solve
w = Function(W)
solve(a == L, w, bcs)

# Postprocess
u, p = w.split()
plot(u, title="Velocity")
plt.show()