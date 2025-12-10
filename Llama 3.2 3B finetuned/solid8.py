from dolfin import *
import mshr
import matplotlib.pyplot as plt

# Geometry
mesh = mshr.Rectangle(Point(0, 0), Point(1, 0.2)) + mshr.Rectangle(Point(0, 0), Point(1, 0.2))
mesh = mshr.generate_mesh(mesh, 50)

# Material (plane-stress)
E1 = 40e9
E2 = 10e9
G12 = 5e9
nu12 = 0.25

theta = 30 * pi / 180 # rotate by 30 degrees anticlockwise from x-y axes
Q = as_matrix([[E1, E1*nu12, 0.0],
               [E1*nu12, E2, G12],
               [0.0, G12, G12]])

# Function space
V = VectorFunctionSpace(mesh, "P", 1)

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
tol = 1e-3
class Bottom(SubDomain):
    def inside(self, x, on_b): return on_b and near(x[1], 0.0, tol)
class Top(SubDomain):
    def inside(self, x, on_b): return on_b and near(x[1], 0.2, tol)
Bottom().mark(boundaries, 1); Top().mark(boundaries, 2)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# BCs
u_bottom = Constant((0.0, 0.0))
bcs = [DirichletBC(V, u_bottom, boundaries, 1)]

# Variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant((0.0, -10.0))

a = inner(Q*grad(u), grad(v))*dx
L = dot(f, v)*dx

uh = Function(V); solve(a == L, uh, bcs)

# Output
plt.figure(); p = plot(uh[0], title="q8_ux.png"); plt.colorbar(p)
plt.figure(); q = plot(sqrt(2.0*(uh[1]**2 + uh[0]**2 - 3*uh[0]*uh[1])), title="q8_vm.png"); plt.colorbar(q)

# XDMF
file = File("q8_solution.xdmf")
file << (uh, mesh)
