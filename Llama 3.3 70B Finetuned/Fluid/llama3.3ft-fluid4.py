from dolfin import *
import matplotlib.pyplot as plt

# Geometry and mesh
L, H = 2.0, 0.20
mesh = RectangleMesh(Point(0, 0), Point(L, H), 160, 16)

# Function space
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Boundary markers
class Inlet(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0)
class Outlet(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], L)
class Walls(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and (near(x[1], 0) or near(x[1], H))
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
Inlet().mark(boundaries, 1); Outlet().mark(boundaries, 2); Walls().mark(boundaries, 3)

# Boundary conditions
Ubar = 2.5
u_in = Expression(("6*ubar*x[1]*(H-x[1])/(H*H)", "0"), ubar=Ubar, H=H, degree=2)
noslip = Constant((0, 0))
bcs = [
    DirichletBC(W.sub(0), u_in, boundaries, 1),
    DirichletBC(W.sub(0), noslip, boundaries, 3)
]

# Parameters
mu = Constant(0.01); rho = Constant(1)

# Variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0, 0))
a = (rho * inner(dot(u, nabla_grad(u)), v) + mu * inner(grad(u), grad(v)) - div(v) * p + q * div(u)) * dx
L = inner(f, v) * dx

# Solve
w = Function(W)
solve(a == L, w, bcs)

# Extract components
u_soln, p_soln = w.split()

# Visualization and saving
plot_object = plot(u_soln[0], title="u_x")
plt.colorbar(plot_object); plt.savefig("q4_ux.png"); plt.close()
file = XDMFFile("q4_soln.xdmf")
file.parameters["flush_output"] = True; file.parameters["functions_share_mesh"] = True
file.write(u_soln, 0.0); file.write(p_soln, 0.0)