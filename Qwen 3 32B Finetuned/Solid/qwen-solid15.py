from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt
import numpy as np

# Geometry
rect = Rectangle(Point(0, 0), Point(1.2, 0.2))
holes = [Circle(Point(0.3, 0.1), 0.03),
         Circle(Point(0.6, 0.1), 0.03),
         Circle(Point(0.9, 0.1), 0.03)]
mesh = generate_mesh(rect - holes[0] - holes[1] - holes[2], 64)

# Material
E, nu = 10.0, 0.3
mu = E/(2*(1+nu)); lmbda = E*nu/((1+nu)*(1-2*nu))

# Function space
V = VectorFunctionSpace(mesh, "CG", 2)
u, v = TrialFunction(V), TestFunction(V)
I = Identity(2); F = I + grad(u); E = 0.5*(F.T*F - I)
S = lmbda*tr(E)*I + 2*mu*E

# BCs
bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)
tol = 1e-6
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0, tol)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1.2, tol)
Left().mark(bdr, 1); Right().mark(bdr, 2)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)
bc_left = DirichletBC(V, Constant((0, 0)), bdr, 1)
bc_right = DirichletBC(V, Constant((0.012, 0)), bdr, 2)

# Load stepping
u0 = Function(V); du = Function(V)
problem = NonlinearVariationalProblem(
    inner(S, grad(v))*dx, u0, [bc_left, bc_right],
    J=derivative(inner(S, grad(v))*dx, u0, du))
solver = NonlinearVariationalSolver(problem)
solver.parameters["newton_solver"]["maximum_iterations"] = 50

# Solve
E_max_vals = []
while True:
    solver.solve()
    E_mat = project(E, TensorFunctionSpace(mesh, "CG", 1))
    E_eig = np.linalg.eig(E_mat.compute_vertex_values(mesh).reshape((-1, 2, 2)))
    E_max = Function(FunctionSpace(mesh, "CG", 1))
    E_max.vector()[:] = E_eig[0].max(axis=1)
    E_max_vals.append(max(E_max.vector()))
    if E_max_vals[-1] > 0.03: break

# Postprocess
plot(mesh, title="Deformed"); plt.savefig("q15_def.png")
plot(E_max, title="E_max"); plt.savefig("q15_Emax.png")
s = S - (1/3)*tr(S)*I
vmS = project(sqrt(1.5*inner(s, s)), FunctionSpace(mesh, "CG", 1))
plot(vmS, title="vmS"); plt.savefig("q15_vmS.png")

# Export
File("q15_u.xdmf") << u0
File("q15_Emax.xdmf") << E_max