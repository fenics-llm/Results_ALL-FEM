from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

# Geometry
mesh = generate_mesh(Rectangle(Point(0,0), Point(1,1)), 64)

# Parameters
rho = 1.0
mu = 1.5e-5
alpha = 2.1e-5
g_beta = 3.15e-5
T_ref = 0.5

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p, T) = TrialFunctions(W)
(v, q, s) = TestFunctions(W)

# Boundaries
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
facets.set_all(0)
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1.0)
class TopBot(SubDomain):
    def inside(self, x, on): return on and (near(x[1], 0.0) or near(x[1], 1.0))
Left().mark(facets, 1)
Right().mark(facets, 2)
TopBot().mark(facets, 3)
ds = Measure("ds", domain=mesh, subdomain_data=facets)

# BCs
noslip = Constant((0.0, 0.0))
bc_u_left = DirichletBC(W.sub(0), noslip, facets, 1)
bc_u_right = DirichletBC(W.sub(0), noslip, facets, 2)
bc_u_topbot = DirichletBC(W.sub(0), noslip, facets, 3)
bc_T_left = DirichletBC(W.sub(2), Constant(1.0), facets, 1)
bc_T_right = DirichletBC(W.sub(2), Constant(0.0), facets, 2)
bcs = [bc_u_left, bc_u_right, bc_u_topbot, bc_T_left, bc_T_right]

# Variational forms
f = Constant((0.0, rho*g_beta*(T-T_ref)))
a = (mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u)
     + alpha*dot(u, grad(T))*s + alpha*dot(grad(T), grad(s)))
L = inner(f, v)*dx

# Solve
w = Function(W)
solve(a == L, w, bcs, solver_parameters={"newton_solver":{"relative_tolerance":1e-6}})
u_sol, p_sol, T_sol = w.split()

# Postprocess
avg_Nu = assemble(-alpha*dot(grad(T_sol), Constant((1.0,0.0)))*ds(1))/assemble(Constant(1.0)*ds(1))
print("Average Nusselt number (left wall):", avg_Nu)

# Save
with XDMFFile("q11_solution.xdmf") as file:
    file.write(u_sol, 0)
    file.write(p_sol, 0)
    file.write(T_sol, 0)

# Plot
plt.figure()
p = plot(T_sol)
plt.colorbar(p)
plt.savefig("q11_T.png")