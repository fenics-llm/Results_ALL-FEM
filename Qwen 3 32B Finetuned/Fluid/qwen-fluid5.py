from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

# Geometry & mesh
mesh = generate_mesh(Rectangle(Point(0,0), Point(1,1)), 128)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Boundary markers
bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bndry.set_all(0)
class Top(SubDomain):
    def inside(self, x, on): return on and near(x[1], 1.0)
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1.0)
class Bottom(SubDomain):
    def inside(self, x, on): return on and near(x[1], 0.0)
Top().mark(bndry, 1); Left().mark(bndry, 2); Right().mark(bndry, 3); Bottom().mark(bndry, 4)
ds = Measure("ds", domain=mesh, subdomain_data=bndry)

# BCs
noslip = Constant((0.0, 0.0))
lid = Constant((1.0, 0.0))
bc_top = DirichletBC(W.sub(0), lid, bndry, 1)
bc_wall = DirichletBC(W.sub(0), noslip, bndry, (2,3,4))
bc_p = DirichletBC(W.sub(1), Constant(0.0), Point(0.0, 0.0))
bcs = [bc_top, bc_wall, bc_p]

# Parameters
rho, mu = 1.0, 0.01

# Variational form
a = (rho*dot(u, v)*dx + mu*inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx)
L = Constant((0.0, 0.0))*v*dx

# Solve
w = Function(W)
solve(a == L, w, bcs, solver_parameters={"linear_solver": "mumps"})

# Postprocess
u_sol, p_sol = w.split()
speed = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "CG", 1))

# Save
with XDMFFile("q5_soln.xdmf") as file:
    file.write(u_sol, 0); file.write(p_sol, 0)
plt.figure(); p=plot(speed, title="|u|"); plt.colorbar(p); plt.savefig("q5_speed.png")
