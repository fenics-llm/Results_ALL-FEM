from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt

# Geometry
channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
circle = Circle(Point(0.2, 0.2), 0.05)
mesh = generate_mesh(channel - circle, 64)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Parameters
mu = Constant(0.001)
rho = Constant(1.0)
U_bar = 0.2
H = 0.41

# Boundary markers
bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bndry.set_all(0)
tol = 1e-6

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0, tol) or near(x[1], 0.41, tol))
class Circle(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[0]-0.2)**2 + (x[1]-0.2)**2 <= 0.05**2 + tol)
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 2.2, tol)

Inlet().mark(bndry, 1)
Walls().mark(bndry, 2)
Circle().mark(bndry, 3)
Outlet().mark(bndry, 4)
ds = Measure("ds", domain=mesh, subdomain_data=bndry)

# Boundary conditions
inlet_profile = Expression(("6*U_bar*x[1]*(H - x[1])/(H*H)", "0.0"),
                           U_bar=U_bar, H=H, degree=2)
bc_inlet = DirichletBC(W.sub(0), inlet_profile, bndry, 1)
bc_walls = DirichletBC(W.sub(0), Constant((0, 0)), bndry, 2)
bc_circle = DirichletBC(W.sub(0), Constant((0, 0)), bndry, 3)

# Variational form
a = (mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u))*dx
L = Constant(0.0)*v*dx

# Solve
w = Function(W)
solve(a == L, w, [bc_inlet, bc_walls, bc_circle])

# Postprocessing
u_sol, p_sol = w.split()
speed = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "CG", 1))
plt.figure(); p = plot(speed); plt.colorbar(p); plt.savefig("q7_speed.png")

# Drag coefficient
n = FacetNormal(mesh)
F_D = assemble((mu*dot(grad(u_sol), n) - p_sol*n)*ds(3))
C_D = 2*F_D/(rho*U_bar**2*0.1)

# Save
with XDMFFile("q7_soln.xdmf") as file:
    file.write(u_sol, 0)
    file.write(p_sol, 0)