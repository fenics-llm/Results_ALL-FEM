from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

mesh = generate_mesh(Rectangle(Point(0,0), Point(1,0.2)), 60)
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(1)
class Porous(SubDomain):
    def inside(self, x, on_b): return between(x[0], (0.4, 0.6))
Porous().mark(subdomains, 2)
dx = Measure("dx", domain=mesh, subdomain_data=subdomains)

V = VectorFunctionSpace(mesh, "P", 2)
Q = FunctionSpace(mesh, "P", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p), (v, q) = TrialFunctions(W), TestFunctions(W)

mu = Constant(0.01)
K = Constant(1e-6)
rho = Constant(1.0)
U_bar = 1.0
H = 0.2

a = (mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u))*dx(1)
a += (mu*inner(grad(u), grad(v)) - (mu/K)*inner(u,v) - div(v)*p - q*div(u))*dx(2)
L = Constant((0.0, 0.0))*v*dx

bcs = []
class Inlet(SubDomain):
    def inside(self, x, on_b): return on_b and near(x[0], 0)
inlet = Inlet()
u_in = Expression(("6*U_bar*x[1]*(H - x[1])/(H*H)", "0.0"),
                  U_bar=U_bar, H=H, degree=2)
bcs.append(DirichletBC(W.sub(0), u_in, inlet))
class Walls(SubDomain):
    def inside(self, x, on_b): return on_b and (near(x[1],0) or near(x[1],0.2))
walls = Walls()
bcs.append(DirichletBC(W.sub(0), Constant((0,0)), walls))

w = Function(W)
solve(a == L, w, bcs, solver_parameters={"linear_solver": "mumps"})
u_sol, p_sol = w.split()

Vp = VectorFunctionSpace(mesh, "P", 1)
up = project(u_sol, Vp)
speed = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "P", 1))
plt.figure(); c = plot(speed); plt.colorbar(c); plt.savefig("q14_speed.png")

y0 = 0.1
p_before = p_sol(0.4, y0)
p_after = p_sol(0.6, y0)
with open("q14_dp.txt", "w") as f:
    f.write(f"{p_before - p_after:.6f}\n")

with XDMFFile("q14_solution.xdmf") as file:
    file.write(u_sol, 0)
    file.write(p_sol, 0)
