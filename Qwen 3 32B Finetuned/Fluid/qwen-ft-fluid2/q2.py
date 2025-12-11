from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

L, H = 2.0, 0.2
mesh = generate_mesh(Rectangle(Point(0,0), Point(L,H)), 120)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p), (v, q) = TrialFunctions(W), TestFunctions(W)
mu, rho = 1.0, 1.0
f = Constant((1.0, 0.0))
bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1],0) or near(x[1],H))
Walls().mark(bdr, 1)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)
noslip = Constant((0.0, 0.0))
bc = DirichletBC(W.sub(0), noslip, bdr, 1)
a = mu*inner(grad(u), grad(v))*dx - div(v)*p*dx - q*div(u)*dx
L = dot(f, v)*dx
w = Function(W)
solve(a == L, w, [bc])
u_sol, p_sol = w.split()
speed = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "CG", 1))
plt.figure(); c = plot(speed); plt.colorbar(c); plt.savefig("q2_speed.png")
with XDMFFile("q2_solution.xdmf") as xf:
    xf.write(u_sol, 0.0)
    xf.write(p_sol, 0.0)
