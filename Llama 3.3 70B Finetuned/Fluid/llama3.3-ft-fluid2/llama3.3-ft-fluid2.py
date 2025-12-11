from dolfin import *
import matplotlib.pyplot as plt

L = 2.0; H = 0.20
mesh = RectangleMesh(Point(0, 0), Point(L, H), 120, 12)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

class Inlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0) and on_boundary
class Outlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], L) and on_boundary
class Walls(SubDomain):
    def inside(self, x, on_boundary): return (near(x[1], 0.0) or near(x[1], H)) and on_boundary
class P0(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0) and near(x[1], 0.0)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0); Inlet().mark(boundaries, 1); Outlet().mark(boundaries, 2); Walls().mark(boundaries, 3)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

mu = Constant(1.0); rho = Constant(1.0)
f = Constant((1.0, 0.0))

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

a = (2*mu*inner(sym(grad(u)), sym(grad(v))) - div(v)*p + q*div(u))*dx
L = dot(f, v)*dx

bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 3)
bc_p0 = DirichletBC(W.sub(1), Constant(0.0), P0(), method="pointwise")
bcs = [bc_walls, bc_p0]

w = Function(W); solve(a == L, w, bcs)

u_sol, p_sol = w.split()
speed = project(sqrt(dot(u_sol, u_sol)), Q)
plt.figure(); c=plot(speed, title="Speed |u|"); plt.colorbar(c); plt.savefig("q2_speed.png")
xf = XDMFFile("q2_solution.xdmf"); xf.write(u_sol); xf.write(p_sol)