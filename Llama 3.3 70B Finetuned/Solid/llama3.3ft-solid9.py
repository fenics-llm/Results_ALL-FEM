from dolfin import *
import matplotlib.pyplot as plt

# Geometry
mesh = RectangleMesh(Point(0, 0), Point(1.0, 0.20), 100, 20)

# Material
E0, E1 = 100e9, 100e9
nu = 0.30
rho, g = 2700, 9.81

V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "CG", 1)

ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

class Left(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0) and on_boundary

class Right(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 1.0) and on_boundary

facets = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
facets.set_all(0)

Left().mark(facets, 3)
Right().mark(facets, 4)

ds = Measure("ds", domain=mesh, subdomain_data=facets)

u, p = TrialFunctions(W)
v, q = TestFunctions(W)

y = SpatialCoordinate(mesh)[1]
E = E0 + E1 * y / 0.20
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

def eps(w): return sym(grad(w[0]))
def sigma(w):
    return lmbda * tr(eps(w)) * Identity(2) + 2 * mu * eps(w)

a = inner(sigma(u), eps(v)) * dx
L = rho * g * dot(v[0], Constant((0, -1))) * dx

bc_left = DirichletBC(W.sub(0).collapse(), Constant((0.0, 0.0)), facets, 3)
tR = Constant((2e6, 0))
L += dot(tR, v[0]) * ds(4)

w = Function(W)
solve(a == L, w, bc_left)

u_sol, p_sol = w.split(deep=True)
plot(sqrt(dot(u_sol, u_sol)), title="|u|")
plt.savefig("q9_disp.png")

File("q9_disp.xdmf") << u_sol