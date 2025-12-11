# filename: q7.py
from dolfin import *
import mshr
import matplotlib.pyplot as plt

L = 1.0
H = 0.20
mesh = RectangleMesh(Point(0, 0), Point(L, H), 80, 16)

E_al, nu_al = 70e9, 0.30
mu_al = E_al / (2 * (1 + nu_al))
lmbda_al = E_al * nu_al / ((1 + nu_al) * (1 - 2 * nu_al))
lmbda_al = 2 * mu_al * lmbda_al / (lmbda_al + 2 * mu_al)

E_steel, nu_steel = 200e9, 0.30
mu_steel = E_steel / (2 * (1 + nu_steel))
lmbda_steel = E_steel * nu_steel / ((1 + nu_steel) * (1 - 2 * nu_steel))
lmbda_steel = 2 * mu_steel * lmbda_steel / (lmbda_steel + 2 * mu_steel)

V = VectorFunctionSpace(mesh, "P", 1)

u = TrialFunction(V)
v = TestFunction(V)

x = SpatialCoordinate(mesh)

def sigma(u):
    mu = conditional(gt(x[1], 0.10), mu_al, mu_steel)
    lmbda = conditional(gt(x[1], 0.10), lmbda_al, lmbda_steel)
    return 2 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(2)

a = inner(sigma(u), sym(grad(v))) * dx

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)
Right().mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
L_form = dot(Constant((0.0, -5000)), v) * ds(1)

bc_left = DirichletBC(V, Constant((0.0, 0.0)), "near(x[0], 0)")

u_sol = Function(V)
solve(a == L_form, u_sol, bc_left)

with XDMFFile("q7_disp.xdmf") as xdmf:
    xdmf.write(u_sol)

plot_object = plot(u_sol, mode="displacement", title="Displacement magnitude |u| (m)")
plt.colorbar(plot_object)
plt.savefig("q7_disp.png")