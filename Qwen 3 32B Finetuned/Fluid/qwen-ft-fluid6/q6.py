from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

H = 1.0
mesh = generate_mesh(Rectangle(Point(-3*H, 0), Point(0, H)) + Rectangle(Point(0, 0), Point(20*H, 2*H)), 64)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

mu, rho = 0.01, 1.0
U_bar = 1.0
f = Constant((0.0, 0.0))

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -3*H)
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0) or (near(x[0], 0) and x[1] >= H))
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 20*H)

Inlet().mark(boundaries, 1)
Walls().mark(boundaries, 2)
Outlet().mark(boundaries, 3)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

u_in = Expression(("6*U_bar*(x[1]/H)*(1 - x[1]/H)", "0.0"), U_bar=U_bar, H=H, degree=2)
noslip = Constant((0.0, 0.0))

bc_inlet = DirichletBC(W.sub(0), u_in, boundaries, 1)
bc_walls = DirichletBC(W.sub(0), noslip, boundaries, 2)

a = (mu*inner(grad(u), grad(v)) - div(v)*p - q*div(u))*dx
L = dot(f, v)*dx

w = Function(W)
solve(a == L, w, [bc_inlet, bc_walls])

u_sol, p_sol = w.split(deepcopy=True)

# Compute wall shear stress on top wall (y=2H, x ∈ [0,20H])
top_facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
top_facets.set_all(0)
class TopWall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 2*H) and x[0] >= 0
TopWall().mark(top_facets, 1)
ds_top = Measure("ds", domain=mesh, subdomain_data=top_facets)

tau_w = mu*grad(u_sol)[0,1]  # y-component of stress
tau_w_values = assemble(tau_w*ds_top)

# Save results
u_sol.rename("velocity", "u")
p_sol.rename("pressure", "p")
with XDMFFile("q6_soln.xdmf") as file:
    file.write(u_sol)
    file.write(p_sol)

plt.figure()
plot(u_sol)
plt.savefig("q6_u.png")