from dolfin import *
import numpy as np

# Geometry
L, H = 2.0, 0.20
mesh = RectangleMesh(Point(0, 0), Point(L, H), 240, 24)

# Function space
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Trial and test functions
up = Function(W); u, p = up.split()
vq = TestFunction(W); v, q = split(vq)

# Material parameters
rho, mu0, n = 1.0, 0.5, 0.5

# Boundary conditions
tol = 1e-10
class Inlet(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0)
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and near(x[1], 0)) or (on_boundary and near(x[1], H))
class Outlet(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], L)

inlet = Inlet(); walls = Walls(); outlet = Outlet()
bc_inlet = DirichletBC(W.sub(0), Expression(("6*U_bar*x[1]*(H-x[1])/pow(H,2)", "0.0"),
    U_bar=1.0, H=H, degree=2), inlet)
bcs = [bc_inlet,
       DirichletBC(W.sub(0).sub(1), Constant(0.0), walls)]

# Variational formulation
F = rho * inner(dot(u, nabla_grad(u)), v) * dx \
    - p * div(v) * dx + q * div(u) * dx

D = 0.5 * (nabla_grad(u) + nabla_grad(u).T)
mu_eff = mu0 * pow((2 * inner(D, D))**0.5 + Constant(1e-8), n - 1)

tau = 2 * mu_eff * D
F += inner(tau, grad(v)) * dx

# Solve
solve(lhs(F) == rhs(F), up, bcs=bcs)

u_sol, p_sol = up.split(deep=True)
mu_eff_sol = project(mu0 * pow((2 * inner(0.5*(nabla_grad(u_sol)+nabla_grad(u_sol).T),
    0.5*(nabla_grad(u_sol)+nabla_grad(u_sol).T)))**0.5 + Constant(1e-8), n - 1),
    FunctionSpace(mesh, "CG", 1))

# Postprocessing
u_mag = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "CG", 1))
plot_object = plot(u_mag, title="Velocity magnitude |u|")
plt.savefig("q12_speed.png")

ux_mid = np.array([u_sol(x, H/2.0)[0] for x in np.linspace(0, L, 100)])
np.savetxt("q12_profile.csv", np.vstack((np.linspace(0, L, 100), ux_mid)).T,
    header="x, u_x(y=H/2)", comments="")
print(f"Max velocity at mid-channel: {max(ux_mid):.4f}")

file = XDMFFile("q12_solution.xdmf")
file.parameters["flush_output"] = True
file.write(u_sol, 0); file.write(p_sol, 0); file.write(mu_eff_sol, 0)