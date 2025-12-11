from dolfin import *
import numpy as np

# Geometry
L = 2.0; H = 0.20
mesh = RectangleMesh(Point(0, 0), Point(L, H), 80, 8)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME_T = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME_T)

# Trial and test functions
up = Function(W); u, p = split(up); vq = TestFunction(W); v, q = split(vq)

# Parameters
rho = Constant(1.0)
Ubar = 1.0; Hval = Constant(H)
mu_ref = Constant(0.02)
beta = Constant(0.05)
Tref = Constant(300.0)
kappa = Constant(1e-3)

# Temperature-dependent viscosity
def mu(T):
    return mu_ref * exp(-beta*(T - Tref))

# Momentum equation residual
F_mom = rho*inner(u, grad(u)*v)*dx \
        + inner(mu(p)*sym(grad(u)), sym(grad(v)))*dx \
        - p*div(v)*dx + div(u)*q*dx

# Temperature equation residual
def F_temp(up, v):
    u_, p_ = up.split()
    T = p_
    return inner(dot(u_, grad(T)), v)*dx + kappa*inner(grad(T), grad(v))*dx

# Boundary conditions
class Inlet(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0)
inlet = Inlet()
u_inlet = Expression(("6*Ubar*x[1]*(Hval-x[1])/(Hval*Hval)", "0.0"),
                     Ubar=Ubar, Hval=Hval, degree=2)

class Bottom(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.0)
bottom = Bottom()
bc_bottom_T = DirichletBC(W.sub(1), Constant(Tref + 10.0), bottom)

# Boundary markers
bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1); bndry.set_all(0)
inlet.mark(bndry, 1); bottom.mark(bndry, 2)
ds = Measure("ds", domain=mesh, subdomain_data=bndry)

# Variational problem
a = lhs(F_mom) + inner(grad(p), grad(q))*dx
L = rhs(F_mom)

bc_inlet_u = DirichletBC(W.sub(0), u_inlet, inlet)
bcs = [bc_inlet_u, bc_bottom_T]

up_ = Function(W); solve(a == L, up_, bcs=bcs)
u_sol, p_sol = up_.split(deepcopy=True)

# Viscosity field
mu_field = project(mu(p_sol), FunctionSpace(mesh, "CG", 2))

# Save viscosity as image
plot(mu_field, title="Viscosity μ(x,y)")
import matplotlib.pyplot as plt; plt.savefig("q13_mu.png")

# Extract velocity profile at x=1.0 m
y_values = np.linspace(0.0, H, 21)
ux_values = [u_sol(Point(1.0, y))[0] for y in y_values]
np.savetxt("q13_profile.csv", np.column_stack((y_values, ux_values)),
           header="y,ux", comments="")

# Save solution fields
file = XDMFFile("q13_solution.xdmf")
file.parameters["functions_share_mesh"] = True
file.parameters["rewrite_function_mesh"] = False
file.write(u_sol, 0.0); file.write(p_sol, 0.0)
file.write(mu_field, 0.0)