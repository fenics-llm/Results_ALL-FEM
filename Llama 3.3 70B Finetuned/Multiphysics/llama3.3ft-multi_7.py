from dolfin import *
import mshr

# Geometry and mesh
Lx, Ly = 1.0, 0.20
domain = mshr.Rectangle(Point(0, 0), Point(Lx, Ly))
filter_region = mshr.Rectangle(Point(0.4, 0), Point(0.6, Ly))
mesh = mshr.generate_mesh(domain - filter_region + filter_region, 64)

# Function space: P2 for velocity and P1 for pressure
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Trial and test functions
up = Function(W); u, p = up.split()
vq = TestFunction(W); v, q = split(vq)

# Parameters
rho, mu, K = 1.0, Constant(0.01), Constant(1e-6)
U_bar, H = 1.0, Ly

# Boundary markers
bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1); bndry.set_all(0)

class Inlet(SubDomain):
    def inside(self, x, on_b): return on_b and near(x[0], 0)
Inlet().mark(bndry, 1)

class Outlet(SubDomain):
    def inside(self, x, on_b): return on_b and near(x[0], Lx)
Outlet().mark(bndry, 2)

class Walls(SubDomain):
    def inside(self, x, on_b): return on_b and (near(x[1], 0) or near(x[1], Ly))
Walls().mark(bndry, 3)

ds = Measure("ds", domain=mesh, subdomain_data=bndry)
n = FacetNormal(mesh)

# Boundary conditions
ux_expr = Expression("6*U_bar*x[1]*(H-x[1])/pow(H,2)", U_bar=U_bar, H=H, degree=2)
noslip = Constant((0.0, 0.0))
bcs = [
    DirichletBC(W.sub(0).sub(0), ux_expr, bndry, 1),
    DirichletBC(W.sub(0).sub(1), noslip, bndry, 1),
    DirichletBC(W.sub(0), noslip, bndry, 3)
]

# Variational formulation
F = (rho*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx

# Add Brinkman terms in the porous region
domain_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
domain_markers.set_all(0)
filter_region.mark(domain_markers, 1)
dx = Measure("dx", domain=mesh, subdomain_data=domain_markers)

F += mu/K*inner(u, v)*dx(1) - p*div(v)*dx(1) + q*div(u)*dx(1)

# Add traction-free condition at outlet
t = -p*n + mu*(grad(u)+grad(u).T)*n

a = lhs(F)
L = rhs(F)

A, b = assemble_system(a, L, bcs)
solve(A, up.vector(), b)

u_sol, p_sol = up.split(True)

# Postprocessing
speed = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "CG", 1))
plot_object(speed, title="|u(x,y)|")
plt.savefig("q14_speed.png")

# Pressure drop across the porous block
x_before = 0.4 - DOLFIN_EPS; x_after = 0.6 + DOLFIN_EPS
p_before = p_sol(0.5*x_before, Ly/2)
p_after = p_sol(0.5*x_after, Ly/2)

dp = float(p_before - p_after)
print("Pressure drop:", dp)
with open("q14_dp.txt", "w") as f:
    f.write(str(dp))

# Save solution to file
file = XDMFFile("q14_solution.xdmf")
file.parameters["functions_share_mesh"] = True
file.parameters["rewrite_function_mesh"] = False
file.write(u_sol, 0)
file.write(p_sol, 0)
