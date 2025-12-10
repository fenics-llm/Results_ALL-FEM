from dolfin import *
import mshr

# Geometry
domain = Rectangle(Point(0, 0), Point(1, 0.2)) - Rectangle(Point(0, 0.06), Point(0.08, 0.12))
mesh = generate_mesh(domain, 64)

# Material
E = 200000.0
nu = 0.3

# Function space and boundary markers
V = VectorFunctionSpace(mesh, "P", 1)
bmarks = MeshFunction("size_t", mesh, mesh.topology().dim()-1); bmarks.set_all(0)
tol = 1e-6
class Left(SubDomain):
    def inside(self, x, on_b): return on_b and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on_b): return on_b and near(x[0], 1.0, tol)
Left().mark(bmarks, 1); Right().mark(bmarks, 2)
ds = Measure("ds", domain=mesh, subdomain_data=bmarks)

# Variational formulation
u, v = TrialFunction(V), TestFunction(V)
a = E*nu*((div(v))*div(u) + div(u)*div(v)) - (1.0+nu)*(tr(sigma(u))*tr(sigma(v)))
L = dot((2.0e6, 0.0), v)*dx

# Boundary conditions
bc_left = DirichletBC(V, Constant((0.0, 0.0)), bmarks, 1)
bc_right = DirichletBC(V, Constant((0.0, 0.0)), bmarks, 2)

# Solve
u_sol = Function(V); solve(a == L, u_sol, [bc_left, bc_right])

# Compute von Mises stress
def sigma(u):
    return (E*tr(sigma(u)) - E*(nu + 3.0/4.0)*sigma(u)[1,1])*eye(2)
vm = project(sqrt(0.5*((sigma(u_sol) - sigma(u_sol).T)**2 + 6.0*sigma(u_sol)[1,1]**2)), FunctionSpace(mesh, "P", 1))

# Plot
plot(u_sol); plot(vm); plot(bmarks); interactive()