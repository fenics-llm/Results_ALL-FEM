from dolfin import *
import mshr

# Geometry
domain = mshr.Rectangle(Point(0, 0), Point(1, 0.2)) - \
         mshr.Circle(Point(0.33, 0.10), 0.04) - \
         mshr.Circle(Point(0.67, 0.10), 0.04)
mesh = mshr.generate_mesh(domain, 64)

# Material
E, nu = 200e9, 0.3
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 - nu) ** 2)

# Function space and boundary markers
V = VectorFunctionSpace(mesh, "P", 1)
bmarks = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
bmarks.set_all(0)
tol = 1e-6

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0, tol)

Left().mark(bmarks, 1)
Right().mark(bmarks, 2)

ds = Measure("ds", domain=mesh, subdomain_data=bmarks)

# Variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = mu * inner(grad(u), grad(v)) + lmbda * div(v) * u[0]
L = Constant((0.0, 2.0)) * v[0] * ds(2)

sigma = Function(V)
solve(a == L, sigma, bmarks=1)

# Von Mises stress
def von_mises(sigma):
    return sqrt(3.0 / 2.0 * (sigma[0, 0] ** 2 + sigma[0, 1] ** 2 - sigma[1, 1] ** 2 -
                             sigma[0, 0] * sigma[0, 1] + sigma[1, 0] * sigma[1, 1]))

vm = von_mises(sigma)

# Plot
plt.figure()
p = plot(vm)
plt.colorbar(p)
plt.title("Problem 4: Von Mises stress")
plt.savefig("q4_vm.png")

# Save displacement field in XDMF format
File("displacement.xdmf") << u

# Report maximum von Mises stress at hole boundary and Stress Concentration Factor (Kt = σ_max/2 MPa)
print(f"Maximum von Mises stress: {vm.max():.3e} Pa")
print(f"Stress concentration factor Kt: {(vm.max() / 2):.3f}")