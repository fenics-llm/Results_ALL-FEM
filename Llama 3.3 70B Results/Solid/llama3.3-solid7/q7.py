from fenics import *

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(1.0, 0.20), 80, 16)

# Define material subdomains
class Aluminum(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 0.10

class Steel(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= 0.10

# Create subdomain markers
subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
subdomains.set_all(0)
aluminum = Aluminum()
steel = Steel()
aluminum.mark(subdomains, 1)
steel.mark(subdomains, 2)

# Define material parameters
E_al = 70e9  # Young's modulus for aluminum (Pa)
nu_al = 0.30  # Poisson's ratio for aluminum

E_steel = 200e9  # Young's modulus for steel (Pa)
nu_steel = 0.30  # Poisson's ratio for steel

# Define variational problem
V = VectorFunctionSpace(mesh, "Lagrange", 1)

u = TrialFunction(V)
v = TestFunction(V)

def sigma(u, E, nu):
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(2)

def epsilon(v):
    return sym(grad(v))

a_al = inner(sigma(u, E_al, nu_al), epsilon(v)) * dx
a_steel = inner(sigma(u, E_steel, nu_steel), epsilon(v)) * dx

l = -5000 * v[1] * ds(4)

# Define boundary conditions
def left_boundary(x, on_boundary):
    return near(x[0], 0) and on_boundary

bc = DirichletBC(V, Constant((0, 0)), left_boundary)

# Solve variational problem
u_sol = Function(V)
solve(a_al + a_steel == l, u_sol, bc)

# Save displacement field to XDMF file
file = XDMFFile("q7_disp.xdmf")
file.write(u_sol)

# Save color map of displacement magnitude
import matplotlib.pyplot as plt

plt.figure()
plot(u_sol, title="Displacement magnitude", cmap="viridis")
plt.savefig("q7_disp.png")