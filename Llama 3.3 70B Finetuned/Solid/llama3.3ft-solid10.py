from dolfin import *
import matplotlib.pyplot as plt

# Geometry: 1.0 m x 0.20 m, structured mesh 100 × 20
mesh = RectangleMesh(Point(0, 0), Point(1, 0.2), 100, 20)

# Material parameters
E = Constant(5e3)  # Young’s modulus (Pa)
nu = Constant(0.49)  # Poisson ratio

# Lame coefficients for plane strain
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Mixed displacement–pressure formulation
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and on_boundary

# Boundary markers
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
Left().mark(boundaries, 1)
Right().mark(boundaries, 2)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# Trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Stress tensor for plane strain
def sigma(u, p):
    return 2 * mu * sym(grad(u)) + (lmbda * tr(sym(grad(u))) - p) * Identity(2)

# Variational form: a(u, v) = L(v)
a = inner(sigma(u, p), epsilon(v)) * dx
L = Constant(0.0) * dot(v, v) * dx

# Boundary conditions
u_L = Expression(("0.03", "0.0"), degree=1)
bc_left = DirichletBC(W.sub(0), Constant((0.0, 0.0)), boundaries, 1)
bc_right = DirichletBC(W.sub(0), u_L, boundaries, 2)

# Solve
w = Function(W)
solve(a == L, w, [bc_left, bc_right])

u_h, p_h = w.split()

# Post-processing: displacement magnitude |u|
V0 = V
u_mag = project(sqrt(dot(u_h, u_h)), V0)

plt.figure(figsize=(8, 2.5))
c = plot(u_mag)
plt.colorbar(c)
plt.title("Problem 10 – Displacement magnitude |u|")
plt.savefig("q10_disp.png", dpi=150)

# Save displacement field
file = XDMFFile("q10_u.xdmf")
file.parameters["flush_output"] = True
file.write(u_h, 0.0)