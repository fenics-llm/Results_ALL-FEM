from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt

# Geometry
rect = Rectangle(Point(0, 0), Point(1.0, 0.20))
hole = Circle(Point(0.50, 0.10), 0.04)
mesh = generate_mesh(rect - hole, 64)

# Material parameters
E, nu = 5e6, 0.5
mu = E/(2*(1+nu))
lmbda = 1e10  # Large lambda for incompressibility

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Kinematics
I = Identity(2)
F = I + grad(u)
C = F.T*F
J = det(F)

# Neo-Hookean energy
psi = (mu/2)*(tr(C) - 2) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Variational form
a = derivative(psi, u, v) - p*div(v)*dx + q*(J - 1)*dx
L = -p*div(v)*dx + q*(J - 1)*dx

# Boundary conditions
bcs = []
# Left edge fixed
bcs.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), "near(x[0], 0.0)"))
# Hole traction
P_hole = Constant(1e5)
tol = 1e-10
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Hole(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near((x[0]-0.5)**2 + (x[1]-0.1)**2, 0.04**2, tol)
Hole().mark(facets, 1)
ds = Measure("ds", domain=mesh, subdomain_data=facets)
a += inner(dot(sigma(u, p)) * v * dx + q * (J - 1) * dx
L = - p * div(v) * dx + q * (J - 1) * dx

# Solve
w = Function(W)
solve(lhs(a) == rhs(L), w, bcs)

# Postprocessing
u_h, p_h = w.split()
V_vm = FunctionSpace(mesh, "P", 1)
vm = project(sqrt(0.5 * ((sigma(u_h, p_h)[0, 0] - sigma(u_h, p_h)[1, 1])**2 + 3 * sigma(u_h, p_h)[0, 1]**2), V_vm)

# Save
u_mag = interpolate(Expression(("x[0] + 5*u[0]", "x[1] + 5*u[1]"), u=u_h, degree=2), V)
plot(u_mag, title="Deformed (×5)")
plt.savefig("q13_def.png")
plot(vm, title="Von Mises")
plt.savefig("q13_vm.png")
File("q13_disp.pvd").write(u_h)
