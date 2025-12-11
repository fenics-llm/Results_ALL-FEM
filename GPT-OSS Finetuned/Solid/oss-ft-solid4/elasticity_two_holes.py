# filename: elasticity_two_holes.py
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt

# Geometry with two holes
a_hole = 0.04
domain = Rectangle(Point(0.0, 0.0), Point(1.0, 0.20)) \
         - Circle(Point(0.33, 0.10), a_hole) \
         - Circle(Point(0.67, 0.10), a_hole)
mesh = generate_mesh(domain, 80)

# Function space (plane stress)
V = VectorFunctionSpace(mesh, "Lagrange", degree=2)

# Material (plane stress)
E, nu = 200e9, 0.30
mu = E/(2.0*(1.0+nu))
lmbda = E*nu/((1.0+nu)*(1.0-nu))

def epsilon(u): return sym(grad(u))
def sigma(u): return lmbda*tr(epsilon(u))*Identity(2) + 2.0*mu*epsilon(u)

# Boundary markers
tol = 1E-10
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

class Left(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 1.0, tol)
class Hole(SubDomain):
    def __init__(self, c): self.c = c; super().__init__()
    def inside(self, x, on_boundary):
        return on_boundary and ((x[0]-self.c[0])**2 + (x[1]-self.c[1])**2 < (a_hole+tol)**2)

left, right = Left(), Right()
hole1, hole2 = Hole((0.33,0.10)), Hole((0.67,0.10))
left.mark(boundaries, 1); right.mark(boundaries, 2)
hole1.mark(boundaries, 3); hole2.mark(boundaries, 3)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Boundary conditions
zero = Constant((0.0, 0.0))
bc_left = DirichletBC(V, zero, left)

# Variational problem
u, v = TrialFunction(V), TestFunction(V)
T = Constant((2e6, 0.0))
a = inner(sigma(u), epsilon(v))*dx
L = dot(T, v)*ds(2)

# Solve
u_sol = Function(V, name="Displacement")
solve(a == L, u_sol, bc_left)

# Von Mises stress
s = sigma(u_sol) - (1./3)*tr(sigma(u_sol))*Identity(2)
vonMises = sqrt(3./2*inner(s, s))

# Max von Mises (approx.) on holes
Vdg = FunctionSpace(mesh, "DG", 0)
vonMises_dg = project(vonMises, Vdg)
max_vm = vonMises_dg.vector().get_local().max()
print("Maximum von Mises stress (approx.) : {:.2f} MPa".format(max_vm/1e6))
print("Stress concentration factor Kt = {:.3f}".format(max_vm/2e6))

# Save results
with XDMFFile(mesh.mpi_comm(), "displacement.xdmf") as xdmf:
    xdmf.write(u_sol)

plt.figure(figsize=(6,3))
p = plot(vonMises, title="Von Mises stress (MPa)", cmap="viridis")
plt.colorbar(p, label="MPa")
plt.axis('off')
plt.tight_layout()
plt.savefig("q4_vm.png", dpi=300)
plt.close()