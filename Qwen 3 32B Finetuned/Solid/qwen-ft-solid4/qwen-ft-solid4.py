from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt
import numpy as np

# Geometry
rect = Rectangle(Point(0, 0), Point(1.0, 0.2))
hole1 = Circle(Point(0.33, 0.10), 0.04)
hole2 = Circle(Point(0.67, 0.10), 0.04)
mesh = generate_mesh(rect - hole1 - hole2, 64)

# Material
E, nu = 200e9, 0.3
mu = E / (2*(1+nu))
lmbda = E*nu / ((1+nu)*(1-2*nu))
lmbda_ps = 2*mu*lmbda/(lmbda + 2*mu)

# Function space
V = VectorFunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(V), TestFunction(V)
def eps(w): return sym(grad(w))
def sigma(w): return lmbda_ps*tr(eps(w))*Identity(2) + 2*mu*eps(w)

# Boundary markers
bmarks = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bmarks.set_all(0)
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1.0)
Left().mark(bmarks, 1)
Right().mark(bmarks, 2)
ds = Measure("ds", domain=mesh, subdomain_data=bmarks)

# BCs
bc_left = DirichletBC(V, Constant((0.0, 0.0)), bmarks, 1)
t_right = Constant((2e6, 0.0))

# Variational form
a = inner(sigma(u), eps(v))*dx
L = dot(t_right, v)*ds(2)

# Solve
u_sol = Function(V)
solve(a == L, u_sol, [bc_left])

# von Mises
s = sigma(u_sol)
Q = FunctionSpace(mesh, "CG", 1)
vm = project(sqrt(s[0,0]**2 - s[0,0]*s[1,1] + s[1,1]**2 + 3*s[0,1]**2), Q)

# Hole boundary max
coords = Q.tabulate_dof_coordinates().reshape((-1, 2))
vm_arr = vm.vector().get_local()
r1 = np.sqrt((coords[:,0]-0.33)**2 + (coords[:,1]-0.10)**2)
r2 = np.sqrt((coords[:,0]-0.67)**2 + (coords[:,1]-0.10)**2)
mask = (np.abs(r1 - 0.04) < mesh.hmin()) | (np.abs(r2 - 0.04) < mesh.hmin())
max_vm = vm_arr[mask].max()

# Save & plot
XDMFFile("q4_disp.xdmf").write(u_sol)
plt.figure(); c = plot(vm, title="von Mises"); plt.colorbar(c); plt.savefig("q4_vm.png")

print(f"Max von Mises: {max_vm:.2f} Pa, Kt: {max_vm/2e6:.2f}")