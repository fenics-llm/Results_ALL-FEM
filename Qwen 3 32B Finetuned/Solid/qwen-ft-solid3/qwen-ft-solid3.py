from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt
import numpy as np

# Geometry
outer = Rectangle(Point(0, 0), Point(1.0, 0.2))
hole = Circle(Point(0.5, 0.1), 0.05)
mesh = generate_mesh(outer - hole, 64)

# Material
E, nu = 200e9, 0.3
mu = E / (2*(1+nu))
lmbda = E*nu / ((1+nu)*(1-2*nu))
lmbda_ps = 2*mu*lmbda/(lmbda + 2*mu)
def eps(v): return sym(grad(v))
def sigma(v): return lmbda_ps*tr(eps(v))*Identity(2) + 2*mu*eps(v)

# Function space
V = VectorFunctionSpace(mesh, "CG", 1)
u, v = TrialFunction(V), TestFunction(V)

# BCs
bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)
tol = 1e-3
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1.0, tol)
Left().mark(bdr, 1); Right().mark(bdr, 2)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)
bc_left = DirichletBC(V, Constant((0.0, 0.0)), bdr, 1)

# Variational form
a = inner(sigma(u), eps(v))*dx
L = dot(Constant((2e6, 0.0)), v)*ds(2)

# Solve
u_sol = Function(V)
solve(a == L, u_sol, [bc_left])

# von Mises
s = sigma(u_sol)
von_mises = project(sqrt(s[0,0]**2 - s[0,0]*s[1,1] + s[1,1]**2 + 3*s[0,1]**2), FunctionSpace(mesh, "CG", 1))

# Post
XDMFFile("q3_disp.xdmf").write(u_sol)
plt.figure(); p = plot(von_mises, title="von Mises (MPa)"); plt.colorbar(p); plt.savefig("q3_vm.png")

# Hole max stress
Q = FunctionSpace(mesh, "CG", 1)
coords = Q.tabulate_dof_coordinates().reshape((-1, 2))
vm_arr = von_mises.vector().get_local()
r = np.sqrt((coords[:,0]-0.5)**2 + (coords[:,1]-0.1)**2)
mask = np.abs(r - 0.05) < mesh.hmin()
hole_max = vm_arr[mask].max()
print(f"Max von Mises at hole: {hole_max:.2f} Pa, Kt = {hole_max/2e6:.2f}")