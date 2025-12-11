from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt

# Geometry
mesh = generate_mesh(Rectangle(Point(0, 0), Point(1.0, 0.20)) - Circle(Point(0.5, 0.1), 0.04), 64)

# Mixed space
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Material (plane strain)
E, nu = 5e6, 0.49
mu = E / (2*(1+nu))
lmbda = E*nu / ((1+nu)*(1-2*nu))
kappa = lmbda + 2*mu/3.0
def eps(w): return sym(grad(w))
def sigma_dev(w, p): return 2*mu*eps(w) - p*Identity(2)

# BCs
bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)
tol = 1e-3
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], 1.0, tol)
Left().mark(bdr, 1)
Right().mark(bdr, 2)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)
bc_left = DirichletBC(W.sub(0), Constant((0.0, 0.0)), bdr, 1)
bc_right = DirichletBC(W.sub(0), Constant((0.001, 0.0)), bdr, 2)

# Variational form
a = (inner(2*mu*eps(u), eps(v)) - p*div(v) - q*div(u) - (1.0/kappa)*p*q)*dx
L = Constant(0.0)*q*dx

# Solve
w = Function(W)
solve(a == L, w, [bc_left, bc_right])

# Postprocess
u_h, p_h = w.split()
sxx = 2*mu*eps(u_h)[0,0] - p_h
syy = 2*mu*eps(u_h)[1,1] - p_h
sxy = 2*mu*eps(u_h)[0,1]
szz = -p_h
vm3d = sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2 + 6*sxy**2))
VM = project(vm3d, FunctionSpace(mesh, "CG", 1))
u_x = project(u_h[0], FunctionSpace(mesh, "CG", 1))

# Save
plt.figure(); pplt = plot(VM, title="von Mises"); plt.colorbar(pplt); plt.savefig("q11_vm.png")
plt.figure(); pplt = plot(u_x, title="u_x"); plt.colorbar(pplt); plt.savefig("q11_ux.png")
XDMFFile("q11_disp.xdmf").write(u_h)