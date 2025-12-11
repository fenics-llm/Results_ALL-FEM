from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt

mesh = generate_mesh(Rectangle(Point(0,0), Point(1.0,0.20)), 100*20)

V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

E, nu = 5.0e6, 0.49
mu = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))
I = Identity(2)
def eps(w): return sym(grad(w))
def sigma(w,p): return 2*mu*eps(w) - p*I

a = (inner(sigma(u,p), eps(v))*dx + q*div(u)*dx + div(u)*p*dx)
L = Constant(0.0)*v[0]*dx + Constant(0.0)*v[1]*dx

bdr = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bdr.set_all(0)
class Left(SubDomain):
    def inside(self,x,on): return on and near(x[0],0.0)
class Right(SubDomain):
    def inside(self,x,on): return on and near(x[0],1.0)
Left().mark(bdr,1); Right().mark(bdr,2)
ds = Measure("ds", domain=mesh, subdomain_data=bdr)
bc_left = DirichletBC(W.sub(0), Constant((0.0,0.0)), bdr, 1)
bc_right = DirichletBC(W.sub(0), Constant((0.03,0.0)), bdr, 2)

w = Function(W)
solve(a == L, w, [bc_left, bc_right])
u_sol, _ = w.split()

u_mag = project(sqrt(dot(u_sol,u_sol)), FunctionSpace(mesh,"CG",1))
plt.figure(); c=plot(u_mag); plt.colorbar(c); plt.savefig("q10_disp.png")
File("q10_disp.xdmf") << u_sol