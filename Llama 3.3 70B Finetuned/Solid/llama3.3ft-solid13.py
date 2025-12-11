from dolfin import *
from mshr   import Rectangle, Circle, generate_mesh

# Geometry: rectangular strip with circular hole
Lx = 1.0; Ly = 0.20; a = 0.04
hole_centre = Point(0.50, 0.10)
domain = Rectangle(Point(0,0), Point(Lx,Ly)) - Circle(hole_centre, a)
mesh = generate_mesh(domain, 64)

# Material parameters (plane strain)
E, nu = 5e6, 0.49
mu    = E/(2*(1 + nu))
lmbda = E*nu/((1 + nu)*(1 - 2*nu))

# Function space: P2 vector + P1 scalar
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

def eps(u):      return 0.5*(grad(u) + grad(u).T)
def sigma(u, p): return lmbda*tr(eps(u))*Identity(2) + 2*mu*eps(u) - p*Identity(2)

# Boundary markers
tol = 1e-10
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Left(SubDomain):
    def inside(self, x, on): return on and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on): return on and near(x[0], Lx, tol)
class Top(SubDomain):
    def inside(self, x, on): return on and near(x[1], Ly, tol)
class Bottom(SubDomain):
    def inside(self, x, on): return on and near(x[1], 0.0, tol)
class Hole(SubDomain):
    def inside(self, x, on): return on and (near((x[0]-hole_centre.x())**2 + (x[1]-hole_centre.y())**2, a*a, tol))
Left().mark(facets,1); Right().mark(facets,2)
Top().mark(facets,3); Bottom().mark(facets,4); Hole().mark(facets,5)

ds = Measure("ds", domain=mesh, subdomain_data=facets)

# Dirichlet BC: left edge fixed
u_L = Constant((0.0, 0.0))
bc = DirichletBC(W.sub(0), u_L, facets, 1)

# Traction on hole boundary (follower pressure)
P_hole = -1e4

(u, p) = TrialFunctions(W); (v, q) = TestFunctions(W)
F = inner(sigma(u, p), eps(v))*dx + div(u)*q*dx + p*div(v)*dx
a = lhs(F)
L = rhs(F)

# Add traction on hole boundary
n = FacetNormal(mesh)
L += P_hole*dot(v, n)*ds(5)

w = Function(W); solve(a == L, w, bc)
u_sol, p_sol = w.split()

# Postprocessing: von Mises stress
s = sigma(u_sol, p_sol) - (tr(sigma(u_sol, p_sol))/3.0)*Identity(2)
vonMises = project(sqrt(0.5*inner(s,s)), FunctionSpace(mesh,"CG",1))

# Visualization and output
u_mag = 5*u_sol
plot(mesh, title="Mesh with hole"); plot(u_mag, mode="displacement", title="q13_def.png")
plot(vonMises, title="q13_vm.png")

file = XDMFFile("q13_u.xdmf")
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True
file.write(u_sol, 0.0)
