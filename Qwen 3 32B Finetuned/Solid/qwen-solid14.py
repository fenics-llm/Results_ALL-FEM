from dolfin import *
from mshr import Rectangle, Circle, generate_mesh

# Geometry
rect = Rectangle(Point(0, 0), Point(1.0, 0.20))
circ1 = Circle(Point(0.40, 0.10), 0.04)
circ2 = Circle(Point(0.60, 0.10), 0.04)
mesh = generate_mesh(rect - circ1 - circ2, 64)

# Material
E, nu = 5e6, 0.49
mu = E / (2*(1+nu)); K = E / (3*(1-2*nu))

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W); (v, q) = TestFunctions(W)

# Variational form
def eps(w): return sym(grad(w))
def sigma(w, p): return 2*mu*eps(w) - p*Identity(2) + K*tr(eps(w))*Identity(2)
a = inner(sigma(u, p), eps(v))*dx + q*div(u)*dx + div(u)*q*dx
L = dot(Constant((0.0, 0.0)), v)*dx

# Boundary conditions
bcs = []
bcs.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), "near(x[0], 0.0)"))
bcs.append(DirichletBC(W.sub(0).sub(0), Constant(0.06), "near(x[0], 1.0)"))

# Hole traction
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class HoleBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[0]-0.40)**2 + (x[1]-0.10)**2 < 0.04**2 + 1e-10 or
                               (x[0]-0.60)**2 +
                     3*sigma(u_h, p_h)[0,1]**2)), FunctionSpace(mesh, "CG", 1))

# Save
File("q14_p.pvd") << p_h
File("q14_vm.pvd") << vm
File("q14_disp.pvd") << u_h
