from dolfin import *
import mshr

# Problem 7
domain = mshr.Rectangle(Point(0, 0), Point(2.2, 0.41)) - mshr.Circle(Point(0.20, 0.20), 0.05)
mesh = mshr.generate_mesh(domain, 64)

V = VectorFunctionSpace(mesh, "CG", 2); Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()]); W = FunctionSpace(mesh, ME)

f = Constant((0.0, 0.0))
u, p = TrialFunctions(W); v, q = TestFunctions(W)
mu = Constant(0.001); rho = Constant(1.0); Ubar = Constant(0.2)
H = Constant(0.41)
x, y = SpatialCoordinate(mesh)

# Boundary markers
bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bndry.set_all(0)
class Inlet(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0)
Inlet().mark(bndry, 1)
class Outlet(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 2.2)
Outlet().mark(bndry, 2)
class Walls(SubDomain):
    def inside(self, x, on_boundary): return (on_boundary
        and ((near(x[1], 0.0)) or (near(x[1], H))))
Walls().mark(bndry, 3)
class Circle(SubDomain):
    def inside(self, x, on_boundary): return on_boundary \
        and near((x[0]-0.20)**2 + (x[1]-0.20)**2, 0.05**2)
Circle().mark(bndry, 4)

# Boundary conditions
bc_inlet = DirichletBC(W.sub(0),
    Expression(("6*Ubar*x[1]*(H-x[1])/pow(H,2)", "0.0"),
        Ubar=Ubar, H=H, degree=3), bndry, 1)
bcs = [bc_inlet,
    DirichletBC(W.sub(0).sub(0), Constant(0.0), bndry, 3),
    DirichletBC(W.sub(0).sub(1), Constant(0.0), bndry, 3),
    DirichletBC(W.sub(0).sub(0), Constant(0.0), bndry, 4),
    DirichletBC(W.sub(0).sub(1), Constant(0.0), bndry, 4)]

# Variational form
a = (mu*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L = inner(f, v)*dx

w = Function(W); solve(a == L, w, bcs)
u_soln, p_soln = w.split()

# Drag force on the circle
n = FacetNormal(mesh)
t = (-mu*grad(u_soln) + p_soln*n)
F_D = assemble(dot(t, as_vector((1.0, 0.0))) * ds(4))
C_D = F_D / (rho * Ubar**2 * 2*0.05)
print("Drag coefficient:", C_D)

# Visualization
u_mag = project(sqrt(dot(u_soln, u_soln)), FunctionSpace(mesh, "CG", 1))
plot_object = plot(u_mag, title="|u|")
plt.savefig("q7_speed.png")

file = XDMFFile("q7_soln.xdmf")
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True
file.write(u_soln, 0.0); file.write(p_soln, 0.0)

# Problem 9
mesh = RectangleMesh(Point(0, 0), Point(1, 0.20), 128, 32)
V = VectorFunctionSpace(mesh, "CG", 2); Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()]); W = FunctionSpace(mesh, ME)

f = Constant((1.0, 0.0))
u, p = TrialFunctions(W); v, q = TestFunctions(W)
mu = Constant(0.01); rho = Constant(1.0)
x, y = SpatialCoordinate(mesh)

# Boundary markers
bndry = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
bndry.set_all(0)
class Gamma_y0(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.0)
Gamma_y0().mark(bndry, 1)
class Gamma_yH(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.20)
Gamma_yH().mark(bndry, 2)

# Boundary conditions
bcs = [
    DirichletBC(W.sub(0).sub(0), Expression("x[0]", degree=3),
        "near(x[0], x1)", "x1", 0.0),
    DirichletBC(W.sub(0).sub(1), Constant(0.0), bndry, 1),
    DirichletBC(W.sub(0).sub(1), Constant(0.0), bndry, 2),
    DirichletBC(W.sub(1), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)", "point"),
]

# Variational form
a = (mu*inner(grad(u), grad(v)) - div(v)*p + q*div(u))*dx
L = inner(f, v)*dx

w = Function(W); solve(a == L, w, bcs)
u_soln, p_soln = w.split()

file = XDMFFile("q9_soln.xdmf")
file.parameters["flush_output"] = True
file.parameters["functions_share_mesh"] = True
file.write(u_soln, 0.0); file.write(p_soln, 0.0)
