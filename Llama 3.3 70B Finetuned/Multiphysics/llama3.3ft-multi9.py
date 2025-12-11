from dolfin import *
from mshr import *

# Geometry
L = 2.5; H = 0.41; flag_L = 0.35; flag_h = 0.02
pole_r = 0.05

domain = Rectangle(Point(0, 0), Point(L, H)) - Circle(Point(0.20, 0.20), pole_r)
flag_domain = Rectangle(Point(0.60 - flag_L, 0.19 - flag_h/2),
                        Point(0.60, 0.19 + flag_h/2))
domain -= flag_domain

mesh = generate_mesh(domain, 64)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Boundary markers
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
class Inlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0) and on_boundary
Inlet().mark(boundaries, 1)
class Outlet(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], L) and on_boundary
Outlet().mark(boundaries, 2)
class Top(SubDomain):
    def inside(self, x, on_boundary): return near(x[1], H) and on_boundary
Top().mark(boundaries, 3)
class Bottom(SubDomain):
    def inside(self, x, on_boundary): return near(x[1], 0.0) and on_boundary
Bottom().mark(boundaries, 4)

# Boundary conditions
u_in = Expression(("6*Ubar*x[1]*(H - x[1])/(H/2)*(H/2)", "0.0"),
                  Ubar=0.2, H=H, degree=2)
noslip = Constant((0.0, 0.0))
bc_inlet = DirichletBC(W.sub(0), u_in, boundaries, 1)

# Material parameters
rho_f = 1000.0; nu_f = 1e-3
f_body = Constant((0.0, -rho_f * 9.81))
mu_s = 5e5; lambda_s = mu_s * (2 * 0.4) / (1 - 2 * 0.4)
rho_s = 1000.0

# Variational formulation
u, p = TrialFunctions(W); v, q = TestFunctions(W)

def sigma_f(u, p):
    return -p * Identity(2) + 2 * nu_f * sym(grad(u))

def epsilon(u): return (grad(u) + grad(u).T) / 2

def sigma_s(u):
    I = Identity(2)
    E = epsilon(u)
    return lambda_s * tr(E) * I + 2 * mu_s * E

F_fluid = rho_f * inner((u - Constant((0.0, 0.0))), v) * dx \
          - div(v) * p * dx \
          + inner(sigma_f(u, p), grad(v)) * dx \
          - dot(f_body, v) * dx
F_solid = rho_s * inner(Dot(u, n), Dot(v, n)) * ds

# Solve steady problem
w = Function(W)
solve(lhs(F_fluid) == rhs(F_fluid), w)

u_h, p_h = w.split()
File("velocity.pvd") << u_h
File("pressure.pvd") << p_h
