from dolfin import *
from mshr   import *

# Geometry: quarter plate with hole
Lx, Ly = 100., 180.
r0    = 50.
domain = Rectangle(Point(0,0), Point(Lx, Ly)) - Circle(Point(0,0), r0)
mesh   = generate_mesh(domain, 64)

# Materials
E, nu = Constant(58.2e3), Constant(0.29) # Plane strain
mu    = E/(2*(1+nu))
lmbda = E*nu/((1+nu)*(1-2*nu))

sigma_Y = Constant(243.)

V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)

ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W  = FunctionSpace(mesh, ME)

du, dp = TrialFunctions(W)
u_, p_ = TestFunctions(W)

w  = Function(W); u, p = split(w) # p is the plastic multiplier

# Initial conditions
u0 = interpolate(Constant((0.0,0.0)), V)
p0 = interpolate(Constant(0.), Q)
w_n = Function(W); w_n.assign(interpolate(Expression(("u0[0]", "u0[1]"), u0=u0, degree=2), W))
u_n, p_n = split(w_n)

# Strain and stress
def eps(u):   return 0.5*(grad(u) + grad(u).T)
def sigma(u):
    e = eps(u); ee = e - (1/3)*tr(e)*Identity(2)
    return lmbda*tr(e)*Identity(2) + 2*mu*ee

# Return mapping
def project(s, p, sigma_Y):
    s_dev = s - (1/3)*tr(s)*Identity(2)
    q     = sqrt((3/2.0)*(s_dev[i,j]*s_dev[i,j]))
    beta  = conditional(lt(q + mu*p, sigma_Y), 0.0,
                        (q + mu*p - sigma_Y)/(mu*(3.0)))
    return s_dev - beta*s_dev

# Variational formulation
F_int = inner(sigma(u), eps(v)) * dx \
      + p*div(v)*dx \
      + q*inner(grad(p), grad(q))*dx \
      - dot(project(sigma(u_n) - sigma_Y, p_n, sigma_Y),
            grad(q))*dx

a, L = system(F_int)

# Boundary conditions
tol = 1e-10
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class Left(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0, tol)
class Bottom(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.0, tol)

Left().mark(facets, 1); Bottom().mark(facets, 2)
ds = Measure("ds", domain=mesh, subdomain_data=facets)

bc_left   = DirichletBC(W.sub(0).sub(0), Constant(0.), facets, 1)
bc_bottom = DirichletBC(W.sub(0).sub(1), Constant(0.), facets, 2)
bc_top    = DirichletBC(W.sub(0), Constant((0.0,-1.0)), facets, 3)

bcs = [bc_left, bc_bottom, bc_top]

A, b = assemble_system(a, L, bcs)
solve(A, w.vector(), b, "gmres", "ilu")

u_h, p_h = w.split()
File("plate_u.xml") << u_h
File("plate_p.xml") << p_h
