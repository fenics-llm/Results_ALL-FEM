from dolfin import *
from mshr import Rectangle, generate_mesh
import numpy as np

# Geometry
mesh = generate_mesh(Rectangle(Point(0, -1), Point(np.pi, 1)), 64)
facets = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
class StokesInterface(SubDomain):
    def inside(self, x, on_b):
        return on_b and near(x[1], 0.0)
class DarcyInterface(SubDomain):
    def inside(self, x, on_b):
        return on_b and near(x[1], 0.0)
class StokesBoundary(SubDomain):
    def inside(self, x, on_b):
        return on_b and (near(x[1], 1.0) or near(x[0], 0.0) or near(x[0], np.pi))
class DarcyBoundary(SubDomain):
    def inside(self, x, on_b):
        return on_b and near(x[1], -1.0)
StokesInterface().mark(facets, 1)
DarcyInterface().mark(facets, 2)
StokesBoundary().mark(facets, 3)
DarcyBoundary().mark(facets, 4)
ds = Measure("ds", domain=mesh, subdomain_data=facets)
n = FacetNormal(mesh)

# Mixed space
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Parameters
g, rho, nu, k, K, alpha = 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
mu = 1.0
w = Expression("K - (g*x[1])/(2*nu) + (K/2 - alpha*g/(4*nu*nu))*x[1]*x[1]", degree=2,
               K=K, g=g, nu=nu, alpha=alpha)
b_x = Expression("(nu*K - alpha*g/(2*nu))*x[1] - g/2)*cos(x[0])", degree=2,
                 nu=nu, K=K, alpha=alpha, g=g)
b_y = Expression("((nu*K/2 - alpha*g/(4*nu))*x[1]*x[1] - (g/2)*x[1] + (alpha*g/(2*nu) - 2*nu*K))*sin(x[0])",
                 degree=2, nu=nu, K=K, alpha=alpha, g=g)

# Variational forms
a = inner(grad(u), grad(v))*dx + div(v)*p*dx + div(u)*q*dx
L = dot(b_x, v[0])*dx + dot(b_y, v[1])*dx

# Boundary conditions
u_D = Expression(("w_x", "w_y"), degree=2,
                 w_x=Expression("cos(x[0])*(K - (g*x[1])/(2*nu) + (K/2 - alpha*g/(4*nu*nu))*x[1]*x[1])",
                                degree=2, K=K, g=g, nu=nu, alpha=alpha),
                 w_y=Expression("sin(x[0])*(K - (g*x[1])/(2*nu) + (K/2 - alpha*g/(4*nu*nu))*x[1]*x[1])",
                                degree=2, K=K, g=g, nu=nu, alpha=alpha))
bc_stokes = DirichletBC(W.sub(0), u_D, facets, 3)
p_D = Expression("rho*g*exp(x[1])*sin(x[0])", degree=2, rho=rho, g=g)
bc_darcy = DirichletBC(W.sub(1), p_D, facets, 4)

# Interface conditions
t = as_vector((1.0, 0.0))
u_S = u("+")
u_D = u("-")
p_S = p("+")
p_D = p("-")
n = FacetNormal(mesh)
a += (dot(n, grad(u_S))*v("+") - dot(n, grad(u_D))*v("-"))*ds(1)
a += (dot(n, grad(u_S))*n*v("+") + p_D*q("+"))*ds(1)
a += (alpha/sqrt(k)*(dot(u_S - u_D, t)*v("+") + dot(n, grad(u_S))*t*q("+")))*ds(1)

# Solve
w_sol = Function(W)
solve(a == L, w_sol, [bc_stokes, bc_darcy])

# Save
u_sol, p_sol = w_sol.split()
u_file = XDMFFile("stokes_velocity.xdmf")
u_file.write(u_sol)
p_file = XDMFFile("darcy_pressure.xdmf")
p_file.write(p_sol)