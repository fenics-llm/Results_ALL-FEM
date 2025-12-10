# Geometry
from mshr   import Rectangle, Circle, generate_mesh
import numpy as np
from dolfin   import Point, PointList, plot, plot_parameters, MeshFunction, subdomain_boundary, FunctionSpace, DirichletBC
from fenics  import Constant, Expression, interpolate, assemble, solve, inner, grad, dx, ds, FiniteElement, VectorElement, TrialFunctions, TestFunctions

def hole(x):
    return (x[0] - Point(0.30, 0.10))**2 + x[1]**2 < a**2
a = 0.03
mesh = generate_mesh(
    Rectangle(Point(0,0), Point(1.20,0.20)),
    lambda x,on: on and hole(x),
)

# Material
E, nu = 5.769e9, 0.3
mu, lmbda = mu, lmbda = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
def sigma(S):
    return lmbda*tr(S)*Identity(2) + 2*mu*S

# Variational form
u, v = TrialFunctions(w), TestFunctions(w)
F = Identity(2) + grad(u)
E = 0.5*(F.T*F - Identity(2))
S = sigma(E)
J = det(F)

def eps(v):
    return 0.5*(grad(v)+grad(v).T)

# Boundary conditions
bcs = [
    DirichletBC(w, Constant((0.012,0)), subdomain_boundary(mesh), 1),
]

# Variational form
a = inner(sigma(E), eps(v))*dx
L = Constant(0.0)*ds(2)
solve(a == L, u, bcs)

# Principal Green–Lagrange strains
E_max = Function(w); E_max.vector()[:] = 0.0
for i in range(3):
    e_i = interpolate(Expression("sqrt(eps(v)[0][0] + eps(v)[1][1])", v=u), w)
    E_max.vector()[i] = e_i.compute_vertex_values(mesh).max()
plot(E_max, title="q15_Emax"); plot_parameters()["colorbar"] = False; plot(); interactive()

# VM stress
s = sigma(E) - 0.5*tr(sigma(E))*Identity(2)
vmS = Function(w); vmS.vector()[:] = 0.0
for i in range(3):
    vmS.vector()[i] = sqrt(inner(s, s))
plot(vmS, title="q15_vmS"); plot_parameters()["colorbar"] = False; plot(); interactive()

# Export displacement and E_max
u_file = File("q15_u.pvd")
Emax_file = File("q15_Emax.pvd")
