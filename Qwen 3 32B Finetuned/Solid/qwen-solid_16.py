from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import numpy as np

# Geometry
outer = Rectangle(Point(0, 0), Point(100, 180))
hole = Circle(Point(0, 0), 50)
mesh = generate_mesh(outer - hole, 64)

# Function spaces
V = VectorFunctionSpace(mesh, "CG", 1)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, e_p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Parameters
E, nu = 70e9, 0.33
lambda_ = E*nu/(1-nu**2)
mu = E/(2*(1+nu))
sigma_Y = 243e6

# Strain measures
def eps(w):
    return sym(grad(w))
def e_dev(w, e_p):
    return eps(w) - (1/3)*tr(eps(w))*Identity(2) - e_p

# Stress
def sigma(w, e_p):
    e = e_dev(w, e_p)
    q = sqrt(3/2*inner(e, e))
    return (lambda_ + 2/3*mu)*tr(eps(w))*Identity(2) + 2*mu*e

# Variational forms
a = inner(sigma(u, e_p), grad(v))*dx + q*inner(e_dev(u, e_p), e_dev(u, e_p))*dx
L = Constant((0, 0))*v*dx

# Boundary conditions
bcs = []
# Symmetry x=0: u_x=0, zero traction in y
bcs.append(DirichletBC(W.sub(0).sub(0), Constant(0.0), "near(x[0], 0.0)"))
# Symmetry y=0: u_y=0, zero traction in x
bcs.append(DirichletBC(W.sub(0).sub(1), Constant(0.0), "near(x[1], 0.0)"))
# Top displacement
bcs.append(DirichletBC(W.sub(0), Constant((0.0, 1.0)), "near(x[1], 180.0)"))

# Solve
w = Function(W)
solve(a == L, w, bcs, solver_parameters={"newton_solver": {"maximum_iterations": 50}})
u_sol = w.sub(0)

# Save
File("displacement.pvd") << u_sol