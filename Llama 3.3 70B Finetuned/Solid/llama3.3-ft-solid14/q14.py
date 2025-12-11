from dolfin import *
from mshr   import *

# Geometry: rectangular strip with two circular holes
Lx, Ly = 1.0, 0.20
a     = 0.04
hole1 = Circle(Point(0.40, 0.10), a)
hole2 = Circle(Point(0.60, 0.10), a)
domain = Rectangle(Point(0,0), Point(Lx,Ly)) - hole1 - hole2

mesh = generate_mesh(domain, 64)

# Function space: P2 for displacement and P1 for pressure
V = VectorFunctionSpace(mesh, "CG", 2); Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()]); W = FunctionSpace(mesh, ME)

def eps(u):   return sym(grad(u))
def sigma(u,p,E,nu):
    mu     = E/(2*(1+nu)); lmbda = E*nu/((1+nu)*(1-2*nu))
    return 2*mu*eps(u) + lmbda*tr(eps(u))*Identity(2), p

def left(x, on_boundary):   return near(x[0], 0.0)
def right(x, on_boundary):  return near(x[0], Lx)
def top_bottom(x,on_b):
    return (near(x[1], Ly) or near(x[1], 0)) and on_b
def hole1bndry(x,on_b):
    return on_b and sqrt((x[0]-0.40)**2 + (x[1]-0.10)**2)<a+1e-12
def hole2bndry(x,on_b):
    return on_b and sqrt((x[0]-0.60)**2 + (x[1]-0.10)**2)<a+1e-12

# Boundary conditions
bc_left = DirichletBC(W.sub(0), Constant((0, 0)), left)
u_r     = Constant((0.06, 0)); bc_right = DirichletBC(W.sub(0), u_r, right)

bcs = [bc_left, bc_right]

# Material parameters
E, nu = 5e6, 0.49; mu, lmbda = E/(2*(1+nu)), E*nu/((1+nu)*(1-2*nu))
P_hole = -10000

(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

n = FacetNormal(mesh)
F_int = inner(sigma(u,p,E,nu)[0], eps(v))*dx + p*q*dx
ds = Measure("ds", domain=mesh)
F_ext = P_hole*(v[0]*n[0] + v[1]*n[1])*(ds(4)+ds(5))

a_form = lhs(F_int)
L_form = rhs(F_int) + F_ext

w = Function(W); solve(a_form == L_form, w, bcs)

u_sol, p_sol = w.split()

# von Mises stress
s = sigma(u_sol,p_sol,E,nu)[0] - (1/3)*tr(sigma(u_sol,p_sol,E,nu)[0])*Identity(2)
vonMises = project(sqrt(0.5*((s[0,0]-s[1,1])**2 + 3*(s[0,1]**2))), Q)

# Save results
plot(p_sol, title="Pressure p (Pa)"); plot(vonMises, title="von Mises stress (Pa)")

# Save displacement in XDMF format
file_u = XDMFFile("q14_displacement.xdmf")
file_u.parameters["flush_output"] = True
file_u.parameters["functions_share_mesh"] = True
file_u.write(u_sol, 0)