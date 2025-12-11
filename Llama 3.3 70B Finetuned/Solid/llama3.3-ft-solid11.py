from dolfin import *
from mshr   import *

# Geometry: rectangular strip with circular hole
L, H = 1.0, 0.20; a = 0.04
domain = Rectangle(Point(0, 0), Point(L, H)) - Circle(Point(0.5, 0.10), a)
mesh = generate_mesh(domain, 64)

# Material (plane strain, nearly incompressible)
E, nu = 5e6, 0.49; mu = E / (2 * (1 + nu)); lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Mixed displacement–pressure formulation
V = VectorFunctionSpace(mesh, "CG", 2); Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()]); W = FunctionSpace(mesh, ME)

def eps(u): return sym(grad(u))
def sigma(w):
    u, p = w.split()
    return lmbda * tr(eps(u)) * Identity(2) + 2 * mu * eps(u) - p * Identity(2)
def left(x, on_boundary): return near(x[0], 0.0) and on_boundary
def right(x, on_boundary): return near(x[0], L) and on_boundary

# Boundary conditions
bc_left = DirichletBC(W.sub(0), Constant((0.0, 0.0)), left)
bc_right = DirichletBC(W.sub(0).sub(1), Constant(0.001), right)

w = Function(W); (u, p) = TrialFunctions(W); (v, q) = TestFunctions(W)
a_form = inner(sigma(w), eps(v)) * dx + q * div(u) * dx + div(v) * p * dx
L_form = Constant(0.0) * v[1] * dx

bcs = [bc_left, bc_right]
solve(a_form == L_form, w, bcs)

u_h, p_h = w.split()
vm = project(sqrt(0.5*((sigma(w)[0, 0]-sigma(w)[1, 1])**2 + 3*sigma(w)[0, 1]**2)),
             FunctionSpace(mesh, "CG", 1))
plot(vm, title="von Mises stress"); interactive()
File("q11_vm.png") << vm
plot(u_h[0], title="Horizontal displacement u_x"); interactive()
File("q11_ux.png") << u_h[0]
File("q11_u.xml") << u_h