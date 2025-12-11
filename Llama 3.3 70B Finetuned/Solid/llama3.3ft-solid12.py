from dolfin import *
import mshr

# Geometry: rectangular strip with circular hole
a = 0.04; Lx, Ly = 1.0, 0.20
hole = mshr.Circle(Point(0.50, 0.10), a)
domain = mshr.Rectangle(Point(0, 0), Point(Lx, Ly)) - hole
mesh = mshr.generate_mesh(domain, 64)

# Material parameters (plane strain)
E, nu = 5e3, 0.4999999 # Avoid division by zero
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Function space: P2 for u and P1 for p
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)

# Trial/test functions
du, dp = TestFunctions(W)
u_, p_ = TrialFunctions(W)

# Deformation gradient and Jacobian
I = Identity(2); F = I + grad(u_)
J = det(F)

# Incompressible neo-Hookean strain energy
psi = (mu / 2) * (tr(F.T * F) - 3) + lmbda * (0.5 * (J - 1)**2)
sigma = diff(psi, u_) # Corrected

# Variational form
L = inner(sigma, grad(du)) * dx
a = lhs(L)

# Boundary conditions
def left(x, on_boundary):
    return near(x[0], 0.0) and on_boundary
def right(x, on_boundary):
    return near(x[0], Lx) and on_boundary

bc_left = DirichletBC(W.sub(0), Constant((0.0, 0.0)), left)
u_R = Constant((0.06, 0.00))
bc_right = DirichletBC(W.sub(0), u_R, right)

bcs = [bc_left, bc_right]

# Solve
w = Function(W); solve(a == L, w, bcs)

# Split solution into displacement and pressure fields
u_h, p_h = w.split()

# Compute von Mises stress from Cauchy stress
s = sigma - tr(sigma) / 3 * Identity(2)
von_Mises = project(sqrt(0.5 * inner(s, s)), FunctionSpace(mesh, "CG", 1))

# Save results
file_p = File("q12_p.png"); file_vm = File("q12_vm.png")
plot(p_h, title="Hydrostatic pressure p (Pa)"); file_p << p_h
plot(von_Mises, title="von Mises stress (Pa)"); file_vm << von_Mises

# Save displacement field to XDMF format
file_u = XDMFFile("q12_displacement.xdmf")
file_u.parameters["flush_output"] = True
file_u.parameters["functions_share_mesh"] = True
file_u.write(u_h, 0.0)
