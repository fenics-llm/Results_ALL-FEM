# filename: q12_neo_hookean.py
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt

# -------------------------------------------------
# Geometry and mesh
L, H = 1.0, 0.20
a, xc, yc = 0.04, 0.50, 0.10
domain = Rectangle(Point(0.0, 0.0), Point(L, H)) - Circle(Point(xc, yc), a, 64)
mesh = generate_mesh(domain, 64)

# -------------------------------------------------
# Material parameters (plane strain, incompressible)
E, nu = 5e6, 0.5
mu = E / (2.0 * (1.0 + nu))

# -------------------------------------------------
# Mixed Taylor–Hood space (P2 for u, P1 for p)
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# -------------------------------------------------
# Boundary conditions
tol = 1E-8
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, tol)

left  = Left()
right = Right()
bcs = [DirichletBC(W.sub(0), Constant((0.0, 0.0)), left),
       DirichletBC(W.sub(0), Constant((0.060, 0.0)), right)]

# -------------------------------------------------
# Mixed formulation (u,p)
w = Function(W)               # current solution
(u, p) = split(w)             # trial functions
(v, q) = TestFunctions(W)     # test functions

I = Identity(2)               # 2D identity tensor
F = I + grad(u)                # deformation gradient
J = det(F)                     # Jacobian

# Cauchy stress for incompressible neo‑Hookean (plane strain)
sigma = mu*F*F.T - p*I

# Weak form (balance of linear momentum + incompressibility constraint)
R = inner(sigma, grad(v))*dx + q*(J - 1.0)*dx

# Jacobian
J_form = derivative(R, w)

# -------------------------------------------------
# Newton solver
problem = NonlinearVariationalProblem(R, w, bcs, J_form)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["newton_solver"]["relative_tolerance"] = 1e-6
prm["newton_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["linear_solver"] = "mumps"
solver.solve()

# -------------------------------------------------
# Extract fields
(u_h, p_h) = w.split(deepcopy=True)

# Von Mises stress (plane strain)
sigma_h = mu*F*F.T - p_h*I
sigma_dev = sigma_h - (1.0/2.0)*tr(sigma_h)*I
von_mises = sqrt(3.0/2.0*inner(sigma_dev, sigma_dev))
V1 = FunctionSpace(mesh, "P", 1)
vm_h = project(von_mises, V1)

# -------------------------------------------------
# Output: PNG plots
plt.figure()
p_plot = plot(p_h, title="Hydrostatic pressure")
plt.colorbar(p_plot)
plt.savefig("q12_p.png", dpi=300)

plt.figure()
vm_plot = plot(vm_h, title="Von Mises stress")
plt.colorbar(vm_plot)
plt.savefig("q12_vm.png", dpi=300)

# -------------------------------------------------
# Output: displacement XDMF
xdmf = XDMFFile(mesh.mpi_comm(), "q12_disp.xdmf")
xdmf.write(u_h)
xdmf.close()