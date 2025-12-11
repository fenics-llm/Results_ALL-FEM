# filename: q11_linear_elasticity.py
from dolfin import *
import mshr
import matplotlib.pyplot as plt

# -------------------------------------------------
# Geometry: rectangle (0,1) x (0,0.20) with a hole radius 0.04 at (0.5,0.1)
rect = mshr.Rectangle(Point(0.0, 0.0), Point(1.0, 0.20))
hole = mshr.Circle(Point(0.5, 0.10), 0.04)
domain = rect - hole
mesh = mshr.generate_mesh(domain, 80)   # increase resolution if needed

# -------------------------------------------------
# Material parameters (plane strain, nearly incompressible)
E  = 5.0e6          # Pa (5 MPa)
nu = 0.49
mu    = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# -------------------------------------------------
# Mixed Taylor–Hood space (P2 for u, P1 for p)
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# -------------------------------------------------
# Boundary markers
tol = 1E-6
class Left(SubDomain):
    def inside(self, x, on):
        return on and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on):
        return on and near(x[0], 1.0, tol)
left  = Left()
right = Right()

# -------------------------------------------------
# Dirichlet BCs for displacement
zero_disp = Constant((0.0, 0.0))
bc_left   = DirichletBC(W.sub(0), zero_disp, left)

right_disp = Expression(("0.001", "0.0"), degree=1)
bc_right  = DirichletBC(W.sub(0), right_disp, right)

# Pressure gauge: fix pressure at a single interior point (here the lower‑left corner)
class PressurePoint(SubDomain):
    def inside(self, x, on):
        return near(x[0], 0.0, tol) and near(x[1], 0.0, tol)
bc_pressure = DirichletBC(W.sub(1), Constant(0.0), PressurePoint(), method="pointwise")

bcs = [bc_left, bc_right, bc_pressure]

# -------------------------------------------------
# Mixed variational formulation
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

def sigma(u, p):
    return 2.0*mu*sym(grad(u)) - p*Identity(2)

a = (inner(sigma(u, p), sym(grad(v))) - div(v)*p - q*div(u))*dx
L = Constant(0.0)*dot(v, Constant((0.0, 0.0)))*dx   # zero traction on Neumann boundaries

# -------------------------------------------------
# Solve
w = Function(W)
solve(a == L, w, bcs, solver_parameters={"linear_solver": "lu"})

# Split solution
(u_h, p_h) = w.split(deepcopy=True)

# -------------------------------------------------
# Compute von Mises stress
s = sigma(u_h, p_h) - (1.0/3.0)*tr(sigma(u_h, p_h))*Identity(2)   # deviatoric part
von_Mises = sqrt(3.0/2.0*inner(s, s))
Vsig = FunctionSpace(mesh, "P", 1)
vonMises_h = project(von_Mises, Vsig)

# -------------------------------------------------
# Plot and save figures
plt.figure()
p = plot(vonMises_h, title="Von Mises stress (Pa)", cmap="viridis")
plt.colorbar(p)
plt.savefig("q11_vm.png", dpi=300)

plt.figure()
p = plot(u_h[0], title="Horizontal displacement u_x (m)", cmap="viridis")
plt.colorbar(p)
plt.savefig("q11_ux.png", dpi=300)

# -------------------------------------------------
# Save displacement field (u_h) in XDMF format
with XDMFFile(mesh.mpi_comm(), "q11_displacement.xdmf") as xdmf:
    xdmf.write(u_h)

print("Computation finished. Files generated: q11_vm.png, q11_ux.png, q11_displacement.xdmf")