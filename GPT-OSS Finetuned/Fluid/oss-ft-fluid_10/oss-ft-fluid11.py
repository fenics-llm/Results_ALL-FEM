# filename: q11_boussinesq.py
from dolfin import *
import matplotlib.pyplot as plt

# -------------------------------------------------
# Physical parameters
# -------------------------------------------------
rho   = 1.0          # reference density
mu    = 1.5e-5      # dynamic viscosity
alpha = 2.1e-5      # thermal diffusivity
gbeta = 3.15e-5     # buoyancy coefficient
T_ref = 0.5          # reference temperature

# -------------------------------------------------
# Mesh and boundary markers
# -------------------------------------------------
mesh = UnitSquareMesh(64, 64)

tol = 1e-14
class Left(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0, tol)
class Right(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[0], 1.0, tol)
class Bottom(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.0, tol)
class Top(SubDomain):
    def inside(self, x, on_boundary): return on_boundary and near(x[1], 1.0, tol)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
Left().mark(boundaries, 1)
Right().mark(boundaries, 2)
Bottom().mark(boundaries, 3)
Top().mark(boundaries, 4)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Mixed Taylor–Hood space (velocity, pressure, temperature)
# -------------------------------------------------
V_el = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # velocity
Q_el = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure
T_el = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # temperature
W_elem = MixedElement([V_el, Q_el, T_el])
W = FunctionSpace(mesh, W_elem)

# -------------------------------------------------
# Boundary conditions
# -------------------------------------------------
noslip = Constant((0.0, 0.0))
bcu = DirichletBC(W.sub(0), noslip, "on_boundary")

T_left  = Constant(1.0)
T_right = Constant(0.0)
bct_left  = DirichletBC(W.sub(2), T_left,  boundaries, 1)
bct_right = DirichletBC(W.sub(2), T_right, boundaries, 2)

# Pressure gauge at a corner (0,0) to fix pressure nullspace
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary): return near(x[0], 0.0, tol) and near(x[1], 0.0, tol)
bc_p = DirichletBC(W.sub(1), Constant(0.0), PressurePoint(), method="pointwise")

bcs = [bcu, bct_left, bct_right, bc_p]

# -------------------------------------------------
# Picard iteration for steady Boussinesq system
# -------------------------------------------------
w   = Function(W)   # mixed solution (u, p, T)
w0  = Function(W)   # previous iterate (initially zero)

# Trial and test functions
(U, P, Theta) = TrialFunctions(W)
(v, q, s)      = TestFunctions(W)

# Split previous iterate for convection term
(u0, p0, T0) = w0.split()

# Buoyancy force (depends on previous temperature)
f = as_vector((0.0, rho*gbeta*(T0 - T_ref)))

# Linearised Boussinesq system (Picard)
a = (mu*inner(grad(U), grad(v))
     + rho*dot(dot(u0, nabla_grad(U)), v)
     - div(v)*P
     + q*div(U)
     + alpha*dot(grad(Theta), grad(s))
     + dot(u0, nabla_grad(Theta))*s)*dx
L = dot(f, v)*dx

# Assemble matrix and RHS
A = PETScMatrix()
b = PETScVector()
assemble_system(a, L, bcs, A_tensor=A, b_tensor=b)

# Direct solver (MUMPS if available)
try:
    solver = LUSolver(A, "mumps")
except RuntimeError:
    solver = LUSolver(A)   # fallback to default LU

# Picard loop
tol_picard = 1e-8
max_iter   = 30
for it in range(max_iter):
    assemble_system(a, L, bcs, A_tensor=A, b_tensor=b)
    solver.solve(w.vector(), b)
    err = (w.vector() - w0.vector()).norm('l2')
    print(f"Picard iteration {it+1}: error = {err:.3e}")
    if err < tol_picard:
        break
    w0.assign(w)

# Split mixed solution into components
(u, p, T) = w.split()

# -------------------------------------------------
# Nusselt number at the left wall (x = 0)
# -------------------------------------------------
n = FacetNormal(mesh)
Nu = -alpha*dot(grad(T), n)          # heat flux = -α ∂T/∂n
Nu_avg = assemble(Nu*ds(1))
print(f"Average Nusselt number at left wall: {Nu_avg:.6f}")

# -------------------------------------------------
# Save results (XDMF format)
# -------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "q11_solution.xdmf") as xdmf:
    xdmf.write(u)
    xdmf.write(p)
    xdmf.write(T)

# -------------------------------------------------
# Plot temperature field
# -------------------------------------------------
plt.figure()
p = plot(T, title="Temperature")
plt.colorbar(p)
plt.savefig("q11_T.png", dpi=300)
plt.close()