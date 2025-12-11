# filename: elasticity_plate.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dolfin import *

# ----------------------------------------
# Geometry and mesh
# ----------------------------------------
Lx, Ly = 1.0, 0.20
nx, ny = 20, 4
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny, "crossed")

# ----------------------------------------
# Material parameters (plane stress)
# ----------------------------------------
E  = 200e9          # Pa
nu = 0.30
mu = E / (2.0*(1.0 + nu))
lmbda = 2.0*mu*nu/(1.0 - nu)   # plane stress

# ----------------------------------------
# Function space
# ----------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# ----------------------------------------
# Boundary definitions
# ----------------------------------------
tol = 1E-14
left  = CompiledSubDomain("near(x[0], side, tol)", side=0.0, tol=tol)
right = CompiledSubDomain("near(x[0], side, tol)", side=Lx,  tol=tol)

# ----------------------------------------
# Dirichlet BCs
# ----------------------------------------
zero_disp = Constant((0.0, 0.0))
right_disp = Expression(("0.001", "0.0"), degree=1)   # 1 mm horizontal stretch

bc_left  = DirichletBC(V, zero_disp, left)
bc_right = DirichletBC(V, right_disp, right)
bcs = [bc_left, bc_right]

# ----------------------------------------
# Variational problem (plane stress)
# ----------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda*tr(epsilon(u))*Identity(2) + 2.0*mu*epsilon(u)

a = inner(sigma(u), epsilon(v))*dx
L = dot(Constant((0.0, 0.0)), v)*dx   # no body forces

# ----------------------------------------
# Solve
# ----------------------------------------
u_sol = Function(V)
solve(a == L, u_sol, bcs, solver_parameters={"linear_solver":"mumps"})

# ----------------------------------------
# Output: XDMF
# ----------------------------------------
with XDMFFile(mesh.mpi_comm(), "plate_displacement.xdmf") as xdmf:
    xdmf.write(u_sol)

# ----------------------------------------
# Output: ux as PNG
# ----------------------------------------
ux = u_sol.sub(0, deepcopy=True)
plt.figure(figsize=(6, 2.5))
p = plot(ux, title=r"$u_x$ (m)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q1_ux.png", dpi=300)