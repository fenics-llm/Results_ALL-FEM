# file: q1_plane_stress_elasticity.py
from dolfin import *
import matplotlib.pyplot as plt

# ---- Geometry & mesh ----
L, H = 1.0, 0.20
nx, ny = 20, 4
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny)

# ---- Function space ----
V = VectorFunctionSpace(mesh, "CG", 1)

# ---- Material (plane stress) ----
E  = 200e9      # Young's modulus [Pa]
nu = 0.30       # Poisson's ratio [-]

mu  = E/(2.0*(1.0 + nu))              # shear modulus
lam_ps = 2.0*mu*nu/(1.0 - nu)         # plane-stress effective lambda

def eps(u):
    return sym(grad(u))

def sigma(u):
    return 2.0*mu*eps(u) + lam_ps*tr(eps(u))*Identity(2)

# ---- Boundary conditions ----
tol = 1e-14

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, tol)

left = LeftBoundary()
right = RightBoundary()

u_L = Constant((0.0, 0.0))        # fixed on left
u_R = Constant((0.001, 0.0))      # prescribed displacement on right (ux=1e-3 m, uy=0)

bc_left  = DirichletBC(V, u_L, left)
bc_right = DirichletBC(V, u_R, right)
bcs = [bc_left, bc_right]

# ---- Variational problem ----
u = TrialFunction(V)
v = TestFunction(V)
f = Constant((0.0, 0.0))          # no body force

a = inner(sigma(u), eps(v))*dx
Lform = dot(f, v)*dx              # traction-free on top/bottom is natural

# ---- Solve ----
uh = Function(V, name="displacement")
solve(a == Lform, uh, bcs, solver_parameters={"linear_solver": "mumps"})

# ---- Outputs ----
# 1) Save displacement to XDMF
with XDMFFile(mesh.mpi_comm(), "q1_displacement.xdmf") as xdmf:
    xdmf.write(mesh)
    xdmf.write(uh)

# 2) Save colour map of u_x
ux_space = FunctionSpace(mesh, "CG", 1)
ux = project(uh.sub(0), ux_space)
ux.rename("u_x", "u_x")

plt.figure()
c = plot(ux)               # FEniCS plot returns a mappable for colourbar
plt.colorbar(c)
plt.title(r"Horizontal displacement $u_x$ (m)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q1_ux.png", dpi=220)
plt.close()

# Optional: also save PVD for quick Paraview checks
# File("q1_displacement.pvd") << uh