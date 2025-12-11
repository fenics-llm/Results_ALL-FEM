# filename: q14_fenics.py
import numpy as np
import matplotlib.pyplot as plt
from dolfin import *
from ufl_legacy import sqrt, dot   # use legacy UFL for sqrt and dot

# ----------------------------------------------------------------------
# Geometry and mesh
# ----------------------------------------------------------------------
Lx, Ly = 1.0, 0.20
nx, ny = 128, 32                     # mesh resolution
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)

# ----------------------------------------------------------------------
# Subdomains (fluid = 0, porous = 1)
# ----------------------------------------------------------------------
class PorousDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (0.4 <= x[0] <= 0.6) and (0.0 <= x[1] <= Ly)

subdomains = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
porous = PorousDomain()
porous.mark(subdomains, 1)

# ----------------------------------------------------------------------
# Physical parameters
# ----------------------------------------------------------------------
rho = 1.0                # kg·m⁻³
mu  = 0.01               # Pa·s
K   = 1.0e-6             # m²
alpha_val = mu / K       # μ/K  (Darcy term coefficient)

# Piecewise coefficient α = μ/K in porous region, 0 elsewhere
DG0 = FunctionSpace(mesh, "DG", 0)
alpha = Function(DG0)
alpha_vals = np.where(subdomains.array() == 1, alpha_val, 0.0)
alpha.vector()[:] = alpha_vals

# ----------------------------------------------------------------------
# Function spaces (Taylor–Hood P2/P1)
# ----------------------------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)   # velocity (P2)
Q = FunctionSpace(mesh, "Lagrange", 1)         # pressure (P1)

# Build mixed space manually (compatible with all dolfin versions)
mixed_elem = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# ----------------------------------------------------------------------
# Boundary definitions
# ----------------------------------------------------------------------
U_bar = 1.0
H = Ly

inlet_profile = Expression(
    ("6.0*U_bar*x[1]*(H - x[1])/pow(H,2)", "0.0"),
    degree=2, U_bar=U_bar, H=H)

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], Lx) and on_boundary

inlet = Inlet()
walls  = Walls()
outlet = Outlet()

# Dirichlet BCs for velocity
bcu_inlet = DirichletBC(W.sub(0), inlet_profile, inlet)
bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)

# Pressure reference point (to fix the null‑space)
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], Lx) and near(x[1], H/2.0)

p_point = PressurePoint()
bc_p = DirichletBC(W.sub(1), Constant(0.0), p_point, method='pointwise')

bcs = [bcu_inlet, bcu_walls, bc_p]

# ----------------------------------------------------------------------
# Trial / test functions
# ----------------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# ----------------------------------------------------------------------
# Picard iteration – linearise convection with previous velocity
# ----------------------------------------------------------------------
u_k = Function(V)                # previous velocity (initially zero)
u_k.assign(Constant((0.0, 0.0)))  # explicit zero initialisation

# Weak form (steady Brinkman / Navier–Stokes)
F = (mu*inner(grad(u), grad(v))*dx
     + inner(dot(u_k, nabla_grad(u)), v)*dx   # convection linearised
     + alpha*inner(u, v)*dx
     - div(v)*p*dx
     - q*div(u)*dx)

a, L = lhs(F), rhs(F)

# ----------------------------------------------------------------------
# Solver (Picard loop)
# ----------------------------------------------------------------------
w = Function(W)                  # (u,p) solution
tol = 1e-6
maxiter = 30
for it in range(maxiter):
    solve(a == L, w, bcs, solver_parameters={"linear_solver": "mumps"})
    u_sol, p_sol = w.split()               # split into sub‑functions
    # compute error between current and previous velocity
    u_prev = u_k.vector().copy()
    u_k.interpolate(u_sol)                  # bring new velocity into V
    err = np.linalg.norm(u_k.vector().get_local() - u_prev.get_local(),
                         ord=np.Inf)
    print(f"Picard iteration {it+1}: max‑norm error = {err:.3e}")
    if err < tol:
        break

# ----------------------------------------------------------------------
# Save solution (XDMF)
# ----------------------------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q14_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.close()

# ----------------------------------------------------------------------
# Velocity magnitude plot → q14_speed.png
# ----------------------------------------------------------------------
Vmag = project(sqrt(dot(u_sol, u_sol)), FunctionSpace(mesh, "Lagrange", 2))

plt.figure(figsize=(8, 2))
p = plot(Vmag, title="Velocity magnitude |u| (m·s⁻¹)",
         cmap="viridis", shading='gouraud')
plt.colorbar(p)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q14_speed.png", dpi=300)
plt.close()

# ----------------------------------------------------------------------
# Pressure drop across porous block (centerline y = H/2)
# ----------------------------------------------------------------------
center_y = H/2.0
x_before = 0.399   # just left of porous region
x_after  = 0.601   # just right of porous region

p_before = p_sol(Point(x_before, center_y))
p_after  = p_sol(Point(x_after,  center_y))
dp = p_before - p_after

with open("q14_dp.txt", "w") as f:
    f.write(f"Pressure drop Δp = {dp:.6e} Pa\n")
print(f"Pressure drop Δp = {dp:.6e} Pa (saved in q14_dp.txt)")