# q4_navier_stokes.py
from dolfin import *
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Geometry & physical parameters
# -------------------------------------------------
L, H   = 2.0, 0.20          # channel dimensions (m)
Ubar   = 2.5                 # mean inflow velocity (m/s)
mu, rho = 0.01, 1.0          # viscosity (Pa·s), density (kg/m³)

# -------------------------------------------------
# 2. Mesh
# -------------------------------------------------
nx, ny = 160, 16
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny)

# -------------------------------------------------
# 3. Mixed Taylor–Hood space (P2 velocity, P1 pressure)
# -------------------------------------------------
V_el = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # velocity (P2)
Q_el = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure (P1)
W_el = MixedElement([V_el, Q_el])
W    = FunctionSpace(mesh, W_el)

# -------------------------------------------------
# 4. Boundary definitions
# -------------------------------------------------
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary

inlet = Inlet()
outlet = Outlet()
walls = Walls()

# -------------------------------------------------
# 5. Inlet velocity (parabolic profile)
# -------------------------------------------------
u_inlet = Expression(("6.0*Ubar*x[1]/H*(1.0 - x[1]/H)", "0.0"),
                     Ubar=Ubar, H=H, degree=2)

# -------------------------------------------------
# 6. Dirichlet conditions
#    - velocity on inlet & walls
#    - pressure pinned at a point on the outlet (pointwise)
# -------------------------------------------------
bc_inlet = DirichletBC(W.sub(0), u_inlet, inlet)
bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and near(x[1], 0.0)
bc_p0    = DirichletBC(W.sub(1), Constant(0.0), PressurePoint(), method='pointwise')
bcs = [bc_inlet, bc_walls, bc_p0]

# -------------------------------------------------
# 7. Variational formulation (steady Navier–Stokes)
# -------------------------------------------------
w = Function(W)                     # unknown (u,p)
(u, p) = split(w)                  # current iterate
(v, q) = TestFunctions(W)

F = (2*mu*inner(sym(grad(u)), sym(grad(v)))*dx
     + rho*dot(dot(u, nabla_grad(u)), v)*dx
     - div(v)*p*dx
     - q*div(u)*dx)

J = derivative(F, w, TrialFunction(W))

# -------------------------------------------------
# 8. Nonlinear solve (Newton)
# -------------------------------------------------
solve(F == 0, w, bcs, J=J,
      solver_parameters={"newton_solver":
                         {"relative_tolerance": 1e-6,
                          "absolute_tolerance": 1e-8,
                          "maximum_iterations": 25,
                          "linear_solver": "lu"}})   # LU fallback if MUMPS unavailable

# -------------------------------------------------
# 9. Split mixed solution (deep copy) for post-processing
# -------------------------------------------------
u_sol, p_sol = w.split(deepcopy=True)   # u_sol ∈ V, p_sol ∈ Q

# -------------------------------------------------
# 10. Plot x-component of velocity
# -------------------------------------------------
ux = u_sol.sub(0)   # uₓ
plt.figure(figsize=(8, 2))
p = plot(ux, title=r"$u_x$", cmap="viridis")
plt.colorbar(p)
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.tight_layout()
plt.savefig("q4_ux.png", dpi=300)

# -------------------------------------------------
# 11. Save mesh and mixed solution to XDMF
# -------------------------------------------------
with XDMFFile("q4_soln.xdmf") as xdmf:
    xdmf.write(mesh)          # write mesh once
    xdmf.write(u_sol)         # mixed (u,p) field
    xdmf.write(p_sol)