# filename: q10_fenics.py
from dolfin import *

# -------------------------------------------------
# Parameters
# -------------------------------------------------
Lx, Ly = 2.0, 0.20          # domain size (m)
nx, ny = 200, 20            # mesh resolution
U_bar = 0.1                 # mean inlet velocity (m/s)
H = Ly                      # channel height (m)
rho = 1.0                   # density (kg/m³)
mu = 0.01                   # dynamic viscosity (Pa·s)
kappa = 1.0e-3              # diffusivity (m²/s)

# -------------------------------------------------
# Mesh
# -------------------------------------------------
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)

# -------------------------------------------------
# Boundary definitions
# -------------------------------------------------
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], Lx) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], Ly)) and on_boundary

inlet = Inlet()
outlet = Outlet()
walls = Walls()

# -------------------------------------------------
# Function spaces (Taylor–Hood)
# -------------------------------------------------
V_el = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # velocity element (P2)
Q_el = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure element (P1)
W = FunctionSpace(mesh, MixedElement([V_el, Q_el]))

# -------------------------------------------------
# Boundary conditions for Navier–Stokes
# -------------------------------------------------
# Parabolic inlet velocity profile
inlet_profile = Expression(
    ("6.0*U_bar*(x[1]/H)*(1.0 - x[1]/H)", "0.0"),
    degree=2, U_bar=U_bar, H=H)

bc_inlet = DirichletBC(W.sub(0), inlet_profile, inlet)
bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)

# Fix pressure at a single point (outlet corner) to remove null‑space
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], Lx) and near(x[1], 0.0)

p_point = PressurePoint()
bc_pressure = DirichletBC(W.sub(1), Constant(0.0), p_point, method='pointwise')

bcs_ns = [bc_inlet, bc_walls, bc_pressure]

# -------------------------------------------------
# Variational formulation (steady Navier–Stokes)
# -------------------------------------------------
w = Function(W)                     # unknown (u,p)
(u, p) = split(w)                   # for readability in the residual
(v, q) = TestFunctions(W)           # test functions

# Residual of the steady Navier–Stokes equations
F = (mu*inner(grad(u), grad(v)) * dx
     + rho*inner(dot(u, nabla_grad(u)), v) * dx
     - div(v)*p * dx
     - q*div(u) * dx)

# Jacobian (derivative of F w.r.t. w)
J = derivative(F, w, TrialFunction(W))

# -------------------------------------------------
# Solve Navier–Stokes (non‑linear)
# -------------------------------------------------
ns_problem = NonlinearVariationalProblem(F, w, bcs_ns, J)
ns_solver = NonlinearVariationalSolver(ns_problem)
ns_solver.parameters["newton_solver"]["relative_tolerance"] = 1e-6
ns_solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-8
ns_solver.parameters["newton_solver"]["maximum_iterations"] = 25
ns_solver.parameters["newton_solver"]["linear_solver"] = "mumps"
ns_solver.solve()

# Extract velocity and pressure as separate Functions
(u_sol, p_sol) = w.split(deepcopy=True)

# -------------------------------------------------
# Advection–Diffusion for concentration
# -------------------------------------------------
V_c = FunctionSpace(mesh, "Lagrange", 1)
c = TrialFunction(V_c)
v_c = TestFunction(V_c)

# Dirichlet BCs for concentration
c_inlet = Constant(0.0)
c_outlet = Constant(1.0)

bc_c_inlet = DirichletBC(V_c, c_inlet, inlet)
bc_c_outlet = DirichletBC(V_c, c_outlet, outlet)
bcs_c = [bc_c_inlet, bc_c_outlet]

# Weak form: κ∇c·∇v + ρ u·∇c v = 0
a_c = (kappa*dot(grad(c), grad(v_c)) + rho*dot(u_sol, grad(c))*v_c) * dx
L_c = Constant(0.0) * v_c * dx

c_sol = Function(V_c)
solve(a_c == L_c, c_sol, bcs_c,
      solver_parameters={"linear_solver": "mumps"})

# -------------------------------------------------
# Output: save fields
# -------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q10_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.write(c_sol, 0.0)
xdmf.close()

# -------------------------------------------------
# Plot concentration and save as PNG
# -------------------------------------------------
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 2))
p = plot(c_sol, title="Concentration", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q10_conc.png", dpi=300)
plt.close()