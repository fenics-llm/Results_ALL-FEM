# filename: q13_solver.py
"""
Steady incompressible Navier–Stokes with temperature‑dependent viscosity
and steady advection–diffusion for temperature in a 2‑D rectangular channel.

Outputs
-------
q13_mu.png        – colour map of the viscosity μ(x,y)
q13_profile.csv   – streamwise velocity u_x(y) at x = 1.0  (columns: y,ux)
q13_solution.xdmf – XDMF file containing u, p, T and μ
"""

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
L = 2.0          # length (m)
H = 0.20         # height (m)
rho = 1.0        # density (kg/m³)
Ubar = 1.0       # mean inlet speed (m/s)
mu_ref = 0.02    # reference viscosity (Pa·s)
beta = 0.05      # 1/K
T_ref = 300.0    # reference temperature (K)
kappa = 1.0e-3   # thermal diffusivity (m²/s)

# ----------------------------------------------------------------------
# Mesh
# ----------------------------------------------------------------------
nx, ny = 80, 20                     # mesh resolution (increase if needed)
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny)

# ----------------------------------------------------------------------
# Function spaces (Taylor‑Hood)
# ----------------------------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)   # velocity
Q = FunctionSpace(mesh, "Lagrange", 1)        # pressure
Tspace = FunctionSpace(mesh, "Lagrange", 1)   # temperature

# mixed (u,p) space
W = FunctionSpace(mesh, MixedElement([V.ufl_element(),
                                      Q.ufl_element()]))

# ----------------------------------------------------------------------
# Boundary definitions
# ----------------------------------------------------------------------
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], H) and on_boundary

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary

class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        # a single point on the outlet (mid‑height)
        return near(x[0], L) and near(x[1], H/2.0)

inlet = Inlet()
walls = Walls()
bottom = Bottom()
top = Top()
outlet = Outlet()
press_pt = PressurePoint()

# ----------------------------------------------------------------------
# Expressions for inlet data
# ----------------------------------------------------------------------
u_inlet_expr = Expression(
    ("6.0*Ubar*x[1]*(H - x[1])/pow(H,2)", "0.0"),
    degree=2, Ubar=Ubar, H=H
)

T_inlet_expr = Constant(T_ref)
T_bottom_expr = Constant(T_ref + 10.0)

# ----------------------------------------------------------------------
# Boundary conditions
# ----------------------------------------------------------------------
# Velocity / pressure (mixed space)
bcu_inlet = DirichletBC(W.sub(0), u_inlet_expr, inlet)
bcu_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)
bcp_point = DirichletBC(W.sub(1), Constant(0.0), press_pt, "pointwise")
bcs_NS = [bcu_inlet, bcu_walls, bcp_point]

# Temperature
bct_inlet = DirichletBC(Tspace, T_inlet_expr, inlet)
bct_bottom = DirichletBC(Tspace, T_bottom_expr, bottom)   # top wall: natural Neumann
bcs_T = [bct_inlet, bct_bottom]

# ----------------------------------------------------------------------
# Functions to hold the fields
# ----------------------------------------------------------------------
w = Function(W)          # (u,p) combined
(u, p) = split(w)        # symbolic split for variational forms
u_sol = Function(V)      # velocity after each NS solve
p_sol = Function(Q)      # pressure after each NS solve
T = Function(Tspace)     # temperature
mu = Function(Tspace)    # viscosity field

# Initialise temperature with inlet value
T.interpolate(Constant(T_ref))

# ----------------------------------------------------------------------
# Helper: compute viscosity from temperature
# ----------------------------------------------------------------------
def update_viscosity():
    """μ = μ_ref * exp[-β (T - T_ref)]"""
    mu_expr = mu_ref*exp(-beta*(T - T_ref))
    mu_proj = project(mu_expr, Tspace)
    mu.assign(mu_proj)

update_viscosity()

# ----------------------------------------------------------------------
# Variational forms (symbols)
# ----------------------------------------------------------------------
# Trial / test for Navier–Stokes (mixed)
(u_trial, p_trial) = TrialFunctions(W)
(v_test, q_test) = TestFunctions(W)

# Convection field from previous iteration (Picard)
u_k = Function(V)
u_k.assign(Constant((0.0, 0.0)))   # initial guess

def ns_form(u_tr, p_tr, v_te, q_te, u_conv):
    """Steady NS residual (Picard linearisation)."""
    eps = sym(grad(u_tr))
    return ( rho*dot(dot(u_conv, nabla_grad(u_tr)), v_te)
           + 2.0*mu*inner(eps, sym(grad(v_te)))
           - div(v_te)*p_tr
           + q_te*div(u_tr) )*dx

# Temperature variational form (will be rebuilt each iteration because it uses u_sol)
def temp_form(T_tr, v_te, u_vel):
    return ( dot(u_vel, grad(T_tr))*v_te + kappa*dot(grad(T_tr), grad(v_te)) )*dx

# ----------------------------------------------------------------------
# Solver settings
# ----------------------------------------------------------------------
max_iter = 30
tol = 1e-6

# ----------------------------------------------------------------------
# Picard iteration
# ----------------------------------------------------------------------
for it in range(max_iter):
    # ---- Navier–Stokes (linearised) ----
    a_NS = lhs(ns_form(u_trial, p_trial, v_test, q_test, u_k))
    L_NS = rhs(ns_form(u_trial, p_trial, v_test, q_test, u_k))   # zero RHS
    solve(a_NS == L_NS, w, bcs_NS,
          solver_parameters={"linear_solver": "mumps"})

    # extract components
    assign(u_sol, w.sub(0))
    assign(p_sol, w.sub(1))

    # update convection field for next NS solve
    u_k.assign(u_sol)

    # ---- Temperature (linear) ----
    T_trial = TrialFunction(Tspace)
    v_T = TestFunction(Tspace)
    a_T = lhs(temp_form(T_trial, v_T, u_sol))
    L_T = rhs(temp_form(T_trial, v_T, u_sol))   # zero RHS
    solve(a_T == L_T, T, bcs_T,
          solver_parameters={"linear_solver": "mumps"})

    # ---- Viscosity update ----
    mu_old = mu.vector().get_local().copy()
    update_viscosity()
    diff_mu = np.linalg.norm(mu.vector().get_local() - mu_old, ord=np.Inf)
    print(f"Iteration {it+1}: max|Δμ| = {diff_mu:.3e}")

    if diff_mu < tol:
        print("Converged.")
        break
else:
    print("Warning: Picard iteration did not converge within the maximum number of iterations.")

# ----------------------------------------------------------------------
# Post‑processing
# ----------------------------------------------------------------------
# 1) Viscosity colour map
mu_vals = mu.compute_vertex_values(mesh)
coords = mesh.coordinates()
triang = tri.Triangulation(coords[:, 0], coords[:, 1], mesh.cells())

plt.figure(figsize=(6, 3))
plt.tripcolor(triang, mu_vals, shading='gouraud')
plt.colorbar(label=r'Viscosity $\mu$ (Pa·s)')
plt.title('Viscosity field')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q13_mu.png", dpi=300)
plt.close()

# 2) Streamwise velocity profile at x = 1.0
x_mid = 1.0
y_vals = np.linspace(0.0, H, 200)
ux_vals = np.empty_like(y_vals)

for i, yy in enumerate(y_vals):
    point = Point(x_mid, yy)
    ux_vals[i] = u_sol(point)[0]

profile = np.column_stack((y_vals, ux_vals))
np.savetxt("q13_profile.csv", profile, delimiter=",", header="y,ux", comments='')

# 3) Export all fields to XDMF
with XDMFFile(mesh.mpi_comm(), "q13_solution.xdmf") as xdmf:
    xdmf.parameters["functions_share_mesh"] = True
    xdmf.parameters["flush_output"] = True
    xdmf.write(u_sol, 0.0)
    xdmf.write(p_sol, 0.0)
    xdmf.write(T, 0.0)
    xdmf.write(mu, 0.0)

print("All outputs written: q13_mu.png, q13_profile.csv, q13_solution.xdmf")