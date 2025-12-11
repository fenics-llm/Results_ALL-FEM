# q6_backward_facing_step_legacy.py  (robust solve: Stokes warm-start + continuation)
from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt

# -------- Parameters --------
H     = 1.0
rho   = 1.0
mu    = 0.01
U_bar = 1.0

x_min, x_split, x_max = -3.0*H, 0.0*H, 20.0*H
y_low, y_mid,  y_top  = 0.0,    1.0*H, 2.0*H

# -------- Geometry & mesh --------
rect_up   = Rectangle(Point(x_min, y_low),  Point(x_split, y_mid))
rect_down = Rectangle(Point(x_split, y_low), Point(x_max,  y_top))
domain    = rect_up + rect_down
mesh = generate_mesh(domain, 120)

# -------- Mixed space (P2-P1) --------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Qe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W  = FunctionSpace(mesh, MixedElement([Ve, Qe]))

# -------- Boundaries --------
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], x_min) and (x[1] >= y_low - DOLFIN_EPS) and (x[1] <= y_mid + DOLFIN_EPS)

class BottomWall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], y_low) and (x[0] >= x_min - DOLFIN_EPS) and (x[0] <= x_max + DOLFIN_EPS)

class TopWallUp(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], y_mid) and (x[0] <= x_split + DOLFIN_EPS)

class TopWallDown(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], y_top) and (x[0] >= x_split - DOLFIN_EPS)

class StepWall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], x_split) and (x[1] >= y_mid - DOLFIN_EPS) and (x[1] <= y_top + DOLFIN_EPS)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], x_max) and (x[1] >= y_low - DOLFIN_EPS) and (x[1] <= y_top + DOLFIN_EPS)

inlet   = Inlet(); bot = BottomWall(); top_up = TopWallUp(); top_dn = TopWallDown(); step_w = StepWall(); outlet = Outlet()

# -------- Inflow profile --------
inlet_profile = Expression(("amp*6.0*Ubar*(x[1]/H)*(1.0 - x[1]/H)", "0.0"),
                           Ubar=U_bar, H=H, amp=1.0, degree=2)

# -------- Dirichlet BCs --------
bc_inlet = DirichletBC(W.sub(0), inlet_profile, inlet)
bc_bot   = DirichletBC(W.sub(0), Constant((0.0, 0.0)), bot)
bc_top_u = DirichletBC(W.sub(0), Constant((0.0, 0.0)), top_up)
bc_top_d = DirichletBC(W.sub(0), Constant((0.0, 0.0)), top_dn)
bc_step  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), step_w)
# Pressure pin to remove nullspace — use a mesh corner that surely exists
p_ref = DirichletBC(W.sub(1), Constant(0.0),
                    "near(x[0], %.16g) && near(x[1], %.16g)" % (x_max, y_low),
                    method="pointwise")
bcs = [bc_inlet, bc_bot, bc_top_u, bc_top_d, bc_step, p_ref]

# -------- Helpers --------
def epsilon(w): return sym(grad(w))
f = Constant((0.0, 0.0))

# -------- 1) Stokes solve (warm start) --------
U  = Function(W)                  # (u, p)
u, p = split(U)
v, q = TestFunctions(W)

F_stokes = 2.0*mu*inner(epsilon(u), epsilon(v))*dx - div(v)*p*dx - q*div(u)*dx - inner(f, v)*dx

solve(lhs(F_stokes) == rhs(F_stokes), U, bcs,
      solver_parameters={
          "linear_solver": "mumps",
          "preconditioner": "default"
      })

# -------- 2) Continuation on inflow amplitude with Newton at each step --------
# This avoids Newton divergence from a too-nonlinear initial state.
continuation_amps = [0.25, 0.5, 0.75, 1.0]

# Start from the Stokes solution as initial guess
U_prev = Function(W); U_prev.assign(U)

for amp in continuation_amps:
    inlet_profile.amp = amp

    # Nonlinear residual with full convection, initialised from previous step
    u, p = split(U)
    v, q = TestFunctions(W)
    F = rho*inner(grad(u)*u, v)*dx + 2.0*mu*inner(epsilon(u), epsilon(v))*dx \
        - div(v)*p*dx - q*div(u)*dx - inner(f, v)*dx

    problem = NonlinearVariationalProblem(F, U, bcs)
    solver  = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["nonlinear_solver"] = "newton"
    prm["newton_solver"]["absolute_tolerance"] = 1e-10
    prm["newton_solver"]["relative_tolerance"] = 1e-8
    prm["newton_solver"]["maximum_iterations"] = 40
    prm["newton_solver"]["linear_solver"] = "mumps"
    prm["newton_solver"]["report"] = True
    prm["newton_solver"]["error_on_nonconvergence"] = True

    # Use the previous solution as initial guess
    U.assign(U_prev)
    solver.solve()
    U_prev.assign(U)  # carry forward

u_h, p_h = U.split(deepcopy=True)
u_h.rename("u", "velocity"); p_h.rename("p", "pressure")

# -------- Save XDMF --------
xdmf = XDMFFile(MPI.comm_world, "q6_soln.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_h, 0.0); xdmf.write(p_h, 0.0); xdmf.close()

# -------- Plot velocity magnitude --------
umag = project(sqrt(dot(u_h, u_h)), FunctionSpace(mesh, "CG", 1))
plt.figure(); c = plot(umag, title="Velocity magnitude |u|"); plt.colorbar(c)
plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout(); plt.savefig("q6_u.png", dpi=200); plt.close()

# -------- Wall shear stress on downstream top wall (y = 2H) --------
TFS = TensorFunctionSpace(mesh, "CG", 1)
G   = project(grad(u_h), TFS)
SFS = FunctionSpace(mesh, "CG", 1)
tau_w_field = project(mu*G[0, 1], SFS)

eps = 1e-6
N  = 801
xs = np.linspace(0.0, x_max, N)
ys = (y_top - eps) * np.ones_like(xs)
tau = []
for xi, yi in zip(xs, ys):
    try:
        tau.append(tau_w_field(Point(float(xi), float(yi))))
    except RuntimeError:
        tau.append(tau_w_field(Point(float(min(max(xi, 1e-8), x_max-1e-8)), float(y_top - 1e-5))))
tau = np.array(tau)
np.savetxt("q6_topwall_tau.csv", np.column_stack([xs, tau]), delimiter=",", header="x, tau_w_top(y=2H)")

# zero-crossing for re-attachment
x_r = None
for i in range(1, len(xs)):
    if tau[i-1]*tau[i] < 0.0:
        x0, x1 = xs[i-1], xs[i]; y0, y1 = tau[i-1], tau[i]
        x_r = x0 - y0*(x1 - x0)/(y1 - y0)
        break

plt.figure()
plt.plot(xs, tau, lw=1.5); plt.axhline(0.0, ls="--")
if x_r is not None:
    plt.axvline(x_r, ls="--"); plt.text(x_r, 0.0, "  x_r ≈ %.4f" % x_r, va="bottom")
plt.xlabel("x along top wall (y=2H)")
plt.ylabel("tau_w = mu * d(u_x)/dy")
plt.title("Top-wall shear stress (downstream)")
plt.tight_layout(); plt.savefig("q6_topwall_tau.png", dpi=200); plt.close()

if x_r is not None:
    print("Estimated re-attachment at x_r ≈ %.6f m" % x_r)
else:
    print("No zero-crossing of tau_w on the downstream top wall in [0, 20H].")

print("Saved: q6_u.png, q6_soln.xdmf, q6_topwall_tau.csv, q6_topwall_tau.png")