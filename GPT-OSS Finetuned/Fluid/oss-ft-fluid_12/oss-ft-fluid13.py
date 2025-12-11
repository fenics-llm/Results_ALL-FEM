# filename: q13_navier_stokes_temperature.py
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Geometry and mesh
L, H = 2.0, 0.20
nx, ny = 80, 20
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny, "crossed")

# --------------------------------------------------------------
# Physical parameters
rho    = 1.0          # kg/m^3
Ubar   = 1.0          # m/s
mu_ref = 0.02         # Pa·s
beta   = 0.05         # 1/K
T_ref  = 300.0        # K
kappa  = 1.0e-3      # m^2/s

# --------------------------------------------------------------
# Function spaces (Taylor–Hood for (u,p) + CG1 for T)
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # velocity
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure
Te = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # temperature
W_elem = MixedElement([Ve, Pe, Te])
W = FunctionSpace(mesh, W_elem)

# --------------------------------------------------------------
# Boundary definitions
tol = 1E-14
inlet  = CompiledSubDomain("near(x[0], side, tol)", side=0.0, tol=tol)
walls  = CompiledSubDomain("near(x[1], 0.0, tol) || near(x[1], H, tol)",
                            H=H, tol=tol)
outlet = CompiledSubDomain("near(x[0], L, tol)", L=L, tol=tol)
bottom = CompiledSubDomain("near(x[1], 0.0, tol)", tol=tol)
top    = CompiledSubDomain("near(x[1], H, tol)", H=H, tol=tol)

# --------------------------------------------------------------
# Inlet velocity profile (parabolic)
u_inlet_expr = Expression(("6.0*Ubar*x[1]*(H - x[1])/(H*H)", "0.0"),
                          Ubar=Ubar, H=H, degree=2)

# --------------------------------------------------------------
# Temperature Dirichlet at inlet and bottom wall
T_inlet_expr  = Constant(T_ref)
T_bottom_expr = Constant(T_ref + 10.0)

# --------------------------------------------------------------
# Define trial and test functions
(u, p, T) = TrialFunctions(W)
(v, q, S) = TestFunctions(W)

# --------------------------------------------------------------
# Function for current solution (Newton)
w = Function(W)
(u_, p_, T_) = split(w)   # for evaluation in forms

# --------------------------------------------------------------
# Temperature-dependent viscosity
mu = mu_ref*exp(-beta*(T_ - T_ref))

def epsilon(v):
    return sym(grad(v))

# --------------------------------------------------------------
# Weak forms (steady Navier–Stokes + advection–diffusion)
F_NS = rho*dot(dot(u_, nabla_grad(u_)), v)*dx \
       + 2*mu*inner(epsilon(u_), epsilon(v))*dx \
       - div(v)*p_*dx + q*div(u_)*dx

F_T  = dot(u_, nabla_grad(T_))*S*dx + kappa*dot(grad(T_), grad(S))*dx

F = F_NS + F_T

# --------------------------------------------------------------
# Jacobian
J = derivative(F, w, TrialFunction(W))

# --------------------------------------------------------------
# Boundary conditions
bcs = []

# Inlet velocity
bcs.append(DirichletBC(W.sub(0), u_inlet_expr, inlet))

# No-slip walls (u=0)
bcs.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls))

# Inlet temperature
bcs.append(DirichletBC(W.sub(2), T_inlet_expr, inlet))

# Bottom wall temperature (pointwise)
bcs.append(DirichletBC(W.sub(2), T_bottom_expr, bottom))

# Pressure reference point to fix nullspace (pin p at (0,0))
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, tol) and near(x[1], 0.0, tol)
p_point = PressurePoint()
bcs.append(DirichletBC(W.sub(1), Constant(0.0), p_point, method="pointwise"))

# --------------------------------------------------------------
# Solve nonlinear system (Newton)
solve(F == 0, w, bcs,
      solver_parameters={"newton_solver":
                         {"relative_tolerance":1e-6,
                          "absolute_tolerance":1e-8,
                          "maximum_iterations":25,
                          "linear_solver":"mumps"}})

# --------------------------------------------------------------
# Split solution
(u_sol, p_sol, T_sol) = w.split(deepcopy=True)

# Compute viscosity field for post-processing
mu_sol = project(mu_ref*exp(-beta*(T_sol - T_ref)), FunctionSpace(mesh, "CG", 1))

# --------------------------------------------------------------
# Save viscosity as PNG
plt.figure(figsize=(6,2))
p = plot(mu_sol, title=r"$\mu(x,y)$ (Pa·s)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q13_mu.png", dpi=300)
plt.close()

# --------------------------------------------------------------
# Extract u_x along mid-channel line x = L/2
y_vals = np.linspace(0.0, H, ny*5+1)
ux_vals = np.array([u_sol(Point(L/2, y))[0] for y in y_vals])

# Save to CSV
np.savetxt("q13_profile.csv", np.column_stack([y_vals, ux_vals]),
           header="y, ux", delimiter=",", comments="")

# --------------------------------------------------------------
# Export solution fields to XDMF
xdmf = XDMFFile(mesh.mpi_comm(), "q13_solution.xdmf")
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["flush_output"] = True
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.write(T_sol, 0.0)
xdmf.write(mu_sol, 0.0)
xdmf.close()