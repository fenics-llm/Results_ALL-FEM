# filename: backward_facing_step_navier_stokes.py
from dolfin import *
import mshr
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Geometry and mesh
# -------------------------------------------------
H = 1.0               # step height
L_up = 3.0 * H        # upstream length
L_down = 20.0 * H     # downstream length

# Upstream rectangle (height H)
up_rect = mshr.Rectangle(Point(-L_up, 0.0), Point(0.0, H))
# Downstream rectangle (height 2H)
down_rect = mshr.Rectangle(Point(0.0, 0.0), Point(L_down, 2.0 * H))

# Union of both rectangles gives the step domain
domain = up_rect + down_rect

# Mesh resolution (adjust as needed)
mesh_res = 0.2 * H
mesh = mshr.generate_mesh(domain, int(L_down / mesh_res))

# -------------------------------------------------
# Function spaces (Taylor–Hood)
# -------------------------------------------------
V_el = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P_el = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([V_el, P_el])
W = FunctionSpace(mesh, TH)

# -------------------------------------------------
# Boundary definitions
# -------------------------------------------------
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], -L_up) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        # Bottom wall
        if near(x[1], 0.0) and on_boundary:
            return True
        # Upper wall upstream (y = H, x < 0)
        if near(x[1], H) and x[0] <= 0.0 + DOLFIN_EPS and on_boundary:
            return True
        # Upper wall downstream (y = 2H, x > 0)
        if near(x[1], 2.0 * H) and x[0] >= 0.0 - DOLFIN_EPS and on_boundary:
            return True
        # Step wall (x = 0, H <= y <= 2H)
        if near(x[0], 0.0) and H - DOLFIN_EPS <= x[1] <= 2.0 * H + DOLFIN_EPS and on_boundary:
            return True
        return False

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L_down) and on_boundary

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
inlet = Inlet()
walls = Walls()
outlet = Outlet()
inlet.mark(boundaries, 1)
walls.mark(boundaries, 2)
outlet.mark(boundaries, 3)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Boundary conditions
# -------------------------------------------------
U_bar = 1.0
mu = 0.01
rho = 1.0

# Parabolic inlet profile
inlet_velocity = Expression(("6.0*U_bar*x[1]/H*(1.0 - x[1]/H)", "0.0"),
                            degree=2, U_bar=U_bar, H=H)

noslip = Constant((0.0, 0.0))

bcu_inlet = DirichletBC(W.sub(0), inlet_velocity, boundaries, 1)
bcu_walls = DirichletBC(W.sub(0), noslip, boundaries, 2)

# Pressure gauge (pointwise) to fix nullspace
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], -L_up) and near(x[1], H / 2.0)

p_point = PressurePoint()
bcp = DirichletBC(W.sub(1), Constant(0.0), p_point, method="pointwise")

bcs = [bcu_inlet, bcu_walls, bcp]

# -------------------------------------------------
# Variational problem (Picard iteration)
# -------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

w = Function(W)                     # current solution (u,p)
(u_k, p_k) = w.split(deepcopy=True)   # previous velocity for convection

# Linearized (Picard) forms
a = (2*mu * inner(sym(grad(u)), sym(grad(v))) * dx
     + rho * inner(dot(u_k, nabla_grad(u)), v) * dx
     - div(v) * p * dx
     + q * div(u) * dx)

L = Constant(0.0) * q * dx   # RHS = 0

# -------------------------------------------------
# Picard iteration
# -------------------------------------------------
tol = 1e-6
max_iter = 30
for iter in range(max_iter):
    solve(a == L, w, bcs,
          solver_parameters={"linear_solver": "mumps"})
    (u_k, p_k) = w.split(deepcopy=True)

    # Assemble momentum residual vector
    R_form = (2*mu * inner(sym(grad(u_k)), sym(grad(v))) * dx
              + rho * inner(dot(u_k, nabla_grad(u_k)), v) * dx
              - div(v) * p_k * dx
              + q * div(u_k) * dx)
    r_vec = assemble(R_form)
    res_norm = r_vec.norm('l2')
    if res_norm < tol:
        print(f"Picard converged in {iter+1} iterations, residual = {res_norm:e}")
        break
else:
    print("Picard did not converge within the maximum number of iterations")

# -------------------------------------------------
# Post-processing: wall shear stress on the top wall (y = 2H)
# -------------------------------------------------
V_scalar = FunctionSpace(mesh, "P", 1)
tau_expr = mu * grad(u_k)[0, 1]          # ∂ux/∂y component
tau = project(tau_expr, V_scalar)

# Sample τ_w along downstream top wall
N_pts = 400
x_vals = np.linspace(0.0, L_down, N_pts)
tau_vals = np.array([tau(Point(x, 2.0 * H)) for x in x_vals])

# Find re-attachment point (zero crossing)
reattach_x = None
for i in range(N_pts - 1):
    if tau_vals[i] * tau_vals[i + 1] < 0.0:
        x0, x1 = x_vals[i], x_vals[i + 1]
        t0, t1 = tau_vals[i], tau_vals[i + 1]
        reattach_x = x0 - t0 * (x1 - x0) / (t1 - t0)
        break

if reattach_x is not None:
    print(f"Re-attachment point (τ_w = 0) at x = {reattach_x:.4f} m")
else:
    print("No zero crossing of τ_w detected on the sampled wall segment")

# -------------------------------------------------
# Save solution (XDMF) and velocity magnitude plot (PNG)
# -------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "q6_soln.xdmf") as xdmf:
    xdmf.write(mesh)
    xdmf.write(u_k)
    xdmf.write(p_k)

# Velocity magnitude plot
u_mag = sqrt(dot(u_k, u_k))
V_plot = FunctionSpace(mesh, "P", 2)
u_mag_plot = project(u_mag, V_plot)

plt.figure(figsize=(8, 3))
c = plot(u_mag_plot, title="Velocity magnitude", cmap="viridis")
plt.colorbar(c)
plt.savefig("q6_u.png", dpi=300)
plt.close()