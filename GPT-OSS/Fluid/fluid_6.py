# filename: q6_navier_stokes.py
"""
Steady incompressible Navier–Stokes flow in a 2‑D channel with a backward‑facing step.
The script

1. builds the geometry and mesh,
2. solves the steady Navier–Stokes equations using a Picard (fixed‑point) iteration
   (more robust than a pure Newton solve for Re≈100),
3. computes the wall shear stress τ_w = μ ∂u_x/∂y on the upper wall (y = H, x∈[0,20H]),
4. finds the re‑attachment point where τ_w changes sign,
5. saves a velocity glyph plot (q6_u.png) and the (u,p) fields in XDMF
   (q6_soln.xdmf), and
6. writes the sampled τ_w curve to a text file.

All output files are written to the current working directory.
"""

from dolfin import *
import mshr
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------
H    = 1.0               # characteristic height (m)
Ubar = 1.0               # mean inlet velocity (m/s)
rho  = 1.0               # density (kg/m³)
mu   = 0.01              # dynamic viscosity (Pa·s)

# --------------------------------------------------------------------
# Geometry and mesh
# --------------------------------------------------------------------
up_rect   = mshr.Rectangle(Point(-3*H, 0.0), Point(0.0, H))      # upstream part
down_rect = mshr.Rectangle(Point(0.0, 0.0), Point(20*H, 2*H))    # downstream part
domain = up_rect + down_rect
mesh = mshr.generate_mesh(domain, 80)   # increase the second argument for a finer mesh

# --------------------------------------------------------------------
# Function spaces (Taylor–Hood P2/P1)
# --------------------------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)   # velocity (P2)
Q = FunctionSpace(mesh, "Lagrange", 1)       # pressure (P1)

mixed_elem = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# --------------------------------------------------------------------
# Boundary definitions
# --------------------------------------------------------------------
tol = 1e-14

class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], -3*H, tol)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 20*H, tol)

class BottomWall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)

class TopWallLeft(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H, tol) and x[0] <= 0.0 + tol

class TopWallRight(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 2*H, tol) and x[0] >= 0.0 - tol

class StepWall(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol) and x[1] >= H - tol

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
Inlet().mark(boundaries, 1)
Outlet().mark(boundaries, 2)
BottomWall().mark(boundaries, 3)
TopWallLeft().mark(boundaries, 4)
TopWallRight().mark(boundaries, 5)
StepWall().mark(boundaries, 6)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# --------------------------------------------------------------------
# Boundary conditions
# --------------------------------------------------------------------
# Parabolic inlet profile (only on the lower part of the inlet)
inlet_profile = Expression(("6.0*Ubar*(x[1]/H)*(1.0 - x[1]/H)", "0.0"),
                           degree=2, Ubar=Ubar, H=H)

bcu_inlet = DirichletBC(W.sub(0), inlet_profile, boundaries, 1)

zero_vec = Constant((0.0, 0.0))
bcu_walls = [
    DirichletBC(W.sub(0), zero_vec, boundaries, 3),  # bottom
    DirichletBC(W.sub(0), zero_vec, boundaries, 4),  # top left
    DirichletBC(W.sub(0), zero_vec, boundaries, 5),  # top right
    DirichletBC(W.sub(0), zero_vec, boundaries, 6)   # step wall
]

# Pressure reference (set p = 0 on the outlet to fix the null‑space)
bc_pressure = DirichletBC(W.sub(1), Constant(0.0), boundaries, 2)

bcs = [bcu_inlet] + bcu_walls + [bc_pressure]

# --------------------------------------------------------------------
# Picard (fixed‑point) iteration for the steady Navier–Stokes equations
# --------------------------------------------------------------------
# Mixed unknown (velocity + pressure)
up = Function(W)                     # will contain the current iterate
(u, p) = split(up)                   # for use in forms
(v, q) = TestFunctions(W)

# Linearised (Picard) form: convection uses previous velocity u_k
u_k = Function(V)                    # previous velocity
# Initialise u_k with the inlet profile extended to the whole domain
u_k.interpolate(inlet_profile)

# Define the bilinear and linear forms
a = (mu*inner(grad(u), grad(v))*dx
     + rho*inner(dot(u_k, nabla_grad(u)), v)*dx
     - div(v)*p*dx
     - q*div(u)*dx)

L = Constant(0.0)*v[0]*dx  # RHS = 0

# Solver for the linearised system
problem = LinearVariationalProblem(a, L, up, bcs)
solver  = LinearVariationalSolver(problem)

# Picard parameters
max_iter = 30
tol_picard = 1e-6
converged = False

for it in range(max_iter):
    # Solve the linearised system
    solver.solve()
    # Extract the new velocity
    u_new, _ = up.split()
    # Compute norm of difference
    diff = errornorm(u_new, u_k, norm_type='l2', mesh=mesh)
    print(f"Picard iteration {it+1}: ||u_new - u_k||_L2 = {diff:.3e}")
    if diff < tol_picard:
        converged = True
        break
    # Update u_k for next iteration
    u_k.assign(u_new)

if not converged:
    print("Warning: Picard iteration did not converge within the prescribed tolerance.")

# --------------------------------------------------------------------
# Extract final velocity and pressure
# --------------------------------------------------------------------
(u_sol, p_sol) = up.split()

# --------------------------------------------------------------------
# Wall shear stress τ_w = μ ∂u_x/∂y on the upper wall (y = H, x∈[0,20H])
# --------------------------------------------------------------------
# Project ∂u_x/∂y onto a DG0 space (piecewise constant) for cheap evaluation
Vdg0 = FunctionSpace(mesh, "DG", 0)
ux = u_sol.sub(0, deepcopy=True)          # x‑component of velocity
duxdY = project(ux.dx(1), Vdg0)            # ∂u_x/∂y

# Sample τ_w along the wall
num_pts = 400
x_vals = np.linspace(0.0, 20*H, num_pts)
tau_vals = np.empty(num_pts)

for i, x in enumerate(x_vals):
    pt = Point(x, H)                       # point on the upper wall
    cell_id = mesh.bounding_box_tree().compute_first_entity_collision(pt)
    if cell_id < mesh.num_cells():
        tau_vals[i] = mu * duxdY(pt)
    else:
        tau_vals[i] = np.nan

# Locate re‑attachment point (zero crossing from negative to positive)
sign_change = np.where(np.diff(np.sign(tau_vals)) > 0)[0]
if len(sign_change) > 0:
    idx = sign_change[0]
    x1, x2 = x_vals[idx], x_vals[idx + 1]
    t1, t2 = tau_vals[idx], tau_vals[idx + 1]
    x_reattach = x1 - t1 * (x2 - x1) / (t2 - t1)   # linear interpolation
else:
    x_reattach = None

print("Approximate re‑attachment point (m):", x_reattach)

# --------------------------------------------------------------------
# Save velocity field plot (glyphs) as PNG
# --------------------------------------------------------------------
plt.figure(figsize=(9, 3))
plot(u_sol, title="Velocity field", mode='glyphs', scale=0.5)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q6_u.png", dpi=300)
plt.close()

# --------------------------------------------------------------------
# Save solution (u,p) in XDMF format
# --------------------------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q6_soln.xdmf")
xdmf.parameters["functions_share_mesh"] = True
xdmf.parameters["flush_output"] = True
xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.close()

# --------------------------------------------------------------------
# Optional: write τ_w distribution for post‑processing
# --------------------------------------------------------------------
np.savetxt("tau_w_top_wall.txt",
           np.column_stack([x_vals, tau_vals]),
           header="x (m)    tau_w (Pa)")
