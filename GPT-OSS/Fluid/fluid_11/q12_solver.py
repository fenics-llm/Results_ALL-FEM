# filename: q12_solver.py
from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Parameters --------------------
L = 2.0          # channel length (m)
H = 0.20         # channel height (m)
rho = 1.0        # density (kg/m³)
mu0 = 0.5        # consistency index (Pa·sⁿ)
n = 0.5          # flow behaviour index
U_bar = 1.0      # mean inlet velocity (m/s)

# -------------------- Mesh --------------------
nx, ny = 240, 24
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny)

# -------------------- Function spaces --------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)   # velocity (P2)
Q = FunctionSpace(mesh, "Lagrange", 1)        # pressure (P1)

# Mixed space (velocity, pressure) via MixedElement (compatible with all FEniCS versions)
mixed_elem = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# -------------------- Boundary definitions --------------------
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return (near(x[1], 0.0) or near(x[1], H)) and on_boundary

inlet = Inlet()
walls = Walls()

# Inlet velocity (parabolic profile)
class InletVelocity(UserExpression):
    def eval(self, values, x):
        y = x[1]
        values[0] = 6.0 * U_bar * y * (H - y) / H**2
        values[1] = 0.0
    def value_shape(self):
        return (2,)

inlet_vel = InletVelocity(degree=2)

# Dirichlet BCs (velocity only)
bc_inlet = DirichletBC(W.sub(0), inlet_vel, inlet)
bc_walls = DirichletBC(W.sub(0), Constant((0.0, 0.0)), walls)
bcs = [bc_inlet, bc_walls]

# -------------------- Helper: strain‑rate tensor --------------------
def D(u):
    return sym(grad(u))

# -------------------- Picard iteration (fixed‑point) --------------------
# Initial guess: zero velocity, zero pressure
w = Function(W)               # (u, p)
(u_k, p_k) = w.split()        # current iterates (both initially zero)

# Function to hold the effective viscosity (scalar CG1)
Vmu = FunctionSpace(mesh, "CG", 1)
mu_eff = Function(Vmu)

# Tolerance and maximum number of Picard iterations
tol = 1e-6
max_iter = 30

for it in range(max_iter):
    # ---- Update effective viscosity based on current velocity ----
    gamma = sqrt(2.0 * inner(D(u_k), D(u_k))) + 1e-8
    mu_expr = mu0 * gamma**(n - 1.0)
    mu_eff.assign(project(mu_expr, Vmu))

    # ---- Define variational problem with the *known* viscosity mu_eff ----
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    # Linearised convection term using the previous velocity u_k
    a = ( rho * dot(dot(u_k, nabla_grad(u)), v) * dx
          + inner(2.0 * mu_eff * D(u), D(v)) * dx
          - p * div(v) * dx
          + q * div(u) * dx )

    L_form = Constant(0.0) * v[0] * dx   # RHS = 0 (no body forces)

    # Solve the linear system
    w_new = Function(W)
    solve(a == L_form, w_new, bcs, 
          solver_parameters={'linear_solver': 'mumps'})

    # Split the new solution
    (u_new, p_new) = w_new.split()

    # ---- Convergence check (velocity L2 norm) ----
    diff = u_new.vector() - u_k.vector()
    err = diff.norm('l2')
    print(f"Picard iteration {it+1}: ||u_new - u_k||_L2 = {err:.3e}")

    if err < tol:
        w.assign(w_new)   # accept final solution
        print("Converged.")
        break

    # Update for next iteration
    w.assign(w_new)
    u_k, p_k = w.split()
else:
    print("Warning: Picard iteration did not converge within the maximum number of iterations.")

# -------------------- Extract converged fields --------------------
(u_sol, p_sol) = w.split()

# -------------------- Post‑processing --------------------
# Velocity magnitude
u_mag = sqrt(dot(u_sol, u_sol))

# Plot and save velocity magnitude
plt.figure(figsize=(8, 3))
p = plot(u_mag, title='Velocity magnitude |u| (m/s)', cmap='viridis')
plt.colorbar(p)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig('q12_speed.png', dpi=300)
plt.close()

# Streamwise velocity profile at x = L/2
x_mid = L / 2.0
y_vals = np.linspace(0.0, H, 200)
ux_vals = np.array([u_sol(Point(x_mid, y))[0] for y in y_vals])

# Save profile to CSV
profile = np.column_stack((y_vals, ux_vals))
np.savetxt('q12_profile.csv', profile, delimiter=',',
           header='y (m), ux (m/s)', comments='')

# Report maximum ux at the mid‑channel
max_ux = ux_vals.max()
print(f"Maximum streamwise velocity ux at x = {x_mid:.2f} m: {max_ux:.6f} m/s")

# Effective viscosity field (already computed on the last iteration)
# Project to CG1 for output (already mu_eff)
mu_eff_proj = mu_eff

# Save fields to XDMF
xdmf = XDMFFile(mesh.mpi_comm(), "q12_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True

xdmf.write(u_sol, 0.0)
xdmf.write(p_sol, 0.0)
xdmf.write(mu_eff_proj, 0.0)
xdmf.close()