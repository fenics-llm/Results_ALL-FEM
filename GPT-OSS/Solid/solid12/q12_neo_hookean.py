# filename: q12_neo_hookean.py
"""
Large‑deformation incompressible neo‑Hookean strip with a circular hole.
- Plane‑strain mixed (u,p) formulation.
- Taylor‑Hood (P2 for displacement, P1 for pressure).
- Left edge fixed, right edge displaced by 0.060 m in x.
- Traction‑free elsewhere.
- Saves:
    * hydrostatic pressure → q12_p.png
    * von Mises stress   → q12_vm.png
    * displacement field → displacement.xdmf
"""

from fenics import *
import mshr
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Geometry & mesh
# -------------------------------------------------------------------
Lx, Ly = 1.0, 0.20                # dimensions (m)
a = 0.04                         # hole radius (m)
center = Point(0.5, 0.10)

rect = mshr.Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
hole = mshr.Circle(center, a, 64)   # 64 points for a smooth circle
domain = rect - hole

mesh = mshr.generate_mesh(domain, 64)   # mesh resolution (adjust if needed)

# -------------------------------------------------------------------
# Material parameters (plane strain, incompressible)
# -------------------------------------------------------------------
E  = 5.0e6          # Young modulus [Pa]
nu = 0.5            # Poisson ratio (incompressible limit)
mu = E/(2.0*(1.0+nu))   # shear modulus

# -------------------------------------------------------------------
# Function spaces (Taylor‑Hood)
# -------------------------------------------------------------------
V = VectorFunctionSpace(mesh, "CG", 2)   # displacement (P2)
Q = FunctionSpace(mesh, "CG", 1)         # pressure (P1)

# Build mixed space manually (required for recent dolfin versions)
mixed_elem = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, mixed_elem)

# -------------------------------------------------------------------
# Unknown / test functions
# -------------------------------------------------------------------
w = Function(W)               # (u,p) unknown
(u, p) = split(w)             # for variational form
(v, q) = TestFunctions(W)     # test functions

# -------------------------------------------------------------------
# Kinematics
# -------------------------------------------------------------------
d = u.geometric_dimension()          # should be 2
I = Identity(d)                      # identity tensor
F = I + grad(u)                       # deformation gradient
J = det(F)                            # Jacobian

# -------------------------------------------------------------------
# Cauchy stress for incompressible neo‑Hookean (plane strain)
# sigma = -p*I + mu * B,   B = F*F^T
# -------------------------------------------------------------------
B = F*F.T
sigma = -p*I + mu*B

# -------------------------------------------------------------------
# Variational form (mixed)
#   equilibrium: ∫ sigma : grad(v) dx
#   incompressibility: ∫ q (J-1) dx
# -------------------------------------------------------------------
R_mech = inner(sigma, grad(v))*dx
R_incomp = q*(J - 1)*dx
R = R_mech + R_incomp

# Jacobian (automatic differentiation)
Jac = derivative(R, w)

# -------------------------------------------------------------------
# Boundary conditions
# -------------------------------------------------------------------
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], Lx) and on_boundary

left_bc  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), LeftBoundary())
right_bc = DirichletBC(W.sub(0), Constant((0.060, 0.0)), RightBoundary())
bcs = [left_bc, right_bc]

# -------------------------------------------------------------------
# Solve the nonlinear problem
# -------------------------------------------------------------------
problem = NonlinearVariationalProblem(R, w, bcs, Jac)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters
prm['newton_solver']['relative_tolerance'] = 1e-6
prm['newton_solver']['absolute_tolerance'] = 1e-8
prm['newton_solver']['maximum_iterations'] = 30
prm['newton_solver']['linear_solver'] = 'mumps'   # use a direct solver if available
prm['newton_solver']['report'] = True

print("=== Solving the nonlinear problem ===")
solver.solve()
print("=== Solver finished ===")

# -------------------------------------------------------------------
# Extract solutions
# -------------------------------------------------------------------
(u_sol, p_sol) = w.split()

# -------------------------------------------------------------------
# Save displacement (XDMF)
# -------------------------------------------------------------------
xdmf_file = XDMFFile(mesh.mpi_comm(), "displacement.xdmf")
xdmf_file.write(u_sol)
xdmf_file.close()
print("Displacement saved as displacement.xdmf")

# -------------------------------------------------------------------
# Plot & save hydrostatic pressure
# -------------------------------------------------------------------
plt.figure(figsize=(6,4))
p_plot = plot(p_sol, title="Hydrostatic pressure (Pa)", cmap='viridis')
plt.colorbar(p_plot)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q12_p.png", dpi=300)
plt.close()
print("Pressure plot saved as q12_p.png")

# -------------------------------------------------------------------
# Compute von Mises stress from the converged fields
# -------------------------------------------------------------------
# Re‑evaluate kinematics with the converged displacement
F_sol = I + grad(u_sol)
B_sol = F_sol*F_sol.T
sigma_sol = -p_sol*I + mu*B_sol

# Extract components (as scalar Functions)
sigma_xx = sigma_sol[0, 0]
sigma_yy = sigma_sol[1, 1]
sigma_xy = sigma_sol[0, 1]   # = sigma_yx

# Plane‑strain von Mises expression
vm_expr = sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3.0*sigma_xy**2)

V_scalar = FunctionSpace(mesh, "CG", 1)
vm = project(vm_expr, V_scalar)

# Plot & save von Mises stress
plt.figure(figsize=(6,4))
vm_plot = plot(vm, title="Von Mises stress (Pa)", cmap='plasma')
plt.colorbar(vm_plot)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q12_vm.png", dpi=300)
plt.close()
print("Von Mises stress plot saved as q12_vm.png")