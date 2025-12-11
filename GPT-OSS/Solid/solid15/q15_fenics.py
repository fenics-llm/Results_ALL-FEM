# filename: q15_fenics.py
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
import matplotlib.pyplot as plt

# -------------------------------------------------
# Geometry
Lx, Ly = 1.20, 0.20          # domain size (m)
a = 0.03                     # hole radius (m)
hole_centers = [Point(0.30, 0.10), Point(0.60, 0.10), Point(0.90, 0.10)]

# Build rectangle with three circular holes
domain = Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
for c in hole_centers:
    domain = domain - Circle(c, a, 64)

# Mesh (increase resolution if needed)
mesh = generate_mesh(domain, 64)

# -------------------------------------------------
# Function space (quadratic Lagrange for displacement)
V = VectorFunctionSpace(mesh, "Lagrange", degree=2)

# -------------------------------------------------
# Material parameters (Pa)
lam = 5.769e6   # λ
mu  = 3.846e6   # μ

# -------------------------------------------------
# Boundary conditions
tol = 1e-8

def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0, tol)

def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], Lx, tol)

# Left edge: fixed
bc_left = DirichletBC(V, Constant((0.0, 0.0)), left_boundary)

# Right edge: prescribed horizontal displacement (updated each step)
total_disp = 0.012                     # 12 mm
u_R_expr = Expression(("t*dx", "0.0"), t=0.0, dx=total_disp, degree=1)
bc_right = DirichletBC(V, u_R_expr, right_boundary)

bcs = [bc_left, bc_right]

# -------------------------------------------------
# Kinematics (UFL expressions)
u = Function(V, name="Displacement")   # unknown displacement
du = TrialFunction(V)
v  = TestFunction(V)

I = Identity(2)                         # 2‑D identity tensor
F = I + grad(u)                         # deformation gradient
C = F.T*F                               # right Cauchy‑Green tensor
E = 0.5*(C - I)                         # Green–Lagrange strain

# Second Piola–Kirchhoff stress
S = lam*tr(E)*I + 2.0*mu*E

# First Piola–Kirchhoff stress
P = F*S

# Weak form (internal virtual work)
R = inner(P, grad(v))*dx

# Jacobian for Newton's method
J = derivative(R, u, du)

# -------------------------------------------------
# Non‑linear solver setup
problem = NonlinearVariationalProblem(R, u, bcs, J)
solver  = NonlinearVariationalSolver(problem)

prm = solver.parameters["newton_solver"]
prm["absolute_tolerance"] = 1e-8
prm["relative_tolerance"] = 1e-6
prm["maximum_iterations"] = 25
prm["relaxation_parameter"] = 1.0

# -------------------------------------------------
# Load stepping
max_steps = 30                     # maximum number of increments
Emax_func = Function(FunctionSpace(mesh, "DG", 0), name="Emax")
step_done = 0

for step in range(1, max_steps + 1):
    # Linear ramp for the prescribed displacement
    u_R_expr.t = float(step) / max_steps

    # Solve the non‑linear problem for the current load step
    solver.solve()

    # ----- evaluate maximum principal Green‑Lagrange strain -----
    # λ₁ = (Eₓₓ+Eyy)/2 + sqrt(((Eₓₓ-Eyy)/2)² + Eₓy²)
    E_xx = E[0, 0]
    E_yy = E[1, 1]
    E_xy = E[0, 1]

    E1_expr = (E_xx + E_yy) / 2 + sqrt(((E_xx - E_yy) / 2)**2 + E_xy**2)
    E1 = project(E1_expr, FunctionSpace(mesh, "DG", 0))

    Emax_val = E1.vector().get_local().max()

    if Emax_val > 0.03:
        # Exceeded allowed strain – revert to previous step and stop
        u_R_expr.t = float(step - 1) / max_steps
        solver.solve()                     # recompute solution at last admissible step
        # recompute strain at the reverted step
        E1 = project(E1_expr, FunctionSpace(mesh, "DG", 0))
        Emax_func.assign(E1)
        break
    else:
        step_done = step
        Emax_func.assign(E1)

# -------------------------------------------------
# Post‑processing -------------------------------------------------

# 1) Deformed configuration (saved as PNG)
mesh_def = Mesh(mesh)                     # copy original mesh
V_disp = VectorFunctionSpace(mesh, "CG", 1)
u_disp = interpolate(u, V_disp)          # displacement on CG1
ALE.move(mesh_def, u_disp)

plt.figure(figsize=(6, 3))
plot(mesh_def, linewidth=0.5)
plt.title("Deformed configuration (scale = 1)")
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q15_def.png", dpi=300)

# 2) Maximum principal Green‑Lagrange strain map
plt.figure(figsize=(6, 3))
p = plot(Emax_func, title=r"Maximum principal $E$")
plt.colorbar(p)
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q15_Emax.png", dpi=300)

# 3) von Mises stress based on S
trS = tr(S)
s_dev = S - (1.0/3.0)*trS*I               # deviatoric part of S
vm_expr = sqrt(1.5*inner(s_dev, s_dev))
vm = project(vm_expr, FunctionSpace(mesh, "DG", 0))
vm.rename("von_Mises_S", "")

plt.figure(figsize=(6, 3))
p = plot(vm, title=r"von Mises stress (based on $S$)")
plt.colorbar(p)
plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q15_vmS.png", dpi=300)

# -------------------------------------------------
# Export results (XDMF)
with XDMFFile("q15_u.xdmf") as file_u:
    file_u.write(u)

with XDMFFile("q15_Emax.xdmf") as file_E:
    file_E.write(Emax_func)

print(f"Simulation finished after {step_done} load steps.")