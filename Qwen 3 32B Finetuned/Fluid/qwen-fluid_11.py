from dolfin import *
from mshr import Rectangle, generate_mesh
import matplotlib.pyplot as plt
import numpy as np

# Geometry
mesh = generate_mesh(Rectangle(Point(0, 0), Point(2.0, 0.2)), 240*24)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
ME = MixedElement([V.ufl_element(), Q.ufl_element()])
W = FunctionSpace(mesh, ME)
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Parameters
rho = 1.0
mu0 = 0.5
n = 0.5
U_bar = 1.0
H = 0.2

# Boundary conditions
bcs = []
# Inlet (x=0)
inlet = Expression(("6*U_bar*x[1]*(H - x[1])/(H*H)", "0.0"), degree=2, U_bar=U_bar, H=H)
bc_inlet = DirichletBC(W.sub(0), inlet, "near(x[0], 0.0)")
bcs.append(bc_inlet)
# Walls (y=0, y=H)
noslip = Constant((0.0, 0.0))
bc_walls = DirichletBC(W.sub(0), noslip, "near(x[1], 0.0) || near(x[1], 0.2)")
bcs.append(bc_walls)

# Variational form
D = sym(grad(u))
mag_D = sqrt(2 * inner(D, D) + 1e-16)
mu_eff = mu0 * power(mag_D, n - 1)
tau = 2 * mu_eff * D
F = (rho * dot(u, grad(u)) * v - div(tau) * v + div(u) * q) * dx
a, L = lhs(F), rhs(F)

# Solve
w = Function(W)
solve(a == L, w, bcs, solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}})
u_h, p_h = w.split()

# Post-processing
V_mag = FunctionSpace(mesh, "CG", 1)
u_mag = project(sqrt(dot(u_h, u_h)), V_mag)
plt.figure()
p = plot(u_mag)
plt.colorbar(p)
plt.savefig("q12_speed.png")

# Mid-channel profile
x_mid = 1.0
y_vals = np.linspace(0, 0.2, 100)
ux_profile = [u_h((x_mid, y))[0] for y in y_vals]
np.savetxt("q12_profile.csv", np.column_stack((y_vals, ux_profile)))

# Save solution
with XDMFFile("q12_solution.xdmf") as file:
    file.write(u_h, 0)
    file.write(p_h, 0)
    file.write(project(mu_eff, Q), 0)

# Max velocity at x=1.0
max_ux = max(ux_profile)
print(f"Maximum velocity at x=1.0: {max_ux:.4f} m/s")
