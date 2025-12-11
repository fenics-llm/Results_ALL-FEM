# filename: q14_brinkman_navier_stokes.py
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. Mesh and subdomains
# ------------------------------------------------------------
L, H = 1.0, 0.20
Nx, Ny = 200, 40
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), Nx, Ny)

# Porous filter region Π = [0.4,0.6] × [0,0.20]
class Porous(SubDomain):
    def inside(self, x, on_boundary):
        return (0.4 - DOLFIN_EPS <= x[0] <= 0.6 + DOLFIN_EPS)

porous = Porous()
domains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
porous.mark(domains, 1)               # 1 → porous, 0 → fluid
dx = Measure('dx', domain=mesh, subdomain_data=domains)
dx_f = dx(0)                          # fluid part
dx_p = dx(1)                          # porous part

# ------------------------------------------------------------
# 2. Function spaces (Taylor–Hood)
# ------------------------------------------------------------
Ve = VectorElement('Lagrange', mesh.ufl_cell(), 2)
Pe = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W  = FunctionSpace(mesh, TH)

# ------------------------------------------------------------
# 3. Boundary conditions
# ------------------------------------------------------------
U_bar = 1.0
mu    = Constant(0.01)   # Pa·s
rho   = Constant(1.0)    # kg/m³
K_val = 1e-6
alpha = mu / K_val         # 1e4

# Inlet velocity profile (Poiseuille)
inlet_profile = ('6.0*U_bar*x[1]*(H - x[1]) / (H*H)', '0.0')
inlet_expr = Expression(inlet_profile, U_bar=U_bar, H=H, degree=2)

# Boundary markers
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[1], 0.0) or near(x[1], H))
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)

bcs = []
# Velocity Dirichlet on inlet and walls
bcs.append(DirichletBC(W.sub(0), inlet_expr, Inlet()))
bcs.append(DirichletBC(W.sub(0), Constant((0.0, 0.0)), Walls()))
# Pressure gauge (fix p at a point to avoid nullspace)
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)
bcs.append(DirichletBC(W.sub(1), Constant(0.0), PressurePoint(), method='pointwise'))

# ------------------------------------------------------------
# 4. Variational formulation
# ------------------------------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Nonlinear term (only in fluid)
U = Function(W)                     # current iterate (u,p)
(u_k, p_k) = split(U)

# Brinkman term (Darcy drag) active only in porous region
a_drag = alpha * inner(u_k, v) * dx_p

# Viscous term (fluid + porous)
a_visc = 2*mu * inner(sym(grad(u_k)), sym(grad(v))) * dx

# Convective term (fluid only)
a_conv = rho * inner(dot(u_k, nabla_grad(u_k)), v) * dx_f

# Pressure-divergence coupling (global)
b_div  = - div(v) * p_k * dx
c_div  = + q * div(u_k) * dx

F = a_visc + a_drag + a_conv + b_div + c_div

# Jacobian
J = derivative(F, U, TrialFunction(W))

# ------------------------------------------------------------
# 5. Solve nonlinear problem
# ------------------------------------------------------------
problem = NonlinearVariationalProblem(F, U, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['relative_tolerance'] = 1e-6
prm['newton_solver']['absolute_tolerance'] = 1e-8
prm['newton_solver']['maximum_iterations'] = 25
prm['newton_solver']['linear_solver'] = 'mumps'
solver.solve()

# Split solution
(u_sol, p_sol) = U.split(deepcopy=True)

# ------------------------------------------------------------
# 6. Post-processing
# ------------------------------------------------------------
# Velocity magnitude
speed = sqrt(dot(u_sol, u_sol))

# Save solution to XDMF
with XDMFFile(mesh.mpi_comm(), "q14_solution.xdmf") as xdmf:
    xdmf.write(u_sol, 0.0)
    xdmf.write(p_sol, 0.0)

# Plot speed magnitude and save as PNG
plt.figure(figsize=(8, 3))
p = plot(speed, title='Velocity magnitude |u| (m/s)', cmap='viridis')
plt.colorbar(p)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig('q14_speed.png', dpi=300)
plt.close()

# ------------------------------------------------------------
# 7. Pressure drop across the porous block (centerline y = H/2)
# ------------------------------------------------------------
y_mid = H / 2.0
eps   = mesh.hmin() * 0.5
x_before = 0.4 - eps
x_after  = 0.6 + eps
p_before = p_sol(Point(x_before, y_mid))
p_after  = p_sol(Point(x_after,  y_mid))
dp = p_before - p_after

# Save pressure drop
with open('q14_dp.txt', 'w') as f:
    f.write('Pressure drop across porous block (Pa): {:.6e}\n'.format(dp))

print('Pressure drop Δp = {:.6e} Pa'.format(dp))