# -*- coding: utf-8 -*-
"""
Legacy FEniCS (dolfin) solver for 2‑D channel flow with temperature‑dependent viscosity.
Outputs:
    - q13_mu.png          : colour map of viscosity μ(T)
    - q13_profile.csv    : centre‑line u_x(y) at x = 1.0
    - q13_solution.xdmf  : XDMF file containing u, p, T, μ
"""

from dolfin import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------ Geometry and mesh -------------------------------------------------
L, H = 2.0, 0.20
Nx, Ny = 80, 20
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), Nx, Ny, "crossed")

# ------ Physical parameters ------------------------------------------------
rho    = 1.0
Ubar   = 1.0
mu_ref = 0.02
beta   = 0.05
T_ref  = 300.0
kappa  = 1.0e-3

# ------ Function spaces (Taylor–Hood + temperature) -----------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = MixedElement([Ve, Pe])
W_velp = FunctionSpace(mesh, TH)          # (u,p)
W_T    = FunctionSpace(mesh, "Lagrange", 2)  # T

# ------ Boundary markers --------------------------------------------------
tol = 1e-10
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)
class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, tol)
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H, tol)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
Inlet().mark(boundaries, 1)
Outlet().mark(boundaries, 2)
Bottom().mark(boundaries, 3)
Top().mark(boundaries, 4)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# ------ Dirichlet BCs ----------------------------------------------------
u_inlet_expr = Expression(
    ("6.0*Ubar*x[1]*(H - x[1])/(H*H)", "0.0"),
    degree=2, Ubar=Ubar, H=H)

bcu_inlet = DirichletBC(W_velp.sub(0), u_inlet_expr, boundaries, 1)
bcu_walls = DirichletBC(W_velp.sub(0), Constant((0.0, 0.0)), boundaries, 3)
bcu_walls_top = DirichletBC(W_velp.sub(0), Constant((0.0, 0.0)), boundaries, 4)

bct_inlet = DirichletBC(W_T, Constant(T_ref), boundaries, 1)
bct_bottom = DirichletBC(W_T, Constant(T_ref + 10.0), boundaries, 3)

bcu = [bcu_inlet, bcu_walls, bcu_walls_top]
bct = [bct_inlet, bct_bottom]

# ------ Mixed unknowns and test functions ---------------------------------
w = Function(W_velp)               # (u,p)
(u, p) = split(w)
(v, q) = TestFunctions(W_velp)

T = Function(W_T)                  # temperature
T.interpolate(Constant(T_ref))      # initialise temperature
psi = TestFunction(W_T)

# ------ Viscosity as a Function of T ------------------------------------
mu = mu_ref*exp(-beta*(T - T_ref))

# ------ Weak forms (separate blocks) ------------------------------------
def epsilon(v):
    return sym(grad(v))

# Navier–Stokes residual
F_mom = ( rho*dot(dot(u, nabla_grad(u)), v)*dx
          + 2*mu*inner(epsilon(u), epsilon(v))*dx
          - p*div(v)*dx
          + q*div(u)*dx )

# Temperature residual
F_temp = ( dot(u, nabla_grad(T))*psi*dx
           + kappa*dot(grad(T), grad(psi))*dx )

# ------ Pressure nullspace (fix at a point) -----------------------------
class PointPressure(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0, tol) and near(x[1], 0.0, tol)
point_bc = DirichletBC(W_velp.sub(1), Constant(0.0), PointPressure(), "pointwise")
bcu.append(point_bc)

# ------ Picard iteration (update mu(T) -> Navier–Stokes -> Temperature) --
max_iter = 8
tol_u = 1e-8
tol_T = 1e-8

w_old = Function(W_velp)
T_old = Function(W_T)

for it in range(max_iter):
    # update viscosity from current temperature
    mu = mu_ref*exp(-beta*(T - T_ref))

    # Navier–Stokes block (direct LU solve)
    J_mom = derivative(F_mom, w, TrialFunction(W_velp))
    solve(F_mom == 0, w, bcu, J=J_mom,
          solver_parameters={"newton_solver":
                             {"relative_tolerance": tol_u,
                              "absolute_tolerance": 1e-10,
                              "maximum_iterations": 30,
                              "linear_solver": "lu"}})   # <-- preconditioner removed

    # Temperature block
    J_T = derivative(F_temp, T, TrialFunction(W_T))
    solve(F_temp == 0, T, bct, J=J_T,
          solver_parameters={"newton_solver":
                             {"relative_tolerance": tol_T,
                              "absolute_tolerance": 1e-10,
                              "maximum_iterations": 30,
                              "linear_solver": "gmres",
                              "preconditioner": "ilu"}})

    # convergence check
    du = norm(w.vector() - w_old.vector(), 'l2')
    dT = norm(T.vector() - T_old.vector(), 'l2')
    if du < 1e-9 and dT < 1e-9:
        print(f'Picard converged after {it+1} iterations')
        break
    w_old.assign(w)
    T_old.assign(T)

# ------ Extract final fields ---------------------------------------------
(u_sol, p_sol) = w.split(deepcopy=True)
T_sol = T

# ------ Viscosity field for output ------------------------------------
mu_sol = project(mu, FunctionSpace(mesh, "Lagrange", 1))

# ------ Save viscosity image --------------------------------------------
plt.figure(figsize=(6, 2))
p = plot(mu_sol, cmap='viridis')
plt.colorbar(p, label=r'$\mu\;({\rm Pa\,s})$')
plt.title(r'Viscosity $\mu(T)$')
plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig('q13_mu.png', dpi=300)
plt.close()

# ------ Centre‑line velocity profile (x = 1.0) -----------------------
x_mid = L/2
y_vals = np.linspace(0.0, H, Ny*5+1)
u_mid = np.array([u_sol(Point(x_mid, y))[0] for y in y_vals])

np.savetxt('q13_profile.csv',
           np.column_stack([y_vals, u_mid]),
           delimiter=',',
           header='y, ux',
           comments='')

# ------ Export solution in XDMF ----------------------------------------
with XDMFFile(mesh.mpi_comm(), 'q13_solution.xdmf') as xdmf:
    xdmf.write(u_sol, 0.0)
    xdmf.write(p_sol, 0.0)
    xdmf.write(T_sol, 0.0)
    xdmf.write(mu_sol, 0.0)

print('Simulation completed. Files written: q13_mu.png, q13_profile.csv, q13_solution.xdmf')