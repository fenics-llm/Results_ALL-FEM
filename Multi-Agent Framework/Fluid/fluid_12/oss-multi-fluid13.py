# -*- coding: utf-8 -*-
#
# 2-D channel flow with temperature-dependent viscosity (legacy dolfin)
#
#  Domain: Ω = [0, L] × [0, H]   with L = 2.0 m, H = 0.20 m
#  Governing equations: steady Navier–Stokes + advection–diffusion
#  Viscosity: μ(T) = μ_ref * exp[-β (T - T_ref)]
#
#  Output:
#    - μ(x,y) as PNG   : q13_mu.png
#    - u_x(y) at x = L/2: q13_profile.csv  (columns: y, ux)
#    - Full solution (u,p,T,μ) in XDMF: q13_solution.xdmf
#
#  NOTE: This script is written for the legacy FEniCS (dolfin) API.
#
from dolfin import *
from ufl_legacy import sym, exp   # legacy UFL symbols
from dolfin import near
import numpy as np
import matplotlib.pyplot as plt

# ------ Parameters -------------------------------------------------
L, H = 2.0, 0.20
rho   = 1.0
Ubar  = 1.0
T_ref = 300.0
T_bot = T_ref + 10.0
mu_ref = 0.02
beta   = 0.05
kappa  = 1.0e-3

# ------ Mesh -------------------------------------------------------
Nx, Ny = 80, 20                     # mesh resolution
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), Nx, Ny, "crossed")

# ------ Function spaces ---------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)   # velocity (P2)
Pe = FiniteElement("Lagrange", mesh.ufl_cell(), 1)   # pressure (P1)
TH = MixedElement([Ve, Pe])                           # Taylor–Hood
W  = FunctionSpace(mesh, TH)                          # (u,p)
V_T = FunctionSpace(mesh, "Lagrange", 2)              # temperature (P2)

# collapsed spaces for post-processing
V_u = FunctionSpace(mesh, Ve)      # velocity space
V_p = FunctionSpace(mesh, Pe)      # pressure space

# ------ Boundaries -------------------------------------------------
class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0)

class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H)

inlet   = Inlet()
outlet  = Outlet()
bottom  = Bottom()
top     = Top()

# ------ Inlet velocity profile (Poiseuille) -----------------------
u_inlet_expr = Expression(
    ("6*Ubar*x[1]*(H - x[1])/(H*H)", "0.0"),
    Ubar=Ubar, H=H, degree=2)

# ------ Dirichlet BCs ---------------------------------------------
bcu_inlet  = DirichletBC(W.sub(0), u_inlet_expr, inlet)
bcu_walls  = DirichletBC(W.sub(0), Constant((0.0, 0.0)), bottom)
bcu_walls2 = DirichletBC(W.sub(0), Constant((0.0, 0.0)), top)
bcu = [bcu_inlet, bcu_walls, bcu_walls2]

bcT_inlet = DirichletBC(V_T, Constant(T_ref), inlet)
bcT_bot   = DirichletBC(V_T, Constant(T_bot), bottom)
bcT = [bcT_inlet, bcT_bot]   # top & outlet are natural (Neumann)

# Pressure gauge (pointwise at (0,0) to fix nullspace)
class PressurePoint(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and near(x[1], 0.0)

p_point = PressurePoint()
bcp = DirichletBC(W.sub(1), Constant(0.0), p_point, method="pointwise")
bcu.append(bcp)

# ------ Variational forms -----------------------------------------
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
T      = Function(V_T)          # temperature (unknown)
t      = TrialFunction(V_T)      # trial for temperature
s      = TestFunction(V_T)
T.assign(Constant(T_ref))

mu_expr = mu_ref*exp(-beta*(T - T_ref))

# ------ Picard iteration -----------------------------------------
w   = Function(W)               # mixed (u,p) solution
tol = 1e-6
max_iter = 30

# initialise u_k and p_k (zero fields) on the collapsed spaces
u_k = Function(V_u)
p_k = Function(V_p)
u_k.assign(Constant((0.0, 0.0)))
p_k.assign(Constant(0.0))

for it in range(max_iter):
    # Momentum (Navier–Stokes)
    a11 = rho*dot(dot(u_k, nabla_grad(u)), v)*dx \
          + 2*mu_expr*inner(sym(grad(u)), sym(grad(v)))*dx \
          - div(v)*p*dx + q*div(u)*dx
    L11 = dot(Constant((0.0, 0.0)), v)*dx   # zero RHS
    solve(a11 == L11, w, bcu)

    # extract velocity and pressure as independent Functions (collapse the sub-solutions)
    u_k, p_k = w.split(deepcopy=True)

    # Temperature (advection–diffusion)
    aT = dot(u_k, grad(t))*s*dx + kappa*dot(grad(t), grad(s))*dx
    LT = dot(u_k, grad(T))*s*dx          # explicit advection using previous T
    solve(aT == LT, T, bcT)

    if norm(u_k, 'l2') < tol:
        break

# ------ Post-processing -------------------------------------------
mu_proj = project(mu_expr, V_T)

plt.figure(figsize=(6, 2.5))
p = plot(mu_proj, cmap='viridis')
plt.colorbar(p, label=r'$\mu\,$(Pa·s)')
plt.title(r'Viscosity $\mu(T)$')
plt.xlabel('x (m)'); plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig('q13_mu.png', dpi=300)
plt.close()

y_vals = np.linspace(0.0, H, Ny*5+1)
ux_vals = np.array([u_k(Point(L/2.0, y))[0] for y in y_vals])
np.savetxt('q13_profile.csv',
           np.column_stack([y_vals, ux_vals]),
           delimiter=',',
           header='y,ux',
           comments='')

with XDMFFile(mesh.mpi_comm(), 'q13_solution.xdmf') as xdmf:
    xdmf.write(u_k, 0.0)
    xdmf.write(p_k, 0.0)
    xdmf.write(T,   0.0)
    xdmf.write(mu_proj, 0.0)