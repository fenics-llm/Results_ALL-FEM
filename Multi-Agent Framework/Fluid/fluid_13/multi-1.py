# -*- coding: utf-8 -*-
"""
Steady advection–diffusion of a scalar in a 2‑D channel using legacy FEniCS.
Prescribed Poiseuille velocity; SUPG stabilization added due to high Peclet number.
"""

from dolfin import (RectangleMesh, Point, FunctionSpace, TrialFunction,
                    TestFunction, Function, Constant, DirichletBC,
                    Expression, inner, grad, dx, solve,
                    XDMFFile, CellDiameter, near, DOLFIN_EPS)
from ufl_legacy import sqrt   # <-- legacy UFL import

# ------ Geometry and mesh ----------------------------------------------------
L = 1.0          # channel length (m)
H = 0.10         # channel height (m)
Nx = 100         # elements along x
Ny = 10          # elements along y
mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), Nx, Ny, "crossed")

# ------ Prescribed velocity field (Poiseuille profile) -----------------------
U_max = 0.75                     # max velocity (m/s)
D_val = 1.0e-5                   # diffusivity (m^2/s)

u_expr = Expression(("U_max*4.0*x[1]*(H - x[1]) / (H*H)", "0.0"),
                    U_max=U_max, H=H, degree=2)

# ------ Function space for concentration --------------------------------------
V = FunctionSpace(mesh, "Lagrange", 1)   # linear Lagrange

# ------ Boundary conditions --------------------------------------------------
inlet = lambda x, on: on and near(x[0], 0.0, DOLFIN_EPS)
bc_in = DirichletBC(V, Constant(0.0), inlet)

outlet = lambda x, on: on and near(x[0], L, DOLFIN_EPS)
bc_out = DirichletBC(V, Constant(1.0), outlet)

bcs = [bc_in, bc_out]   # walls have natural (zero flux) condition

# ------ SUPG stabilization parameter τ ---------------------------------------
h = CellDiameter(mesh)
u_norm = sqrt(u_expr[0]**2 + u_expr[1]**2) + DOLFIN_EPS
tau = h / (2.0 * u_norm)   # elementwise τ

# ------ Variational problem (SUPG) -----------------------------------------
c = TrialFunction(V)
v = TestFunction(V)

# Galerkin terms
a = D_val * inner(grad(c), grad(v)) * dx \
    + inner(u_expr, grad(c)) * v * dx

# SUPG term (streamline direction = u)
a += tau * inner(u_expr, grad(c)) * inner(u_expr, grad(v)) * dx

# Zero RHS (no source term)
Lrhs = Constant(0.0) * v * dx

# ------ Solve ---------------------------------------------------------------
c_sol = Function(V)
solve(a == Lrhs, c_sol, bcs)

# ------ Output to XDMF ------------------------------------------------------
with XDMFFile(mesh.mpi_comm(), "c_channel.xdmf") as xdmf:
    xdmf.write(c_sol, 0.0)