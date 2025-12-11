# -*- coding: utf-8 -*-
from dolfin import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------
# 1. Mesh and subdomain markers
# --------------------------------------------------------------
nx, ny = 80, 16
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 0.20), nx, ny, "crossed")

materials = MeshFunction("size_t", mesh, mesh.topology().dim())
materials.set_all(0)  # steel = 0, aluminum = 1

class Aluminum(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 0.10 - DOLFIN_EPS

Aluminum().mark(materials, 1)

# --------------------------------------------------------------
# 2. Facet markers for Neumann boundaries
# --------------------------------------------------------------
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)

class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1.0) and on_boundary

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0) and on_boundary

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.20) and on_boundary

Left().mark(boundaries, 1)
Right().mark(boundaries, 2)
Bottom().mark(boundaries, 3)
Top().mark(boundaries, 4)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# --------------------------------------------------------------
# 3. Function space (vector P1) and Dirichlet BC on left edge
# --------------------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 1)

zero = Constant((0.0, 0.0))
bc_left = DirichletBC(V, zero, boundaries, 1)

# --------------------------------------------------------------
# 4. Material parameters (plane stress)
# --------------------------------------------------------------
nu = 0.30
E_steel = 2.0e11   # Pa
E_al = 7.0e10       # Pa

# piecewise constant Young's modulus (DG0)
V0 = FunctionSpace(mesh, "DG", 0)
E = Function(V0)
E_vec = np.where(materials.array() == 1, E_al, E_steel)
E.vector()[:] = E_vec

# Lame parameters as DG0 functions
mu = Function(V0)
lmbda = Function(V0)
mu.vector()[:] = E_vec / (2.0 * (1.0 + nu))
lmbda.vector()[:] = E_vec * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
lmbda.vector()[:] = (2.0 * mu.vector().get_local() * lmbda.vector().get_local()) / (lmbda.vector().get_local() + 2.0 * mu.vector().get_local())

# --------------------------------------------------------------
# 5. Strain and stress (plane stress reduction)
# --------------------------------------------------------------
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda * tr(epsilon(u)) * Identity(2) + 2.0 * mu * epsilon(u)

# --------------------------------------------------------------
# 6. Variational problem
# --------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

a = inner(sigma(u), epsilon(v)) * dx
t = Constant((0.0, -5000.0))  # N/m, downward on right edge
L = dot(t, v) * ds(2)

# --------------------------------------------------------------
# 7. Solve
# --------------------------------------------------------------
u_sol = Function(V)
solve(a == L, u_sol, bc_left, solver_parameters={"linear_solver": "mumps"})

# --------------------------------------------------------------
# 8. Post-processing: displacement magnitude and output
# --------------------------------------------------------------
V_scalar = FunctionSpace(mesh, "Lagrange", 1)
u_magnitude = project(sqrt(dot(u_sol, u_sol)), V_scalar)

# Save XDMF
with XDMFFile(mesh.mpi_comm(), "q7_disp.xdmf") as xdmf:
    xdmf.write(u_sol, 0.0)

# Plot magnitude to PNG
plt.figure(figsize=(6, 3))
p = plot(u_magnitude, title="Displacement magnitude (m)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q7_disp.png", dpi=300)
plt.close()