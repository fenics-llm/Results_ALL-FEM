# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dolfin import *

# ----------------------------------------------------------------------
# 1. Mesh and geometry
# ----------------------------------------------------------------------
Lx, Ly = 1.0, 0.20
nx, ny = 50, 25
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny, "crossed")

# ----------------------------------------------------------------------
# 2. Function space (P2 vector)
# ----------------------------------------------------------------------
Ve = VectorElement("Lagrange", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, Ve)

# ----------------------------------------------------------------------
# 3. Material data (orthotropic plane-stress, rotated 30°)
# ----------------------------------------------------------------------
theta = np.pi/6.0                     # 30°
c, s = np.cos(theta), np.sin(theta)

# local orthotropic constants
E1, E2 = 40e9, 10e9                 # Pa
G12 = 5e9
nu12 = 0.25
nu21 = nu12*E2/E1

# reduced stiffness in material axes (plane-stress)
Delta = 1.0 - nu12*nu21
Q = np.array([[E1/Delta,      nu12*E2/Delta, 0.0],
              [nu21*E1/Delta, E2/Delta,      0.0],
              [0.0,            0.0,          G12]])

# rotation matrix for Voigt (xx,yy,xy) components
T = np.array([[ c*c,      s*s,      2*c*s],
              [ s*s,      c*c,     -2*c*s],
              [-c*s,      c*s,  c*c - s*s]])

C = T.T @ Q @ T                      # global plane-stress stiffness (3×3)

# ----------------------------------------------------------------------
# 4. Boundary conditions
# ----------------------------------------------------------------------
tol = 1e-10

def bottom(x, on_boundary):
    return on_boundary and abs(x[1]) < tol

bc = DirichletBC(V, Constant((0.0, 0.0)), bottom)

# Mark top boundary for Neumann traction
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - Ly) < tol
Top().mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

t_val = -10e6                         # -10 MPa (downward)
traction = Constant((0.0, t_val))

# ----------------------------------------------------------------------
# 5. Variational formulation
# ----------------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

def eps(v):
    return sym(grad(v))

def sigma(v):
    e = eps(v)
    gamma = 2.0*e[0, 1]               # engineering shear strain
    s_xx = C[0, 0]*e[0, 0] + C[0, 1]*e[1, 1] + C[0, 2]*gamma
    s_yy = C[1, 0]*e[0, 0] + C[1, 1]*e[1, 1] + C[1, 2]*gamma
    s_xy = C[2, 0]*e[0, 0] + C[2, 1]*e[1, 1] + C[2, 2]*gamma
    return as_tensor([[s_xx, s_xy],
                       [s_xy, s_yy]])

a = inner(sigma(u), eps(v))*dx
L = dot(traction, v)*ds(1)

# ----------------------------------------------------------------------
# 6. Solve
# ----------------------------------------------------------------------
u_sol = Function(V)
solve(a == L, u_sol, bc)

# ----------------------------------------------------------------------
# 7. Post-processing: stress, von Mises, PNGs, XDMF
# ----------------------------------------------------------------------
# stress tensor field (projected to DG0 for output)
W_sigma = TensorFunctionSpace(mesh, "DG", 0)
sigma_proj = project(sigma(u_sol), W_sigma)

# von Mises stress
s = sigma_proj
vm = sqrt(s[0, 0]**2 - s[0, 0]*s[1, 1] + s[1, 1]**2 + 3.0*s[0, 1]**2)

# Plot horizontal displacement ux
plt.figure()
p = plot(u_sol.sub(0), title=r"$u_x$", cmap="viridis")
plt.colorbar(p)
plt.savefig("q8_ux.png", dpi=300)

# Plot von Mises stress
plt.figure()
p = plot(vm, title=r"$\sigma_{\mathrm{vm}}$", cmap="viridis")
plt.colorbar(p)
plt.savefig("q8_vm.png", dpi=300)

# Write solution to XDMF
xdmf = XDMFFile(mesh.mpi_comm(), "q8_solution.xdmf")
xdmf.write(u_sol, 0.0)
xdmf.write(sigma_proj, 0.0)
xdmf.close()