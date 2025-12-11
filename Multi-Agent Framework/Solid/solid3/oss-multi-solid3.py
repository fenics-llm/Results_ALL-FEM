# -*- coding: utf-8 -*-
#
# Plane-stress elasticity of a rectangular plate with a centred circular hole.
# Left edge clamped, right edge loaded with 2 MPa·m (plane-stress traction).
# Output: von Mises stress PNG, displacement XDMF, max stress on hole & Kt.
#
# Legacy FEniCS (dolfin) implementation.
#
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh
from dolfin import SubsetIterator          # needed for hole vertex extraction
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------
# 1. Geometry & mesh
# --------------------------------------------------------------
Lx, Ly = 1.0, 0.20          # plate dimensions (m)
a = 0.05                     # hole radius (m)
hole_center = Point(0.5, 0.1)

plate = Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
hole  = Circle(hole_center, a, 64)
domain = plate - hole

# Generate mesh (refine near hole)
mesh = generate_mesh(domain, 200)   # adjust resolution if needed

# --------------------------------------------------------------
# 2. Function space
# --------------------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", degree=2)

# --------------------------------------------------------------
# 3. Boundary definitions
# --------------------------------------------------------------
tol = 1E-8

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, tol)

class HoleBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and ((x[0]-hole_center.x())**2 + (x[1]-hole_center.y())**2 < (a+tol)**2)

left  = LeftBoundary()
right = RightBoundary()
hole  = HoleBoundary()

# Dirichlet BC on left edge (u = 0)
bc_left = DirichletBC(V, Constant((0.0, 0.0)), left)

# --------------------------------------------------------------
# 3b. Facet markers for Neumann BCs
# --------------------------------------------------------------
boundaries = MeshFunction('size_t', mesh, mesh.topology().dim() - 1, 0)
boundaries.set_all(0)
left.mark(boundaries, 1)      # left  = 1 (Dirichlet)
right.mark(boundaries, 2)     # right = 2 (traction)
hole.mark(boundaries, 3)      # hole  = 3 (traction-free)

# ds measure using the markers
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# --------------------------------------------------------------
# 4. Material parameters (plane-stress)
# --------------------------------------------------------------
E  = 200e9          # Pa
nu = 0.30
mu = E/(2.0*(1.0+nu))
lmbda = E*nu/(1.0 - nu**2)

# Plane-stress stiffness matrix
C = as_matrix(((lmbda+2*mu, lmbda, 0.0),
                (lmbda, lmbda+2*mu, 0.0),
                (0.0, 0.0, mu)))

def sigma(v):
    """Plane-stress Cauchy stress."""
    eps = sym(grad(v))
    eps_vec = as_vector((eps[0,0], eps[1,1], 2*eps[0,1]))
    sig_vec = dot(C, eps_vec)
    # 2×2 stress tensor
    return as_tensor(((sig_vec[0], sig_vec[2]), (sig_vec[2], sig_vec[1])))

# --------------------------------------------------------------
# 5. Variational problem
# --------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

a = inner(sigma(u), sym(grad(v))) * dx

# Traction on right edge: (2 MPa·m, 0) → 2e6 N/m
t_val = Constant((2e6, 0.0))
L = dot(t_val, v) * ds(2)   # apply only on right boundary (marker 2)

# Solve
u_sol = Function(V, name="Displacement")
solve(a == L, u_sol, bc_left, solver_parameters={"linear_solver":"mumps"})

# --------------------------------------------------------------
# 6. Post-processing: von Mises stress (plane-stress)
# --------------------------------------------------------------
s = sigma(u_sol)
s_xx = s[0, 0]
s_yy = s[1, 1]
s_xy = s[0, 1]

von_mises = sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3*s_xy**2)
von_mises = project(von_mises, FunctionSpace(mesh, "Lagrange", 2))
von_mises.rename("von_Mises", "von_Mises")

# --------------------------------------------------------------
# 7. Evaluate max stress on the hole boundary and Kt
# --------------------------------------------------------------
# Build set of vertex indices belonging to the hole facets (marker 3)
hole_vert_set = set()
for f in SubsetIterator(boundaries, 3):
    for v in vertices(f):
        hole_vert_set.add(v.index())

# Vertex values of von Mises
vm_vals = von_mises.compute_vertex_values(mesh)

# Extract values at hole vertices
hole_vals = np.array([vm_vals[i] for i in hole_vert_set])
sigma_max_hole = hole_vals.max()
K_t = sigma_max_hole / 2e6

print("\n=== Results ===")
print("Maximum von Mises stress on hole boundary : {:.2e} Pa".format(sigma_max_hole))
print("Stress concentration factor Kt            : {:.3f}".format(K_t))

# --------------------------------------------------------------
# 8. Save outputs
# --------------------------------------------------------------
# XDMF for displacement
with XDMFFile(mesh.mpi_comm(), "q3_disp.xdmf") as xdmf:
    xdmf.write(u_sol)

# PNG of von Mises
plt.figure(figsize=(8, 3))
p = plot(von_mises, title="von Mises stress (Pa)", cmap="viridis")
plt.colorbar(p)
plt.axis("off")
plt.tight_layout()
plt.savefig("q3_vm.png", dpi=300)
plt.close()

# XDMF for von Mises (optional)
with XDMFFile(mesh.mpi_comm(), "q3_vm.xdmf") as xdmf:
    xdmf.write(von_mises)

print("\nOutputs written: q3_vm.png, q3_disp.xdmf, q3_vm.xdmf")