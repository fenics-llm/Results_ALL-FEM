# filename: plate_with_hole.py
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
import math

# -------------------------------------------------
# Geometry and mesh
# -------------------------------------------------
L, H = 1.0, 0.20          # plate dimensions (m)
a = 0.05                 # hole radius (m)
center = Point(0.5, 0.1) # hole centre

# Create domain: rectangle minus a circle
domain = Rectangle(Point(0.0, 0.0), Point(L, H)) - Circle(center, a, 64)
mesh = generate_mesh(domain, 64)   # mesh resolution (increase for finer results)

# -------------------------------------------------
# Function spaces
# -------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", degree=2)   # displacement
W = FunctionSpace(mesh, "DG", 0)                     # piecewise‑constant stress

# -------------------------------------------------
# Material parameters (plane stress)
# -------------------------------------------------
E  = 200e9          # Pa
nu = 0.30
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / (1.0 - nu**2)   # plane‑stress Lamé parameter

def sigma(u):
    """Cauchy stress tensor for plane‑stress linear elasticity."""
    eps = sym(grad(u))
    return lmbda * tr(eps) * Identity(2) + 2.0 * mu * eps

# -------------------------------------------------
# Boundary definitions
# -------------------------------------------------
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L) and on_boundary

# Mark facets
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
left = LeftBoundary()
right = RightBoundary()
left.mark(boundaries, 1)
right.mark(boundaries, 2)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Boundary conditions
# -------------------------------------------------
zero = Constant((0.0, 0.0))
bc_left = DirichletBC(V, zero, left)
bcs = [bc_left]

# -------------------------------------------------
# Variational problem
# -------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

a_form = inner(sigma(u), sym(grad(v))) * dx

# Neumann traction on the right edge: (2 MPa, 0)
t_val = Constant((2e6, 0.0))
L_form = dot(t_val, v) * ds(2)

# Solve
u_sol = Function(V, name="Displacement")
solve(a_form == L_form, u_sol, bcs)

# -------------------------------------------------
# Post‑processing: von Mises stress (plane stress)
# -------------------------------------------------
stress = sigma(u_sol)

sigma_xx = stress[0, 0]
sigma_yy = stress[1, 1]
sigma_xy = stress[0, 1]

von_mises_expr = sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3.0*sigma_xy**2)
von_mises = project(von_mises_expr, W)   # no 'name' argument

# -------------------------------------------------
# Save displacement field (XDMF)
# -------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "plate_displacement.xdmf")
xdmf.write(u_sol)
xdmf.close()

# -------------------------------------------------
# Plot von Mises stress and save as PNG
# -------------------------------------------------
plt.figure(figsize=(8, 3))
p = plot(von_mises, title="Von Mises stress (Pa)", cmap="viridis")
plt.colorbar(p, label='Stress [Pa]')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.tight_layout()
plt.savefig("q3_vm.png", dpi=300)
plt.close()

# -------------------------------------------------
# Maximum von Mises stress on the hole boundary
# -------------------------------------------------
# Sample points uniformly on the hole circumference
n_pts = 360
angles = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
hole_points = [Point(center.x() + a * math.cos(theta),
                     center.y() + a * math.sin(theta)) for theta in angles]

# Evaluate von Mises at those points
vm_vals = np.array([von_mises(p) for p in hole_points])

max_vm_hole = np.max(vm_vals)          # Pa
Kt = max_vm_hole / 2e6                 # reference stress = 2 MPa

print(f"Maximum von Mises stress on hole boundary: {max_vm_hole/1e6:.3f} MPa")
print(f"Stress concentration factor Kt = {Kt:.3f}")