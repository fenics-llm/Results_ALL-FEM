# filename: plane_stress_plate.py
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# Geometry and mesh
Lx, Ly = 1.0, 0.20
nx, ny = 40, 8
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny, "crossed")

# -------------------------------------------------
# Function space (Q2 elements)
V = VectorFunctionSpace(mesh, "Lagrange", degree=2)

# -------------------------------------------------
# Material parameters (plane stress)
E  = 200e9          # Pa
nu = 0.30
mu = E / (2.0*(1.0 + nu))
lmbda = E*nu / ((1.0 + nu)*(1.0 - 2.0*nu))   # 3D lambda, will be modified for plane stress
# Plane stress reduction
lmbda_ps = (E*nu) / (1.0 - nu**2)
mu_ps    = E / (2.0*(1.0 + nu))

def sigma(eps):
    """Plane‑stress stress tensor."""
    return lmbda_ps*tr(eps)*Identity(2) + 2.0*mu_ps*eps

def epsilon(u):
    return sym(grad(u))

# -------------------------------------------------
# Boundary definitions
class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], Ly) and on_boundary

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
left = LeftBoundary()
top  = TopBoundary()
left.mark(boundaries, 1)
top.mark(boundaries, 2)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Dirichlet BC (fixed left edge)
zero = Constant((0.0, 0.0))
bc_left = DirichletBC(V, zero, left)

# -------------------------------------------------
# Traction on top edge
t_val = Constant((0.0, -2000.0))   # N/m (negative y‑direction)

# -------------------------------------------------
# Variational problem
u = TrialFunction(V)
v = TestFunction(V)

a = inner(sigma(epsilon(u)), epsilon(v))*dx
L = dot(t_val, v)*ds(2)   # only top edge contributes

# -------------------------------------------------
# Solve
u_sol = Function(V, name="Displacement")
solve(a == L, u_sol, bcs=[bc_left])

# -------------------------------------------------
# Save solution in XDMF
with XDMFFile(mesh.mpi_comm(), "plate_displacement.xdmf") as xdmf:
    xdmf.write(u_sol)

# -------------------------------------------------
# Plot vertical displacement u_y
uy = u_sol.sub(1)   # second component
# Create a regular grid for plotting
vertex_values = uy.compute_vertex_values(mesh)
x = mesh.coordinates()[:,0]
y = mesh.coordinates()[:,1]

# Triangulation for tripcolor
triangles = mesh.cells()
plt.figure(figsize=(8,3))
tpc = plt.tripcolor(x, y, triangles, vertex_values, shading='gouraud', cmap='viridis')
plt.colorbar(tpc, label=r'$u_y$ (m)')
plt.title('Vertical displacement $u_y$')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.axis('equal')
plt.tight_layout()
plt.savefig("q2_uy.png", dpi=300)
plt.close()