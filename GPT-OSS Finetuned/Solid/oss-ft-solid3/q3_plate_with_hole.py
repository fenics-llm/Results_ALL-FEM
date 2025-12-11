# filename: q3_plate_with_hole.py
import dolfin as dl
import mshr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------
# Geometry: rectangle (0,1)×(0,0.20) with a hole
# -------------------------------------------------
rect = mshr.Rectangle(dl.Point(0.0, 0.0), dl.Point(1.0, 0.20))
hole = mshr.Circle(dl.Point(0.5, 0.10), 0.05, 64)
domain = rect - hole
mesh = mshr.generate_mesh(domain, 80)   # increase resolution if needed

# -------------------------------------------------
# Function space (vector Lagrange, degree 2)
# -------------------------------------------------
V = dl.VectorFunctionSpace(mesh, "Lagrange", 2)

# -------------------------------------------------
# Plane‑stress material parameters
# -------------------------------------------------
E  = 200e9          # Pa
nu = 0.30
mu = E/(2.0*(1.0 + nu))
lmbda = E*nu/((1.0 + nu)*(1.0 - nu))   # plane stress

def epsilon(u):
    return dl.sym(dl.grad(u))

def sigma(u):
    eps = epsilon(u)
    return lmbda*dl.tr(eps)*dl.Identity(2) + 2.0*mu*eps

# -------------------------------------------------
# Boundary markers
# -------------------------------------------------
tol = 1e-8
boundaries = dl.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Left(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[0], 0.0, tol)

class Right(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[0], 1.0, tol)

class HoleBoundary(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near((x[0] - 0.5)**2 + (x[1] - 0.10)**2,
                                 0.05**2, 1e-6)

Left().mark(boundaries, 1)
Right().mark(boundaries, 2)
HoleBoundary().mark(boundaries, 3)

ds = dl.Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Dirichlet BC on left edge (u = 0)
# -------------------------------------------------
zero = dl.Constant((0.0, 0.0))
bc_left = dl.DirichletBC(V, zero, boundaries, 1)

# -------------------------------------------------
# Neumann traction on right edge: (2 MPa, 0)
# -------------------------------------------------
t_val = dl.Constant((2e6, 0.0))

# -------------------------------------------------
# Variational problem
# -------------------------------------------------
u = dl.TrialFunction(V)
v = dl.TestFunction(V)

a = dl.inner(sigma(u), epsilon(v))*dl.dx
L = dl.dot(t_val, v)*ds(2)   # only right edge

# -------------------------------------------------
# Solve
# -------------------------------------------------
u_sol = dl.Function(V, name="Displacement")
dl.solve(a == L, u_sol, bc_left, solver_parameters={"linear_solver": "mumps"})

# -------------------------------------------------
# Von Mises stress (plane stress)
# -------------------------------------------------
s = sigma(u_sol)
S = dl.TensorFunctionSpace(mesh, "DG", 0)
s_proj = dl.project(s, S)

s_xx = s_proj[0, 0]
s_yy = s_proj[1, 1]
s_xy = s_proj[0, 1]

von_mises = dl.sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3.0*s_xy**2)
von_mises_proj = dl.project(von_mises, dl.FunctionSpace(mesh, "DG", 0))

# -------------------------------------------------
# Save displacement (XDMF)
# -------------------------------------------------
with dl.XDMFFile(mesh.mpi_comm(), "q3_plate_with_hole_disp.xdmf") as xdmf:
    xdmf.write(u_sol)

# -------------------------------------------------
# Plot von Mises and save PNG
# -------------------------------------------------
plt.figure(figsize=(8, 3))
c = dl.plot(von_mises_proj, title=r"Von Mises stress (Pa)", cmap="viridis")
plt.colorbar(c)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q3_vm.png", dpi=300)

# -------------------------------------------------
# Max von Mises on hole boundary and Stress Concentration Factor
# -------------------------------------------------
# Build a BoundaryMesh of the exterior facets
bmesh = dl.BoundaryMesh(mesh, "exterior", order=False)
bcoords = bmesh.coordinates()

# Identify points on the hole (distance ≈ 0.05 from centre)
center = np.array([0.5, 0.10])
dist = np.sqrt(((bcoords - center)**2).sum(axis=1))
hole_pts = bcoords[np.abs(dist - 0.05) < 1e-6]

# Evaluate von Mises at those points (loop because Function expects scalars)
vm_vals = np.array([von_mises_proj(p) for p in hole_pts])
max_vm_hole = np.max(vm_vals)

print("Maximum von Mises stress on hole boundary: {:.2f} MPa".format(max_vm_hole / 1e6))
print("Stress Concentration Factor Kt = σ_max / 2 MPa = {:.3f}".format(max_vm_hole / 2e6))