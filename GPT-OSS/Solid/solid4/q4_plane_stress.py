# filename: q4_plane_stress.py
from dolfin import *
import mshr
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Geometry
Lx, Ly = 1.0, 0.20
a = 0.04
hole_centers = [Point(0.33, 0.10), Point(0.67, 0.10)]

domain = mshr.Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
for c in hole_centers:
    domain = domain - mshr.Circle(c, a)

mesh = mshr.generate_mesh(domain, 80)  # adjust resolution if needed

# -------------------------------------------------
# Function spaces
V = VectorFunctionSpace(mesh, "Lagrange", degree=2)

# -------------------------------------------------
# Material parameters (plane stress)
E  = 200e9          # Pa
nu = 0.30
mu = E / (2.0*(1.0 + nu))
lmbda = E*nu / (1.0 - nu**2)   # plane‑stress Lame parameter

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda*tr(epsilon(u))*Identity(2) + 2.0*mu*epsilon(u)

# -------------------------------------------------
# Boundary definitions
tol = 1E-14

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], Lx, tol)

left_boundary  = LeftBoundary()
right_boundary = RightBoundary()

# Mark boundaries for Neumann term
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
left_boundary.mark(boundaries, 1)
right_boundary.mark(boundaries, 2)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Boundary conditions
zero = Constant((0.0, 0.0))
bc_left = DirichletBC(V, zero, left_boundary)

# -------------------------------------------------
# Variational problem
u = TrialFunction(V)
v = TestFunction(V)

t = Constant((2.0e6, 0.0))          # traction on the right edge (Pa)

a = inner(sigma(u), epsilon(v))*dx
L = dot(t, v)*ds(2)                # apply only on right edge (marker 2)

# Solve
u_sol = Function(V, name="Displacement")
solve(a == L, u_sol, bc_left, solver_parameters={"linear_solver":"mumps"})

# -------------------------------------------------
# Post‑processing: von Mises stress
W = FunctionSpace(mesh, "DG", 0)   # piecewise constant for stress plot
sigma_tensor = sigma(u_sol)

sigma_xx = sigma_tensor[0, 0]
sigma_yy = sigma_tensor[1, 1]
sigma_xy = sigma_tensor[0, 1]

von_mises_expr = sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3.0*sigma_xy**2)
von_mises = project(von_mises_expr, W, solver_type='cg')

# Save displacement field
xdmf_file = XDMFFile(mesh.mpi_comm(), "q4_displacement.xdmf")
xdmf_file.write(u_sol)
xdmf_file.close()

# -------------------------------------------------
# Plot von Mises stress and save figure
plt.figure(figsize=(8, 3))
p = plot(von_mises, title="Von Mises stress (Pa)", cmap="viridis")
plt.colorbar(p)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q4_vm.png", dpi=300)
plt.close()

# -------------------------------------------------
# Compute maximum von Mises stress (expected at hole boundary)
max_vm = von_mises.vector().get_local().max()
Kt = max_vm / 2.0e6

print(f"Maximum von Mises stress: {max_vm/1e6:.2f} MPa")
print(f"Stress concentration factor Kt = σ_max / 2 MPa = {Kt:.3f}")
