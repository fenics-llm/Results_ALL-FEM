# filename: main.py
from dolfin import *
import mshr
import matplotlib.pyplot as plt

# -------------------------------------------------
# Geometry
Lx, Ly = 1.0, 0.20
a = 0.04
center = Point(0.5, 0.10)

rect = mshr.Rectangle(Point(0.0, 0.0), Point(Lx, Ly))
hole = mshr.Circle(center, a, 64)
domain = rect - hole
mesh = mshr.generate_mesh(domain, 64)   # increase resolution if needed

# -------------------------------------------------
# Material parameters (plane strain)
E  = 5.0e6          # Pa
nu = 0.49
mu = E / (2.0*(1.0+nu))
lmbda = E*nu/((1.0+nu)*(1.0-2.0*nu))

# -------------------------------------------------
# Function spaces (mixed displacement–pressure)
V = VectorElement("CG", mesh.ufl_cell(), 2)   # displacement
Q = FiniteElement("CG", mesh.ufl_cell(), 1)   # pressure (Lagrange multiplier)
W_elem = MixedElement([V, Q])
W = FunctionSpace(mesh, W_elem)

# -------------------------------------------------
# Boundary conditions
tol = 1E-14

def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0, tol)

def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], Lx, tol)

# Fixed left edge (u = 0)
bc_left = DirichletBC(W.sub(0), Constant((0.0, 0.0)), left_boundary)

# Prescribed displacement on right edge (u_x = 0.001, u_y = 0)
bc_right = DirichletBC(W.sub(0), Constant((0.001, 0.0)), right_boundary)

# Pressure null‑space removal: fix pressure at a single point (e.g., (0,0))
def pressure_point(x, on_boundary):
    return near(x[0], 0.0, tol) and near(x[1], 0.0, tol)

bc_pressure = DirichletBC(W.sub(1), Constant(0.0), pressure_point, method="pointwise")

bcs = [bc_left, bc_right, bc_pressure]

# -------------------------------------------------
# Mixed variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

def epsilon(v):
    return sym(grad(v))

# Weak form (mixed displacement–pressure)
a = (2.0*mu*inner(epsilon(u), epsilon(v)))*dx \
    - p*div(v)*dx \
    + q*div(u)*dx

L = Constant((0.0, 0.0))
L = inner(L, v)*dx   # zero body force

# -------------------------------------------------
# Solve
w = Function(W)
solve(a == L, w, bcs,
      solver_parameters={"linear_solver": "mumps"})

(u_h, p_h) = w.split()

# -------------------------------------------------
# Post‑processing: von Mises stress
sigma = 2.0*mu*epsilon(u_h) - p_h*Identity(2)   # Cauchy stress
s = sigma - (1./3)*tr(sigma)*Identity(2)       # deviatoric part
von_mises = sqrt(3./2*inner(s, s))

V_vm = FunctionSpace(mesh, "CG", 1)
vm_proj = project(von_mises, V_vm)

# -------------------------------------------------
# Plot and save von Mises stress
plt.figure()
p = plot(vm_proj, title="Von Mises stress (Pa)", cmap="viridis")
plt.colorbar(p)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.tight_layout()
plt.savefig("q11_vm.png", dpi=300)

# -------------------------------------------------
# Plot and save horizontal displacement u_x
V_ux = FunctionSpace(mesh, "CG", 1)
ux_proj = project(u_h[0], V_ux)

plt.figure()
p = plot(ux_proj, title=r"$u_x$ (m)", cmap="viridis")
plt.colorbar(p)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.tight_layout()
plt.savefig("q11_ux.png", dpi=300)

# -------------------------------------------------
# Save displacement field (vector) to XDMF
xdmf_file = XDMFFile(mesh.mpi_comm(), "displacement.xdmf")
xdmf_file.write(u_h)
xdmf_file.close()

print("Simulation completed.")
print(" - von Mises stress saved as q11_vm.png")
print(" - horizontal displacement saved as q11_ux.png")
print(" - displacement field saved as displacement.xdmf")