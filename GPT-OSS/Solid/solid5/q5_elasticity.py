# filename: q5_elasticity.py
from dolfin import *
import mshr
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# Geometry: rectangle (0,0)-(1.0,0.20) with notch cut out
# -------------------------------------------------
rect = mshr.Rectangle(Point(0.0, 0.0), Point(1.0, 0.20))
notch = mshr.Rectangle(Point(0.0, 0.08), Point(0.06, 0.12))
domain = rect - notch

# Mesh resolution (increase for finer results)
mesh = mshr.generate_mesh(domain, 64)

# -------------------------------------------------
# Function space
# -------------------------------------------------
V = VectorFunctionSpace(mesh, "CG", 2)

# -------------------------------------------------
# Material parameters (plane stress)
# -------------------------------------------------
E  = 200e9          # Pa
nu = 0.30
mu = E/(2.0*(1.0+nu))
lmbda = E*nu/((1.0+nu)*(1.0-2.0*nu))

# Plane stress constitutive matrix
coeff = E/(1.0 - nu**2)
D = as_matrix([[coeff,      coeff*nu, 0.0],
               [coeff*nu,   coeff,    0.0],
               [0.0,        0.0,      coeff*(1.0-nu)/2.0]])

# -------------------------------------------------
# Boundary definitions
# -------------------------------------------------
tol = 1E-8

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0, tol)

left_boundary  = LeftBoundary()
right_boundary = RightBoundary()

# Mark boundaries for ds integration
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
left_boundary.mark(boundaries, 1)
right_boundary.mark(boundaries, 2)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Variational problem
# -------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    eps = epsilon(u)
    # Voigt notation: [e_xx, e_yy, 2e_xy]
    eps_voigt = as_vector([eps[0,0], eps[1,1], 2*eps[0,1]])
    sig_voigt = dot(D, eps_voigt)
    # Return as tensor
    return as_tensor([[sig_voigt[0], sig_voigt[2]],
                      [sig_voigt[2], sig_voigt[1]]])

# Traction on the right edge (2 MPa in x-direction)
t = Constant((2e6, 0.0))

a = inner(sigma(u), epsilon(v))*dx
L = dot(t, v)*ds(2)

# Dirichlet BC on left edge (zero displacement)
zero = Constant((0.0, 0.0))
bc = DirichletBC(V, zero, left_boundary)

# Solve
u_sol = Function(V, name="Displacement")
solve(a == L, u_sol, bc, solver_parameters={"linear_solver": "mumps"})

# -------------------------------------------------
# Post‑processing: von Mises stress
# -------------------------------------------------
# Compute stress tensor
S = sigma(u_sol)

# Extract components
sigma_xx = S[0,0]
sigma_yy = S[1,1]
sigma_xy = S[0,1]

# von Mises formula for plane stress
von_mises_expr = sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3*sigma_xy**2)

V_scalar = FunctionSpace(mesh, "CG", 2)
von_mises = project(von_mises_expr, V_scalar, solver_type="cg")

# -------------------------------------------------
# Save results
# -------------------------------------------------
# Displacement to XDMF
xdmf_file = XDMFFile(mesh.mpi_comm(), "q5_displacement.xdmf")
xdmf_file.write(u_sol)
xdmf_file.close()

# Plot von Mises and save as PNG
plt.figure(figsize=(8,4))
p = plot(von_mises, title="Von Mises stress (Pa)", cmap="viridis")
plt.colorbar(p)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q5_vm.png", dpi=300)
plt.close()