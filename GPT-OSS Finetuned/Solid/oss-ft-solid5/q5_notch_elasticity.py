# filename: q5_notch_elasticity.py
from dolfin import *
import mshr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------
# Geometry: rectangle (0,1)×(0,0.2) with a rectangular notch
# --------------------------------------------------------------
rect   = mshr.Rectangle(Point(0.0, 0.0), Point(1.0, 0.20))
notch  = mshr.Rectangle(Point(0.0, 0.08), Point(0.06, 0.12))
domain = rect - notch
mesh   = mshr.generate_mesh(domain, 120)   # mesh resolution, increase if needed

# --------------------------------------------------------------
# Function space (vector, quadratic)
# --------------------------------------------------------------
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# --------------------------------------------------------------
# Material parameters (plane stress)
# --------------------------------------------------------------
E  = 200e9          # Pa
nu = 0.30
mu = E/(2.0*(1.0+nu))
lmbda = E*nu/((1.0+nu)*(1.0-nu))   # plane‑stress Lamé constant

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda*tr(epsilon(u))*Identity(2) + 2.0*mu*epsilon(u)

# --------------------------------------------------------------
# Boundary conditions
# --------------------------------------------------------------
tol = 1E-8

class LeftBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.0, tol)

left_bc = DirichletBC(V, Constant((0.0, 0.0)), LeftBoundary())

class RightBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 1.0, tol)

right_boundary = RightBoundary()
traction = Constant((2e6, 0.0))   # (2 MPa·m, 0)

# --------------------------------------------------------------
# Mark boundaries for ds
# --------------------------------------------------------------
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
right_boundary.mark(boundaries, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# --------------------------------------------------------------
# Variational problem
# --------------------------------------------------------------
u = TrialFunction(V)
v = TestFunction(V)

a = inner(sigma(u), epsilon(v))*dx
L = dot(traction, v)*ds(1)

# --------------------------------------------------------------
# Solve
# --------------------------------------------------------------
u_sol = Function(V)
solve(a == L, u_sol, left_bc)

# --------------------------------------------------------------
# Von Mises stress
# --------------------------------------------------------------
s = sigma(u_sol) - (1./3)*tr(sigma(u_sol))*Identity(2)   # deviatoric part
von_mises = sqrt(3./2*inner(s, s))

V0 = FunctionSpace(mesh, "Lagrange", 2)
vm_proj = project(von_mises, V0)

# --------------------------------------------------------------
# Save displacement (XDMF)
# --------------------------------------------------------------
xdmf = XDMFFile(mesh.mpi_comm(), "q5_displacement.xdmf")
xdmf.write(u_sol)
xdmf.close()

# --------------------------------------------------------------
# Plot and save von Mises map
# --------------------------------------------------------------
plt.figure(figsize=(8, 3))
p = plot(vm_proj, title=r"Von Mises stress (Pa)", cmap="viridis")
plt.colorbar(p)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q5_vm.png", dpi=300)
plt.close()