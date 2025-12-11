# filename: q6_elasticity.py
from dolfin import *
import mshr
import matplotlib.pyplot as plt

# -------------------------------------------------
# Geometry
Lx, Ly = 1.0, 0.20          # plate dimensions (m)
a = 0.05                    # notch radius (m)
center = Point(0.5, Ly)     # center of the full circle

# Rectangle domain
rectangle = mshr.Rectangle(Point(0.0, 0.0), Point(Lx, Ly))

# Full circle (only the part inside the rectangle will be removed,
# i.e. the lower semicircle – the notch)
circle = mshr.Circle(center, a, 64)

# Domain with the semicircular notch removed
domain = rectangle - circle

# Generate mesh
mesh = mshr.generate_mesh(domain, 64)

# -------------------------------------------------
# Function space (quadratic Lagrange for displacement)
V = VectorFunctionSpace(mesh, "Lagrange", degree=2)

# -------------------------------------------------
# Material parameters (plane‑stress)
E  = 200e9          # Pa
nu = 0.30
coeff = E / (1.0 - nu**2)
D = coeff * as_matrix([[1.0,    nu,          0.0],
                       [nu,     1.0,         0.0],
                       [0.0,    0.0, (1.0 - nu)/2.0]])

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    eps = epsilon(u)
    eps_vec = as_vector([eps[0, 0], eps[1, 1], 2.0*eps[0, 1]])
    sig_vec = D * eps_vec
    return as_tensor([[sig_vec[0], sig_vec[2]],
                      [sig_vec[2], sig_vec[1]]])

# -------------------------------------------------
# Boundary definitions
tol = 1e-8

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)

class TopTraction(SubDomain):
    def inside(self, x, on_boundary):
        # top edge except the notch interval [0.45,0.55] (a = 0.05)
        return (on_boundary and near(x[1], Ly, tol) and
                (x[0] < 0.5 - a - tol or x[0] > 0.5 + a + tol))

# Mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
bottom = Bottom()
top_trac = TopTraction()
bottom.mark(boundaries, 1)
top_trac.mark(boundaries, 2)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Dirichlet condition on the bottom edge (fixed)
zero = Constant((0.0, 0.0))
bc_bottom = DirichletBC(V, zero, bottom)

# -------------------------------------------------
# Variational problem
u = TrialFunction(V)
v = TestFunction(V)

a_form = inner(sigma(u), epsilon(v)) * dx
traction = Constant((0.0, -10e6))          # 10 MPa downward traction
L_form = dot(traction, v) * ds(2)         # apply on marked top edge (2)

# Solve linear elasticity system
u_sol = Function(V, name="Displacement")
solve(a_form == L_form, u_sol, bc_bottom,
      solver_parameters={"linear_solver": "mumps"})

# -------------------------------------------------
# Post‑processing: von Mises stress (plane‑stress)
s = sigma(u_sol)          # stress tensor expression
sxx = s[0, 0]
syy = s[1, 1]
sxy = s[0, 1]

von_mises_expr = sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)

V_vm = FunctionSpace(mesh, "DG", 0)   # scalar discontinuous space
von_mises = project(von_mises_expr, V_vm)
von_mises.rename("von_Mises", "von_Mises")

# -------------------------------------------------
# Save von Mises plot
plt.figure(figsize=(8, 3))
p = plot(von_mises, title="Von Mises Stress (Pa)", cmap="viridis")
plt.colorbar(p)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.savefig("q6_vm.png", dpi=300)
plt.close()

# -------------------------------------------------
# Save displacement field (XDMF)
xdmf_file = XDMFFile(mesh.mpi_comm(), "q6_displacement.xdmf")
xdmf_file.write(u_sol)
xdmf_file.close()