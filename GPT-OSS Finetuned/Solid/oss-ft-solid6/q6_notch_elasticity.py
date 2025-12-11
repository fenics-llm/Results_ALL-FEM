# filename: q6_notch_elasticity.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dolfin import *
from mshr import Rectangle, Circle, generate_mesh

# -------------------------------------------------
# Geometry and mesh
a = 0.05
domain = Rectangle(Point(0.0, 0.0), Point(1.0, 0.20)) - Circle(Point(0.5, 0.20), a)
mesh = generate_mesh(domain, 80)   # increase resolution if needed

# -------------------------------------------------
# Material parameters (plane stress)
E  = 200e9          # Pa
nu = 0.30
mu = E/(2.0*(1.0+nu))
lmbda = E*nu/((1.0+nu)*(1.0-nu))

# -------------------------------------------------
# Function space (Taylor–Hood not needed for elasticity)
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# -------------------------------------------------
# Boundary markers
tol = 1E-6
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, tol)
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.20, tol) and (x[0] < 0.45 - tol or x[0] > 0.55 + tol)
class NotchArc(SubDomain):
    def inside(self, x, on_boundary):
        # points on the circular arc (distance a from centre) and y<0.20
        return on_boundary and near((x[0]-0.5)**2 + (x[1]-0.20)**2, a**2, tol) and x[1] < 0.20 + tol

Bottom().mark(boundaries, 1)
Top().mark(boundaries, 2)
NotchArc().mark(boundaries, 3)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Boundary conditions
zero = Constant((0.0, 0.0))
bc_bottom = DirichletBC(V, zero, boundaries, 1)

bcs = [bc_bottom]

# -------------------------------------------------
# Variational problem
u = TrialFunction(V)
v = TestFunction(V)

def epsilon(u):
    return sym(grad(u))

def sigma(u):
    return lmbda*tr(epsilon(u))*Identity(2) + 2.0*mu*epsilon(u)

T = Constant((0.0, -10e6))   # traction on top (Pa)

a_form = inner(sigma(u), epsilon(v))*dx
L_form = dot(T, v)*ds(2)      # only on top edge (excluding notch)

# -------------------------------------------------
# Solve
u_sol = Function(V, name="Displacement")
solve(a_form == L_form, u_sol, bcs, solver_parameters={"linear_solver":"mumps"})

# -------------------------------------------------
# Von Mises stress (plane stress)
s = sigma(u_sol)
s_xx = s[0,0]
s_yy = s[1,1]
s_xy = s[0,1]

von_mises = sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3.0*s_xy**2)
V_scalar = FunctionSpace(mesh, "Lagrange", 2)
von_mises_proj = project(von_mises, V_scalar, solver_type="mumps")
von_mises_proj.rename("von_Mises", "von_Mises")

# -------------------------------------------------
# Save displacement (XDMF)
with XDMFFile(mesh.mpi_comm(), "displacement.xdmf") as xdmf:
    xdmf.write(u_sol)

# -------------------------------------------------
# Plot and save von Mises map
plt.figure(figsize=(8,3))
p = plot(von_mises_proj, title="Von Mises stress (Pa)", cmap="viridis")
plt.colorbar(p)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q6_vm.png", dpi=300)
plt.close()