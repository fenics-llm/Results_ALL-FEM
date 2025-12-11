# filename: q8_orthotropic_elasticity.py
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt
from dolfin import *
import numpy as np

# -------------------------------------------------
# Geometry and mesh
Lx, Ly = 1.0, 0.20
nx, ny = 50, 25
mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny, "crossed")

# -------------------------------------------------
# Orthotropic material (plane stress) rotated by 30°
theta = np.radians(30.0)
c, s = np.cos(theta), np.sin(theta)

E1, E2 = 40e9, 10e9          # Pa
G12 = 5e9                     # Pa
nu12 = 0.25
nu21 = nu12*E2/E1

Q11 = E1/(1 - nu12*nu21)
Q22 = E2/(1 - nu12*nu21)
Q12 = nu12*E2/(1 - nu12*nu21)
Q66 = G12

# Local reduced stiffness (Voigt: 11,22,12)
Q_loc = np.array([[Q11, Q12, 0.0],
                  [Q12, Q22, 0.0],
                  [0.0, 0.0, Q66]])

# Rotation matrix for strain/stress (global → material)
R = np.array([[ c*c, s*s, 2*c*s],
              [ s*s, c*c, -2*c*s],
              [ -c*s, c*s, c*c - s*s]])

# Global stiffness in global axes
C_glob = R.T @ Q_loc @ R          # 3×3 matrix
C_tensor = as_matrix(C_glob.tolist())       # UFL Constant matrix (3×3)

# -------------------------------------------------
# Function space (vector displacement)
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# -------------------------------------------------
# Bottom Dirichlet BC (fixed)
tol = 1E-14
def bottom(x, on_boundary):
    return on_boundary and near(x[1], 0.0, tol)
bc_bottom = DirichletBC(V, Constant((0.0, 0.0)), bottom)
bcs = [bc_bottom]

# -------------------------------------------------
# Traction on top edge (downward)
traction = Constant((0.0, -10e6))   # -10 MPa

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], Ly, tol)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
TopBoundary().mark(boundaries, 1)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# -------------------------------------------------
# Strain and stress (orthotropic)
def epsilon(u):
    return sym(grad(u))

def sigma(u):
    eps = epsilon(u)
    eps_voigt = as_vector([eps[0,0], eps[1,1], 2*eps[0,1]])   # xx, yy, xy
    sig_voigt = C_tensor * eps_voigt
    return as_tensor([[sig_voigt[0], sig_voigt[2]],
                      [sig_voigt[2], sig_voigt[1]]])

# -------------------------------------------------
# Variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(sigma(u), epsilon(v))*dx
L = dot(traction, v)*ds(1)

# -------------------------------------------------
# Solve
u_sol = Function(V)
solve(a == L, u_sol, bcs)

# -------------------------------------------------
# Von Mises stress
s = sigma(u_sol)
s_xx, s_yy, s_xy = s[0,0], s[1,1], s[0,1]
von_mises = sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3*s_xy**2)

# -------------------------------------------------
# Project for output
V_scalar = FunctionSpace(mesh, "Lagrange", 2)
ux_proj = project(u_sol[0], V_scalar)
vm_proj = project(von_mises, V_scalar)
TFS = TensorFunctionSpace(mesh, "Lagrange", 1)
s_proj = project(s, TFS)

# -------------------------------------------------
# XDMF output
xdmf = XDMFFile(mesh.mpi_comm(), "q8_solution.xdmf")
xdmf.parameters["flush_output"] = True
xdmf.parameters["functions_share_mesh"] = True
xdmf.write(u_sol, 0.0)
xdmf.write(vm_proj, 0.0)   # von Mises as scalar field
xdmf.write(s_proj, 0.0)
xdmf.close()

# -------------------------------------------------
# Plotting
plt.figure()
p = plot(ux_proj, title="Horizontal displacement ux (m)", cmap="viridis")
plt.colorbar(p)
plt.savefig("q8_ux.png", dpi=300)

plt.figure()
p = plot(vm_proj, title="Von Mises stress (Pa)", cmap="viridis")
plt.colorbar(p)
plt.savefig("q8_vm.png", dpi=300)

print("Finished: ux, von Mises, and XDMF solution saved.")