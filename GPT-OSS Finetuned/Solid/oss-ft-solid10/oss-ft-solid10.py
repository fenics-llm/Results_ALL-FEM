# filename: linear_elasticity_plate.py
import dolfin as dl
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend
import matplotlib.pyplot as plt

# ---------------------------------
# Geometry and mesh
Lx, Ly = 1.0, 0.20
nx, ny = 100, 20
mesh = dl.RectangleMesh(dl.Point(0.0, 0.0), dl.Point(Lx, Ly), nx, ny, "crossed")

# ---------------------------------
# Material parameters (plane stress)
nu = 0.49

# Spatially varying Young's modulus: E(y) = 100 GPa * (1 + y/0.20)
E = 5e6

# Lame parameters as Expressions
lambda_expr = dl.Constant(E*nu/((1+nu)*(1-2*nu)))
mu_expr     = dl.Constant(E/(2*(1+nu)))

# ---------------------------------
# Function space (Taylor–Hood not needed, pure elasticity)
V = dl.VectorFunctionSpace(mesh, "Lagrange", 2)

# ---------------------------------
# Boundary markers for Neumann traction
boundaries = dl.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Left(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[0], 0.0, dl.DOLFIN_EPS)

class Right(dl.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and dl.near(x[0], Lx, dl.DOLFIN_EPS)

left  = Left()
right = Right()
left.mark(boundaries, 1)
right.mark(boundaries, 2)

ds = dl.Measure("ds", domain=mesh, subdomain_data=boundaries)

# ---------------------------------
# Dirichlet BC on left edge (u = 0)
zero = dl.Constant((0.0, 0.0))
bc_left = dl.DirichletBC(V, zero, left)
bc_right = dl.DirichletBC(V, dl.Constant((0.03, 0.0)), right)

# ---------------------------------
# Traction on right edge: (2e6, 0) Pa per unit thickness
traction = dl.Constant((2e6, 0.0))

# ---------------------------------
# Variational formulation
u = dl.TrialFunction(V)
v = dl.TestFunction(V)

def epsilon(u):
    return dl.sym(dl.grad(u))

def sigma(u):
    return lambda_expr*dl.tr(epsilon(u))*dl.Identity(2) + 2.0*mu_expr*epsilon(u)

a = dl.inner(sigma(u), epsilon(v))*dl.dx
L = dl.dot(dl.Constant((0.0, 0.0)), v)*dl.dx

# ---------------------------------
# Solve
u_sol = dl.Function(V)
dl.solve(a == L, u_sol, [bc_left, bc_right], solver_parameters={"linear_solver": "mumps"})

# ---------------------------------
# Post-processing: displacement magnitude
V_scalar = dl.FunctionSpace(mesh, "Lagrange", 2)
u_magnitude = dl.project(dl.sqrt(dl.dot(u_sol, u_sol)), V_scalar)

# ---------------------------------
# Save magnitude as PNG
plt.figure(figsize=(8, 3))
mappable = dl.plot(u_magnitude, cmap="viridis")
plt.colorbar(mappable, label=r"$|u|$ (m)")
plt.title("Displacement magnitude")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.tight_layout()
plt.savefig("q10_disp.png", dpi=300)

# ---------------------------------
# Save full displacement field in XDMF
with dl.XDMFFile(mesh.mpi_comm(), "q10_disp.xdmf") as xdmf:
    xdmf.write(u_sol)

print("Simulation completed. Results saved as q10_disp.png and q10_disp.xdmf.")