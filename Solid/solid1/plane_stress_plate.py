# filename: plane_stress_plate.py
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Material parameters
E = 200e9          # Young's modulus (Pa)
nu = 0.30          # Poisson's ratio

# Plane stress elasticity tensor
mu = E/(2.0*(1.0+nu))
lmbda = E*nu/((1.0+nu)*(1.0-2.0*nu))   # not used directly for plane stress
# Plane stress constitutive matrix (Voigt notation)
def sigma(eps):
    # eps is a 2x2 symmetric tensor
    eps_xx = eps[0,0]
    eps_yy = eps[1,1]
    eps_xy = eps[0,1]
    factor = E/(1.0 - nu**2)
    s_xx = factor*(eps_xx + nu*eps_yy)
    s_yy = factor*(nu*eps_xx + eps_yy)
    s_xy = factor*((1.0 - nu)/2.0)*2.0*eps_xy
    return as_tensor([[s_xx, s_xy],
                      [s_xy, s_yy]])

# Mesh
nx, ny = 20, 4
mesh = RectangleMesh(Point(0.0, 0.0), Point(1.0, 0.20), nx, ny, "crossed")

# Function space (Q1 vector)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Boundary definitions
tol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and near(x[0], 0.0, tol)

def right_boundary(x, on_boundary):
    return on_boundary and near(x[0], 1.0, tol)

# Dirichlet BCs
zero = Constant((0.0, 0.0))
bc_left = DirichletBC(V, zero, left_boundary)

ux_right = 0.001  # prescribed horizontal displacement (m)
prescribed = Constant((ux_right, 0.0))
bc_right = DirichletBC(V, prescribed, right_boundary)

bcs = [bc_left, bc_right]

# Variational problem
u = TrialFunction(V)
v = TestFunction(V)

def epsilon(u):
    return sym(grad(u))

a = inner(sigma(epsilon(u)), epsilon(v))*dx
L = dot(Constant((0.0, 0.0)), v)*dx   # no body forces

# Solve
u_sol = Function(V, name="Displacement")
solve(a == L, u_sol, bcs)

# Save solution to XDMF
xdmf_file = XDMFFile(mesh.mpi_comm(), "plate_displacement.xdmf")
xdmf_file.write(u_sol)
xdmf_file.close()

# Plot horizontal displacement u_x
ux, uy = u_sol.split(deepcopy=True)
# Create a triangulation for plotting
mesh_coords = mesh.coordinates()
triangles = mesh.cells()

fig, ax = plt.subplots()
tpc = ax.tripcolor(mesh_coords[:,0], mesh_coords[:,1], triangles, ux.vector().get_local(),
                   shading='gouraud', cmap='viridis')
fig.colorbar(tpc, ax=ax, label=r'$u_x$ (m)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title('Horizontal displacement $u_x$')
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('q1_ux.png', dpi=300)
plt.close()