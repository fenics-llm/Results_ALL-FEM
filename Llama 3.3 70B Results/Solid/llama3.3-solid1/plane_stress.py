# filename: plane_stress.py
from fenics import *

# Define geometry and mesh
Lx, Ly = 1.0, 0.20
nx, ny = 20, 4
mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), nx, ny)

# Define material properties
E = 200e9  # Young's modulus in Pa
nu = 0.30  # Poisson's ratio

# Define boundary conditions
u_L = Constant((0, 0))  # fixed displacement on left edge
u_R = Constant((0.001, 0))  # prescribed displacement on right edge

def left_boundary(x, on_boundary):
    return near(x[0], 0) and on_boundary

def right_boundary(x, on_boundary):
    return near(x[0], Lx) and on_boundary

# Define function space
V = VectorFunctionSpace(mesh, 'P', 1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# Define stress and strain tensors
def sigma(u, E, nu):
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    return 2 * mu * sym(grad(u)) + lmbda * tr(sym(grad(u))) * Identity(2)

def epsilon(v):
    return sym(grad(v))

a = inner(sigma(u, E, nu), epsilon(v)) * dx
l = inner(Constant((0, 0)), v) * ds

# Apply boundary conditions
bc_L = DirichletBC(V, u_L, left_boundary)
bc_R = DirichletBC(V, u_R, right_boundary)

# Compute solution
u_sol = Function(V)
solve(a == l, u_sol, [bc_L, bc_R])

# Save horizontal displacement to file
ux = u_sol.sub(0)
vtkfile = File('q1_ux.pvd')
vtkfile << ux

# Save displacement field to XDMF file
xdmffile = XDMFFile('displacement.xdmf')
xdmffile.write(u_sol, 0)

# Plot color map of horizontal displacement
import matplotlib.pyplot as plt
plot(ux, title='Horizontal Displacement')
plt.savefig('q1_ux.png')

print("TERMINATE")