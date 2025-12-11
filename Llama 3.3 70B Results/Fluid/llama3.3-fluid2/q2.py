from fenics import *

# Create mesh
L = 2.0
H = 0.20
mesh = RectangleMesh(Point(0, 0), Point(L, H), 120, 12)

# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define trial and test functions
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)

# Define parameters
mu = 1.0
rho = 1.0
f = Constant((1.0, 0.0))

# Define boundary conditions
def boundary(x, on_boundary):
    return on_boundary

bc_u = DirichletBC(V, Constant((0, 0)), boundary)

# Define variational problem
F = mu * inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx + inner(div(u), q) * dx - inner(f, v) * dx

# Compute solution
u = Function(V)
p = Function(Q)

solve(F == 0, [u, p], [bc_u])

# Save solution to file
vtkfile_u = File('q2_velocity.pvd')
vtkfile_p = File('q2_pressure.pvd')
vtkfile_u << u
vtkfile_p << p

# Save speed to file
speed = sqrt(u[0]**2 + u[1]**2)
vtkfile_speed = File('q2_speed.pvd')
vtkfile_speed << speed

# Save solution to xdmf file
xdmffile = XDMFFile('q2_solution.xdmf')
xdmffile.write(u, 0)
xdmffile.write(p, 0)

# Plot speed
import matplotlib.pyplot as plt
import numpy as np

speed_values = speed.compute_vertex_values(mesh)
speed_min = min(speed_values)
speed_max = max(speed_values)

plt.figure(figsize=(10, 2))
plt.imshow(np.array([speed_values]).reshape(12, 120), cmap='viridis', extent=[0, L, 0, H], vmin=speed_min, vmax=speed_max, origin='lower')
plt.colorbar(label='Speed')
plt.title('Speed')
plt.savefig('q2_speed.png')