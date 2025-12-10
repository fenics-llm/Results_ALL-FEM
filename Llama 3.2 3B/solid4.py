from dolfin import *

# Create a mesh for the domain
mesh = UnitSquareMesh(100, 20)

# Create a function space for the displacement field
V = FunctionSpace(mesh, 'P', 2)

# Define the boundary conditions
u_D = Constant((0.0, 0.0))  # Left edge (x = 0)
u_R = Constant((1.0, 0.0))  # Right edge (x = 1.0)
u_top_bottom = Constant((0.0, 0.0))  # Top and bottom boundaries
u_hole_left = Constant((0.0, 0.0))  # Left hole boundary
u_hole_right = Constant((0.0, 0.0))  # Right hole boundary

# Define the subdomains
left = MeshFunction('double', mesh, 'left')
right = MeshFunction('double', mesh, 'right')
top_bottom = MeshFunction('double', mesh, 'top_bottom')
hole_left = MeshFunction('double', mesh, 'hole_left')
hole_right = MeshFunction('double', mesh, 'hole_right')

# Define the boundary conditions
bc_left = DirichletBC(V, u_D, left)
bc_right = DirichletBC(V, u_R, right)
bc_top_bottom = DirichletBC(V, u_top_bottom, top_bottom)
bc_hole_left = DirichletBC(V, u_hole_left, hole_left)
bc_hole_right = DirichletBC(V, u_hole_right, hole_right)

# Define the problem
u = TrialFunction(V)
v = TestFunction(V)

# Compute the weak form of the equation
a = inner(grad(u), grad(v))*dx

# Apply the boundary conditions
F = -u*v*dx + bc_left*u*v*dx + bc_right*u*v*dx + bc_top_bottom*u*v*dx + bc_hole_left*u*v*dx + bc_hole_right*u*v*dx

# Solve the linear system
solve(a == F, u, 'CG')

# Compute the maximum von Mises stress at the hole boundary and the Stress Concentration Factor (Kt = σ_max/2 MPa)
max_stress = max(von_mises(stress.split(1)))
Kt = max_stress / 2e6
print("Maximum von Mises stress:", max_stress, "MPa")
print("Stress Concentration Factor (Kt):", Kt, "MPa")

# Visualize the results using matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(u[0].plot_data(mesh=True), cmap='viridis')
plt.title("Displacement Field")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(von_mises(stress.split(1)[0]).plot_data(mesh=True), cmap='plasma')
plt.title("Von Mises Equivalent Stress")
plt.colorbar()

plt.show()