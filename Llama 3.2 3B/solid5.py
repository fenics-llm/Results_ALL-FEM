from fenics import Interval, Rectangle, DirichletBoundaryCondition, NeumannBoundaryCondition, assemble, inner, dot, solve, Function, TestFunction, XDMFFile, von_mises, Epsilon
import matplotlib.pyplot as plt

# Define the geometry of the plate and the notch
plate = Rectangle((0, 0), (1.0, 0.20))
notch = Interval(0.08, 0.12)

# Specify the boundary conditions and loads
u_x_bc = DirichletBoundaryCondition((0, 0))
u_y_bc = DirichletBoundaryCondition((0, 0))

top_bc = NeumannBoundaryCondition(0)
bottom_bc = NeumannBoundaryCondition(0)

# Define the variational problem
u = Function('VectorField', (0.0, 1.0), (0.0, 0.20))
v = TestFunction('VectorField', (0.0, 1.0), (0.0, 0.20))

# Assemble the stiffness matrix and load vector
K = assemble(inner(grad(u), grad(v))*200*Epsilon(plate)*dx)
F = assemble(dot(n, sigma*v)*dx)

# Specify the boundary conditions
u_x_bc.apply(u_x)
u_y_bc.apply(u_y)

# Solve the system of equations
u_solution = solve(K*u + F, u, 'CG', 10)

# Compute the von Mises equivalent stress
sigma_vm = von_mises(sigma, u_solution)

# Create an XDMF file
file = XDMFFile('displacement.xdmf')

# Write the solution to the file
file << u_solution

# Close the file
file.close()

# Create a color map
plt.imshow(sigma_vm, cmap='viridis')
plt.colorbar()

# Save the color map as an image file
plt.savefig('q5_vm.png')

# Close the plot
plt.close()
