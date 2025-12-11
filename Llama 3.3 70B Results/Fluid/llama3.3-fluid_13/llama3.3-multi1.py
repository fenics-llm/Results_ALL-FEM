from fenics import *

# Define parameters
L = 1.0         # Length of the channel
H = 0.10        # Height of the channel
U_max = 0.75    # Maximum velocity

# Create mesh and define function space
mesh = RectangleMesh(Point(0, 0), Point(L, H), 100, 10)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
u_D = Constant(0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)  # Source term
a = inner(grad(u), grad(v))*dx + U_max*u*v*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u

# Check Peclet number (for convection-diffusion problems)
h_min = mesh.hmin()
D = 0.01 # Diffusion coefficient
Pe = U_max * h_min / (2*D)
if Pe > 1:
    print("Peclet number is large, stabilization scheme may be required.")
else:
    print("Peclet number is small, no stabilization scheme is required.")

# Plot solution
import matplotlib.pyplot as plt

plot(u)
plt.show()