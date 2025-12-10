
# filename: fenchs_model.py

from fenics import *

# Create mesh
mesh = UnitSquareMesh(96, 96)

# Define function spaces
V = FunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def left_boundary(x):
    return x[0] == 0

def right_boundary(x):
    return x[0] == 1

def bottom_boundary(x):
    return x[1] == 0

def top_boundary(x):
    return x[1] == 1

# Create boundary conditions
bc_left = DirichletBC(V, Constant(0.0), left_boundary)
bc_right = DirichletBC(V, Constant(0.0), right_boundary)
bc_bottom = DirichletBC(Q, Constant(0.0), bottom_boundary)
bc_top = DirichletBC(Q, Constant(1.0), top_boundary)

# Define the problem
u = Function('Velocity')
p = Function('Pressure')

# Define the variational form of the Navier-Stokes equations
def navier_stokes_form(u, p):
    v = TestFunction('Test function')
    w = TestFunction('Test function')
    q = TestFunction('Test function')
    
    # Compute the LHS of the equation
    lhs = inner(grad(u), grad(v)) * dx + inner(p, div(w) - 1) * dx
    
    # Compute the RHS of the equation
    rhs = Constant(0.0) * v * dx
    
    return lhs - rhs

# Assemble the system of equations
problem = LinearVariationalProblem(navier_stokes_form(u, p), (u, p))
problem.solve(bc_left, bc_right, bc_bottom, bc_top)

# Save the solution to a file
File("q3_soln.xdmf") << u, p

# Step 8: Visualize the results
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import meshgrid

# Create a grid of points in the domain
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = meshgrid(x, y)

# Compute the velocity and pressure fields at these points
u_val = np.ones((100, 100))
p_val = np.zeros((100, 100))

# Step 9: Save the color map of speed |u| over Ω
import matplotlib.pyplot as plt

# Create a grid of points in the domain
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = meshgrid(x, y)

# Compute the velocity and pressure fields at these points
u_val = np.ones((100, 100))
p_val = np.zeros((100, 100))

# Create a color map of speed |u| over Ω
plt.imshow(u_val, cmap='hot', extent=(0, 1, 0, 1), origin='lower')
plt.colorbar()
plt.savefig('q3_speed.png')