from dolfin import *
from ufl import *

# Define the mesh
mesh = UnitSquareMesh(100, 100)

# Define the function space
V = FunctionSpace(mesh, 'P', 2)

# Define the boundary conditions
def left_bc(x):
    return DirichletBC(V, Constant(0.0), 'left')

def right_bc(x):
    return DirichletBC(V, Constant(0.03), 'right')

def top_bc(x):
    return DirichletBC(V, Constant(0.0), 'top')

# Define the weak form of the linear elasticity equation with pressure
def weak_form(u, p):
    u_x, u_y = u.split()
    a = inner(grad(u), grad(p)) * dx + inner(p, 0.1) * dx
    L = inner(p, 1) * dx

    # Assemble the stiffness matrix and load vector
    M = assemble(a)
    b = assemble(L)

    # Solve the system of equations
    solve(M, u, b)
    return a, L

# Create a linear system of equations
u, p = Function(V), Function(Q)

a, L = weak_form(u, p)

M = assemble(a)
b = assemble(L)

solve(M, u, b)

### Step 2: Define the boundary conditions and solve the problem

# Apply the boundary conditions
bc_left = left_bc()
bc_right = right_bc()
bc_top = top_bc()

u_left = bc_left(u)
u_right = bc_right(u)
u_top = bc_top(u)

# Solve the system of equations
solve(M, u, b)

### Step 3: Save the displacement field in XDMF format

# Save the displacement field in XDMF format
XDMFFile('displacement.xdmf').write(u)

### Step 4: Create a color map of displacement magnitude and save it as an image

# Create a color map of displacement magnitude
import matplotlib.pyplot as plt

plt.imshow(np.abs(u), cmap='viridis')
plt.colorbar(label='Displacement Magnitude')
plt.savefig('q10_disp.png')