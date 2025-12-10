# filename: femics.py

from fenics import *

# Constants
E = 50000000  # Young's modulus in Pa
nu = 0.49      # Poisson's ratio
a = 0.04       # Radius of the circular hole in m
Lx, Ly = 1.0, 0.2  # Length and width of the rectangular strip in m

# Create a mesh for the geometry
mesh = RectangleMesh(Point(0, 0), Point(Lx, 0), Point(Lx, Ly), Point(0, Ly))

# Define the boundary conditions
def left_bc(x):
    return DirichletBC(V, Constant(0), x[0])

def right_bc(x):
    return DirichletBC(V, Constant(0.001), x[0])

def top_bc(x):
    return DirichletBC(V, Constant(0), x[1])

def bottom_bc(x):
    return DirichletBC(V, Constant(0), x[1])

def hole_bc(x):
    return DirichletBC(V, Constant(0), x[1] - a)

bc_left = left_bc
bc_right = right_bc
bc_top = top_bc
bc_bottom = bottom_bc
bc_hole = hole_bc

# Define the variational problem
u = Function(V)
v = TestFunction(V)
sigma = Expression('E*nu*(u[0]*dx[0] + u[1]*dx[1])', u=u, dx=dx)

problem = LinearVariationalProblem(L, sigma, bc_left, bc_right, bc_top, bc_bottom, bc_hole, v=v)

### Step 3: Assemble the Stiffness Matrix and Right-Hand Side Vector
# Create a function to assemble the stiffness matrix and right-hand side vector
def assemble_problem():
    global problem
    problem.assemble()
    return problem

assemble_problem()

### Step 4: Solve the Problem and Compute the von Mises Equivalent Stress
# Create a function to solve the problem and compute the von Mises equivalent stress
def solve_and_compute_stress():
    global problem
    u = Function(V)
    problem.solve(u)
    sigma = Expression('sqrt((sigma[0, 0] + sigma[1, 1])^2 - (sigma[0, 1] - sigma[1, 0])^2)/2', sigma=sigma)
    return sigma

sigma = solve_and_compute_stress()

### Step 5: Save the von Mises Equivalent Stress as a Color Map and Horizontal Displacement
# Create a function to save the von Mises equivalent stress as a color map and horizontal displacement
def save_results():
    global sigma, u
    from matplotlib import cm
    import numpy as np
    
    # Save the von Mises equivalent stress as a color map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(sigma.array(), cmap=cm.get_cmap('viridis'))
    ax.set_title('Von Mises Equivalent Stress')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig('q11_vm.png', bbox_inches='tight')

    # Save the horizontal displacement as a color map
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(u.array()[:, :, 0], cmap=cm.get_cmap('viridis'))
    ax.set_title('Horizontal Displacement')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig('q11_ux.png', bbox_inches='tight')

    # Save the resulting displacement field in XDMF format
    from xdmf import XDMFWriter
    writer = XDMFWriter()
    writer.write(u, 'displacement.xdmf')

save_results()