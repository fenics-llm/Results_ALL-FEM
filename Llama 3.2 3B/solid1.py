# filename: femics.py
import numpy as np
from fenics import *

# Constants
E = 200e9  # Young's modulus (Pa)
nu = 0.3   # Poisson's ratio
Lx, Ly = 1.0, 0.2  # Lengths of the rectangular plate

# Create a mesh with 20 × 4 subdivisions across (x, y)
mesh = UnitSquareMesh(40, 10)

# Define the boundary conditions
def left_bc(x, on_boundary):
    return on_boundary and x[0] == 0

def right_bc(x, on_boundary):
    return on_boundary and x[0] == 1

def top_bc(x, on_boundary):
    return on_boundary and x[1] == Ly/2

def bottom_bc(x, on_boundary):
    return on_boundary and x[1] == 0

bc_left = DirichletBC(mesh, Constant(0), left_bc)
bc_right = DirichletBC(mesh, Constant(0.001), right_bc)
bc_top = NeumannBC(mesh, Constant(0), top_bc)
bc_bottom = NeumannBC(mesh, Constant(0), bottom_bc)

# Define the material properties
material = LinearElasticMaterial(E=E, Poisson=nu)

### Step 3: Set Up the Variational Formulation

# Define the variational formulation
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
L = inner(F, v)*dx

# Assemble the bilinear form and linear functional
M = assemble(a)
F = assemble(L)

# Solve the system numerically using FEniCS's solver
u_solution = Function(V)
solve(M, u_solution, bc_left, bc_right, bc_top, bc_bottom)

### Step 4: Post-process the Solution

# Save the displacement field in XDMF format
xdmf_file = File("displacement.xdmf")
xdmf_file << u_solution