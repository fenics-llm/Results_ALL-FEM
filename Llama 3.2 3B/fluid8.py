filename: navier_stokes.py

from fenics import *

# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(1, 0.20), 128, 32)

# Define boundary conditions
def boundary_x(x):
    return x[0]

def boundary_y(x):
    if x[1] == 0:
        return 0
    elif x[1] == 0.20:
        return 0

bc_y0 = DirichletBC(Mesh, Constant(0), boundary_y)
bc_yH = DirichletBC(Mesh, Constant(0), boundary_y)

# Define the Navier-Stokes equations
u = Function('VectorField', (mesh.u, mesh.v))
p = Function('ScalarField')

def navier_stokes(u, p):
    v = u.grad
    f = Constant((1, 0))  # body force

    eqn_u_x = inner(f[0], v[0]) - mu * div(v[0])
    eqn_u_y = inner(f[1], v[1]) - mu * div(v[1])

    return eqn_u_x + eqn_u_y, p

# Assemble the equations
u_x, u_y, p = Function('VectorField', (u.x, u.y, p))
eqn_u_x, eqn_u_y, eqn_p = navier_stokes(u_x, u_y)

# Apply boundary conditions
bc_u_x = DirichletBC(Mesh, Constant(0), boundary_x)
bc_u_y = bc_yH

# Solve the equations
solve(eqn_u_x == 0, u_x, bc_u_x)
solve(eqn_u_y == 0, u_y, bc_u_y)

# Compute pressure
p = solve(eqn_p == 0, p)

# Save velocity and pressure fields to XDMF file
File('q9_soln.xdmf') << u_x, u_y, p
```

### Step 3: Execution

To execute the code, save it in a file named `navier_stokes.py` and run it using Python:

