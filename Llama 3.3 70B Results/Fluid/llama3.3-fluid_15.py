# filename: navier_stokes.py
from fenics import *

# Define parameters
rho = 1.0         # density
nu = 1e-3         # kinematic viscosity
T = 1.0           # total time
dt = 0.001        # time step

# Create mesh and finite element space
mesh = UnitSquareMesh(64, 64)
V = VectorFunctionSpace(mesh, 'P', 2)

# Define boundary conditions
u0 = Expression(('sin(2*pi*x[0])*cos(2*pi*x[1])', '-cos(2*pi*x[0])*sin(2*pi*x[1])'), degree=2)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant((0, 0))
p = FunctionSpace(mesh, 'P', 1)

# Split the system into two separate equations
a_u = rho * inner(u, v) * dx + dt * nu * inner(grad(u), grad(v)) * dx - dt * f[0] * v[0] * dx - dt * f[1] * v[1] * dx

# Define the time-stepping scheme
u_n = interpolate(u0, V)
t = 0
while t <= T:
    # Compute tentative velocity
    u_tentative = Function(V)
    a_u = rho * inner(u, v) * dx + dt * nu * inner(grad(u), grad(v)) * dx - dt * f[0] * v[0] * dx - dt * f[1] * v[1] * dx
    L = inner(u_n, v) * dx 
    solve(a_u == L, u_tentative)

    # Update the solution
    u_n.assign(u_tentative)

    # Save solution to file
    if round(t, 2) in [0.0, 0.25, 0.5, 1.0]:
        vtkfile = File('velocity_%s.pvd' % str(round(t, 2)))
        vtkfile << u_n

    # Move to next time step
    t += dt

# Save final solution to file in XDMF format
xdmffile_u = XDMFFile("velocity.xdmf")
xdmffile_u.write(u_n, t)